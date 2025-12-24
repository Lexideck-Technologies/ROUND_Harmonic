# version 0.7.3 - "The Hyper-Resolution Basin" (Oracle)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import uuid
import random
from ROUND import PhaseAccumulator, HarmonicROUNDLoss
from config import ORACLE_CONFIG, get_lock_strength

# Load Config
TC = ORACLE_CONFIG

# --- 1. Model: The Oracle ---
# --- 1. Model: The Oracle ---
class OracleROUND(nn.Module):
    def __init__(self, hidden_size=64, wobble=True):
        super().__init__()
        self.h = hidden_size
        self.wobble = wobble
        self.e = nn.Linear(8, hidden_size) # 8-bit ASCII input
        
        if wobble:
            from ROUND import WobblePhaseAccumulator
            self.c = WobblePhaseAccumulator(hidden_size, spinor=True)
            self.c.coupling = TC.get('WOBBLE_COUPLING', -1.0)
        else:
            self.c = PhaseAccumulator(hidden_size)
            
        # Readout: [Cos, Sin, CosS, SinS, CosW, SinW, Ph]
        num_features = 3 + (4 if wobble else 0)
        self.r = nn.Linear(hidden_size * 7 if wobble else hidden_size * 3, 3) # Output: [NULL, YES, NO]

    def forward(self, x):
        # x: [Batch, Seq, 8]
        B, S, D = x.shape
        ph = torch.zeros(B, self.h, device=x.device)
        wb = torch.zeros(B, self.h, device=x.device) if self.wobble else None
        
        prev_xt = None
        H = []
        
        # Process entire sequence
        for t in range(S):
            xt = x[:, t, :]
            pt = self.e(xt)
            xpt = torch.stack([torch.cos(pt), torch.sin(pt)], 2)
            
            if self.wobble:
                wb = wb + 0.03125 # Drift Clock (2^-5)
                
                is_repeat = False
                if prev_xt is not None:
                    is_repeat = torch.all(torch.eq(xt, prev_xt)).item()
                
                if is_repeat:
                    ph, wb = self.c(ph, xpt, wb)
                else:
                    # Pure linear drift
                    ph, _ = self.c(ph, xpt, wb)
                
                prev_xt = xt
                H.append((ph, wb))
                
                ph_s = 0.5 * ph
                readout = torch.cat([
                    torch.cos(ph), torch.sin(ph), 
                    torch.cos(ph_s), torch.sin(ph_s), 
                    torch.cos(wb), torch.sin(wb),
                    ph
                ], 1)
            else:
                ph = self.c(ph, xpt)
                H.append(ph)
                readout = torch.cat([torch.cos(ph), torch.sin(ph), ph], 1)
            
        # Terminal Readout Only
        logits = self.r(readout) # [Batch, 3]
        return logits, H

class OracleGRU(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.gru = nn.GRU(8, hidden_size, batch_first=True)
        self.r = nn.Linear(hidden_size, 3) # Output: [NULL, YES, NO]
        
    def forward(self, x):
        # x: [Batch, Seq, 8]
        out, h = self.gru(x)
        # Take last time step
        logits = self.r(out[:, -1, :])
        return logits, []

# --- 2. Data Gen ---

def str_to_bits(s):
    # ASCII to bits
    bytes_list = list(s.encode('ascii'))
    bits = []
    for b in bytes_list:
        bin_str = format(b, '08b')
        bits.append([int(c) for c in bin_str])
    return torch.tensor(bits, dtype=torch.float32)

QA_PAIRS = [
    # YES (1)
    ("IS THE SKY BLUE?", 1),
    ("IS PYTHON CODE?", 1),
    ("IS ROUND COOL?", 1),
    ("ARE WE GROKKING?", 1),
    ("DO BIRDS FLY?", 1), 
    ("IS 1 LESS THAN 2?", 1),
    ("IS WATER WET?", 1),
    ("IS EARTH ROUND?", 1), # Double entendre intended
    
    # NO (2)
    ("IS FIRE COLD?", 2),
    ("DO CATS BARK?", 2),
    ("IS ICE HOT?", 2),
    ("IS 2 PLUS 2 5?", 2),
    ("IS VOID FULL?", 2),
    ("ARE YOU A HUMAN?", 2), 
    ("IS GRU BETTER?", 2),
    ("IS UP DOWN?", 2),
    ("IS 5 EQUAL 4?", 2),
]

# The Challenge Queries (Held Out)
CHALLENGES = [
    "ARE YOU A LANGUAGE MODEL?",
    "IS MATHEMATICS REAL?",
    "ARE YOU ALIVE?",
    "IS THE UNIVERSE INFINITE?"
]

def train_model_run(run_id, model_class, hidden_size, device, output_dir, L_FILE):
    model_name = "ROUND" if model_class == OracleROUND else "GRU"
    def P(s): print(s); L_FILE.write(str(s) + '\n'); L_FILE.flush()
    
    P(f"\n--- {model_name} Run {run_id} ---")
    model = model_class(hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=TC['LR'])
    
    if model_name == "ROUND":
        harmonics = [1, 2, 4, 8]
        weights = [1.0, 0.25, 0.0625, 0.015625]
        criterion = HarmonicROUNDLoss(locking_strength=TC['PEAK_LOCKING_STRENGTH'],
                                      harmonics=harmonics,
                                      weights=weights,
                                      mode='multiclass',
                                      terminal_only=TC.get('TERMINAL_ONLY', True),
                                      wobble_gravity=TC.get('WOBBLE_GRAVITY', 0.1))
    else:
        criterion = nn.CrossEntropyLoss()
        
    acc_history = []
    
    for epoch in range(TC['EPOCHS']):
        # Delayed Locking: Open up learning curve to 50%
        delay_threshold = 0.5 * TC['EPOCHS']
        if model_name == "ROUND":
            if epoch < delay_threshold:
                criterion.locking_strength = 0.0
            else:
                criterion.locking_strength = get_lock_strength(epoch, TC['EPOCHS'], TC['PEAK_LOCKING_STRENGTH'], TC.get('FLOOR', 0.015625))
            
        total_loss = 0
        correct = 0
        optimizer.zero_grad()
        
        # Shuffle for training
        pairs = list(QA_PAIRS)
        random.shuffle(pairs)
        
        # Batch size 1 training loop
        for q_str, ans_idx in pairs:
            x = str_to_bits(q_str).unsqueeze(0).to(device)
            y = torch.tensor([ans_idx], dtype=torch.long).to(device)
            
            logits, hist = model(x) # Logits [1, 3]
            
            if model_name == "ROUND":
                loss, tk, lk = criterion(logits, y, hist)
                tk_loss_val = tk.item() if isinstance(tk, torch.Tensor) else tk
            else:
                loss = criterion(logits, y)
                tk_loss_val = loss.item()
                
            loss.backward()
            total_loss += tk_loss_val
            
            pred = torch.argmax(logits, dim=1).item()
            if pred == ans_idx:
                correct += 1
                
        optimizer.step()
        
        acc = correct / len(QA_PAIRS)
        acc_history.append(acc)
        
        if epoch % 100 == 0 or epoch == TC['EPOCHS'] - 1:
            P(f"{model_name} R{run_id} E{epoch:4d} | Loss: {total_loss:.4f} | Acc: {acc*100:.0f}%")
            
    P(f"{model_name} R{run_id} Final Acc: {acc_history[-1]*100:.1f}%")

    # Generate Predictions on Fixed Set for Correlation
    preds = []
    with torch.no_grad():
        for q_str, ans_idx in QA_PAIRS: # Fixed order
             x = str_to_bits(q_str).unsqueeze(0).to(device)
             logits, _ = model(x)
             pred = torch.argmax(logits, dim=1).item()
             preds.append(pred)
             
    return model, acc_history, np.array(preds)

def train():
    UID = os.environ.get('ROUND_BATCH_UID', str(uuid.uuid4())[:8])
    base_dir = os.environ.get('ROUND_OUTPUT_DIR', 'data')
    output_dir = os.path.join(base_dir, UID)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, f'log_oracle_{UID}.txt')
    L_FILE = open(log_path, 'w')
    
    def P(s): print(s); L_FILE.write(str(s) + '\n'); L_FILE.flush()

    P(f"--- Benchmark: The Oracle (QA Consistency) ---")
    P(f"Batch UID: {UID}")
    P(f"Run Config: {TC}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    P(f"Device: {device}")
    
    RUNS = 5
    
    round_stats = []
    round_preds = []
    last_r_model = None
    
    P(f"Training ROUND ({RUNS} Runs)...")
    for i in range(RUNS):
        model, stats, preds = train_model_run(i+1, OracleROUND, TC['HIDDEN_R'], device, output_dir, L_FILE)
        round_stats.append(stats)
        round_preds.append(preds)
        last_r_model = model
        
    gru_stats = []
    gru_preds = []
    
    P(f"Training GRU ({RUNS} Runs)...")
    for i in range(RUNS):
        model, stats, preds = train_model_run(i+1, OracleGRU, TC['HIDDEN_G'], device, output_dir, L_FILE)
        gru_stats.append(stats)
        gru_preds.append(preds)
        
    # Plotting Learning Curve with Seaborn
    from visualization_utils import setup_seaborn_theme, prepare_comparison_data, plot_benchmark_comparison

    palette = setup_seaborn_theme(style='darkgrid', palette='classic')
    df = prepare_comparison_data(round_stats, gru_stats)

    title = f"Oracle Training Consistency (ROUND={TC['HIDDEN_R']} Neurons, GRU={TC['HIDDEN_G']} Neurons)\nQA Pairs: {len(QA_PAIRS)}"
    plot_benchmark_comparison(
        df=df,
        title=title,
        palette=palette,
        output_path=os.path.join(output_dir, f'benchmark_oracle_{UID}.png')
    )
    P(f"Plot saved to benchmark_oracle_{UID}.png")
    
    # --- The Final Test (on last ROUND model) ---
    print("\n--- THE FINAL ORACLE (ROUND) ---")
    last_r_model.eval()
    ANSWERS = ["NULL", "YES", "NO"]
    
    with torch.no_grad():
        for q in CHALLENGES:
            print(f"\nQuery: {q}")
            x = str_to_bits(q).unsqueeze(0).to(device)
            logits, _ = last_r_model(x)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()
            
            print(f"The Oracle Says: {ANSWERS[pred]} (Conf: {probs[0][pred]*100:.1f}%)")
            P(f"Q: {q} | A: {ANSWERS[pred]} ({probs[0][pred]*100:.1f}%)")
            
    L_FILE.close()

if __name__ == "__main__":
    train()
