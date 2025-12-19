# version 0.6.3 - "The Density Duel" (ASCII)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import uuid
from config import ASCII_CONFIG, get_lock_strength

# Load Config
TC = ASCII_CONFIG
# --- 1. Generative ROUND Model (Byte-Level) ---
from ROUND import PhaseAccumulator, HarmonicROUNDLoss

class GenerativeROUNDModel(nn.Module):
    def __init__(self, hidden_size=64, input_dim=8, output_dim=256):
        super().__init__()
        self.h = hidden_size
        # Input: 8 bits of the ASCII char
        self.e = nn.Linear(input_dim, hidden_size) 
        self.c = PhaseAccumulator(hidden_size)
        # Output: Probability of next ASCII char (256 classes)
        self.r = nn.Linear(hidden_size * 3, output_dim)

    def forward(self, x):
        # x: [Batch, Seq, 8] (Bits)
        B, S, D = x.shape
        ph = torch.zeros(B, self.h, device=x.device)
        
        # Store outputs for every step
        logits_seq = []
        hist_seq = []

        for t in range(S):
            # 1. Encode Input Bits -> Phase Drive
            xt = x[:, t, :] # [B, 8]
            pt = self.e(xt) # [B, H]
            
            # 2. Resonate (Recurrence)
            # Create phasors from the drive
            xpt = torch.stack([torch.cos(pt), torch.sin(pt)], 2)
            # Update Phase State
            ph = self.c(ph, xpt)
            
            # 3. Store History
            hist_seq.append(ph)
            
            # 4. Readout (Generate Next Char Prediction)
            # We use Cos, Sin, and raw Phase (Neuro-Symbolic)
            readout_features = torch.cat([torch.cos(ph), torch.sin(ph), ph], 1)
            logits = self.r(readout_features) # [B, 256]
            logits_seq.append(logits)

        # Stack sequences
        # logits_seq: [Batch, Seq, 256]
        logits_out = torch.stack(logits_seq, dim=1)
        
        return logits_out, hist_seq

# --- 2. GRU Baseline Model ---

class GenerativeGRUModel(nn.Module):
    def __init__(self, hidden_size=64, input_dim=8, output_dim=256):
        super().__init__()
        # Matches ROUND hidden size for fair comparison
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)
        
    def forward(self, x):
        # x: [Batch, Seq, 8]
        out, h = self.gru(x)
        # out: [Batch, Seq, Hidden]
        logits = self.fc(out)
        return logits, [] # No phase history for GRU

# --- 3. Data Preparation (HELLO WORLD) ---

def str_to_bits(s):
    bytes_list = list(s.encode('ascii'))
    bits = []
    for b in bytes_list:
        bin_str = format(b, '08b')
        bit_row = [int(c) for c in bin_str]
        bits.append(bit_row)
    return torch.tensor(bits, dtype=torch.float32)

def train_model(model_name, model_class, hidden_size, device, input_bits, target_tensor, loop_text, epochs, uid, output_dir, stats_list, L_FILE):
    def P(s):
        print(s)
        L_FILE.write(str(s) + '\n')
        L_FILE.flush()
        
    P(f"\n--- Training {model_name} ---")
    
    model = model_class(hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=TC['LR'])
    
    if model_name == "ROUND":
        criterion = HarmonicROUNDLoss(locking_strength=TC['PEAK_LOCKING_STRENGTH'],
                                      harmonics=TC['HARMONICS'],
                                      mode='multiclass',
                                      terminal_only=TC.get('TERMINAL_ONLY', False))
    else:
        criterion = nn.CrossEntropyLoss()
        
    acc_history = []
    
    for epoch in range(epochs):
        if model_name == "ROUND":
            criterion.locking_strength = get_lock_strength(epoch, epochs, TC['PEAK_LOCKING_STRENGTH'])
            
        optimizer.zero_grad()
        
        logits, hist = model(input_bits)
        
        logits_flat = logits.view(-1, 256)
        targets_flat = target_tensor.view(-1)
        
        if model_name == "ROUND":
            loss, tk_loss, lk_loss = criterion(logits_flat, targets_flat, hist)
            stats_loss = tk_loss
        else:
            loss = criterion(logits_flat, targets_flat)
            stats_loss = loss.item()
            lk_loss = 0.0
            
        loss.backward()
        optimizer.step()
        
        # Calculate Accuracy
        probs = torch.softmax(logits, dim=2)
        pred_indices = torch.argmax(probs, dim=2)
        correct = (pred_indices == target_tensor).float().sum()
        total = target_tensor.numel()
        acc = correct / total
        acc_history.append(acc.item())
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            pred_bytes = pred_indices[0].cpu().tolist()
            pred_str = "".join([chr(c) if 32 <= c <= 126 else '?' for c in pred_bytes])
            P(f"{model_name} E{epoch:4d} | Acc: {acc:.2f} | Loss: {stats_loss:.4f} | Out: '{pred_str}'")
            
    stats_list.append(acc_history)
    P(f"{model_name} Final Acc: {acc_history[-1]:.4f}")
    
    # Final Predictions (for Correlation)
    with torch.no_grad():
        logits, _ = model(input_bits) # [1, Seq, 256]
        pred_indices = torch.argmax(logits, dim=2)
    
    return model, pred_indices.cpu().numpy().flatten()

def train():
    UID = os.environ.get('ROUND_BATCH_UID', str(uuid.uuid4())[:8])
    output_dir = os.environ.get('ROUND_OUTPUT_DIR', 'data')
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, f'log_creative_ascii_{UID}.txt')
    L_FILE = open(log_path, 'w')
    
    def P(s): print(s); L_FILE.write(str(s) + '\n'); L_FILE.flush()

    P(f"--- Creative Benchmark: ASCII Generator (ROUND vs GRU) ---")
    P(f"Batch UID: {UID}")
    P(f"Run Config: {TC}")
    P(f"Task: Learn 'HELLO WORLD' + 21 Spaces sequence.")
    
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    TEXT = "HELLO WORLD" + " " * 21
    EPOCHS = TC['EPOCHS']
    
    input_bits = str_to_bits(TEXT).unsqueeze(0).to(device)
    targets_list = list(TEXT.encode('ascii'))
    targets_list = targets_list[1:] + [targets_list[0]]
    target_tensor = torch.tensor(targets_list, dtype=torch.long).unsqueeze(0).to(device)
    
    round_stats = []
    gru_stats = []
    round_preds = []
    
    RUNS = 5
    
    # Train ROUND
    P(f"Training ROUND ({RUNS} Runs)...")
    last_r_model = None
    for i in range(RUNS):
        r_model, p = train_model(f"ROUND_{i+1}", GenerativeROUNDModel, TC['HIDDEN_R'], device, input_bits, target_tensor, TEXT, EPOCHS, UID, output_dir, round_stats, L_FILE)
        round_preds.append(p)
        last_r_model = r_model
    
    # Train GRU
    P(f"Training GRU ({RUNS} Runs)...")
    for i in range(RUNS):
        g_model, p = train_model(f"GRU_{i+1}", GenerativeGRUModel, TC['HIDDEN_G'], device, input_bits, target_tensor, TEXT, EPOCHS, UID, output_dir, gru_stats, L_FILE)
    
    # Plotting
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rm, rs = np.mean(round_stats, 0), np.std(round_stats, 0)
    gm, gs = np.mean(gru_stats, 0), np.std(gru_stats, 0)
    ep = np.arange(len(rm))

    ax.fill_between(ep, rm-rs, rm+rs, color='#FF4B4B', alpha=0.1)
    ax.fill_between(ep, gm-gs, gm+gs, color='#4B4BFF', alpha=0.1)
    
    ax.plot(rm, color='#FF4B4B', linewidth=2, label='ROUND (Harmonic)')
    ax.plot(gm, color='#4B4BFF', linewidth=2, label='GRU (Standard)')
    
    ax.set_title(f"Sequence Generation Learning Curve (ROUND={TC['HIDDEN_R']} Neurons, GRU={TC['HIDDEN_G']} Neurons)\nTarget: 'HELLO WORLD' (32 Steps)", fontsize=14, color='white')
    ax.set_xlabel('Epochs', fontsize=12, color='gray')
    ax.set_ylabel('Accuracy (Next Char)', fontsize=12, color='gray')
    ax.grid(True, alpha=0.1)
    ax.legend()
    
    plt.savefig(os.path.join(output_dir, f'benchmark_ascii_{UID}.png'), dpi=300)
    P(f"Plot saved to benchmark_ascii_{UID}.png")
    
    L_FILE.close()

if __name__ == "__main__":
    train()
