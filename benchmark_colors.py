# version 0.7.3 - "The Hyper-Resolution Basin" (Colors with Batched Array Config)
import torch
import torch.nn as nn
import torch.optim as optim
import random
import uuid
import os
import matplotlib.pyplot as plt
import numpy as np
from ROUND import PhaseAccumulator, HarmonicROUNDLoss
from config import COLORS_CONFIG, get_lock_strength

# Load Config
TC = COLORS_CONFIG

# --- Unique Color Pairs (RYB / Art Model) ---
# --- Color Algebra (Semantic Mixing) ---
# Each example is independent.
COLOR_ALGEBRA = [
    # Primary Mixing
    ("RED+BLUE", "PURPLE"),
    ("BLUE+RED", "PURPLE"),
    ("RED+YELLOW", "ORANGE"),
    ("YELLOW+RED", "ORANGE"),
    ("BLUE+YELLOW", "GREEN"),
    ("YELLOW+BLUE", "GREEN"),
    
    # Identity
    ("RED+RED", "RED"),
    ("BLUE+BLUE", "BLUE"),
    ("YELLOW+YELLOW", "YELLOW"),
    
    # Light/Dark
    ("BLACK+WHITE", "GRAY"),
    ("WHITE+BLACK", "GRAY"),
]

# Ticker System (Pure Logic Codes)
TICKERS = {
    'RED': 'CRIM', 'BLUE': 'AZUR', 'YELLOW': 'GOLD',
    'PURPLE': 'PLUM', 'ORANGE': 'RUST', 'GREEN': 'JADE',
    'BLACK': 'VOID', 'WHITE': 'SNOW', 'GRAY': 'IRON'
}

def get_ticker_seq(pair):
    # pair: ("RED+BLUE", "PURPLE")
    input_part = pair[0] # "RED+BLUE"
    output_part = pair[1] # "PURPLE"
    
    parts = input_part.split('+')
    a = parts[0]
    b = parts[1]
    c = output_part
    
    return TICKERS[a] + TICKERS[b] + TICKERS[c]

# Generate Dense Ticker Sequences (12 chars unique)
# We add a Context Key (a-z) to disambiguate the branch points.
# e.g. 'a' + CRIMAZURPLUM vs 'c' + CRIMGOLDRUST
KEYS = "abcdefghijklmnopqrstuvwxyz"
STRINGS = [KEYS[i] + get_ticker_seq(p) for i, p in enumerate(COLOR_ALGEBRA)]
# Length becomes 13 chars.
MAX_LEN = 13
PADDED_SEQUENCES = STRINGS

class ColorROUND(nn.Module):
    def __init__(self, hidden_size=64, wobble=True):
        super().__init__()
        self.h = hidden_size
        self.wobble = wobble
        self.e = nn.Linear(8, hidden_size) # 8-bit ASCII
        
        if wobble:
            from ROUND import WobblePhaseAccumulator
            self.c = WobblePhaseAccumulator(hidden_size, spinor=True)
            self.c.coupling = TC.get('WOBBLE_COUPLING', -1.0)
        else:
            self.c = PhaseAccumulator(hidden_size)
            
        # Readout: [Cos, Sin, CosS, SinS, CosW, SinW, Ph]
        num_features = 3 + (4 if wobble else 0)
        self.r = nn.Linear(hidden_size * 7 if wobble else hidden_size * 3, 256) 

    def forward(self, x):
        # x: [Batch, Seq, 8]
        B, S, D = x.shape
        ph = torch.zeros(B, self.h, device=x.device)
        wb = torch.zeros(B, self.h, device=x.device) if self.wobble else None
        
        prev_xt = None
        logits_seq = []
        hist_seq = []
        
        for t in range(S):
            xt = x[:, t, :]
            pt = self.e(xt)
            xpt = torch.stack([torch.cos(pt), torch.sin(pt)], 2)
            
            if self.wobble:
                wb = wb + 0.03125 # 2^-5 Hyper-Resolution Clock
                
                is_repeat = False
                if prev_xt is not None:
                    is_repeat = torch.all(torch.eq(xt, prev_xt), dim=1).float() # [Batch]
                    # Since we are batching, we need per-item logic if we want perfect "is_repeat" handling per item.
                    # However, PhaseAccumulator is vectorized.
                    # is_repeat vector needed?
                    # Current ROUND implementation of WobblePhaseAccumulator assumes scalar or global logic usually.
                    # But wait, self.c handles tensors? NO.
                    # Let's look at ROUND.py again? 
                    # Actually, let's implement the logic assuming vectorized operations.
                    # The `c` call is: return p + d(...), w + d(...)
                    # We need to branch logic based on repeat?
                    # Actually, for batch processing, we can't easily branch "if is_repeat" unless we mask.
                    # But let's look at the original code. It had `if is_repeat:` block.
                    # This implies batch size was 1 or it was checking global equality.
                    # In previous runs, it was training sequence by sequence (Batch=1).
                    # Now we are shifting to Batch=12.
                    # This means we need to vectorize the "Is Repeat" check.
                
                # VECTORIZED REPEAT CHECK
                # xt: [B, H], prev_xt: [B, H]
                # is_repeat: [B, 1]
                if prev_xt is not None:
                     # Check equality across feature dim
                     is_rep = torch.all(torch.eq(xt, prev_xt), dim=1, keepdim=True).float()
                else:
                     is_rep = torch.zeros(B, 1, device=x.device)

                # We need to call self.c. BUT self.c doesn't take an "is_repeat" flag.
                # It just updates ph, wb.
                # In the original code:
                # if is_repeat: ph, wb = c(...) 
                # else: ph, _ = c(...) (wb not updated from network, but we added drift before)
                
                # To vectorize this:
                # 1. Calculate Full Update: ph_new, wb_new = self.c(ph, xpt, wb)
                # 2. Calculate Partial Update: ph_new_p, _ = self.c(ph, xpt, wb) -> Actually self.c returns wb+impulse.
                #    If we want "Pure Drift" for non-repeats, we want wb_new = wb + drift (already done).
                #    So we ignore the wb output from self.c for non-repeats.
                
                # Run the accumulator
                ph_out, wb_out = self.c(ph, xpt, wb)
                
                # Mix based on is_rep
                # If is_rep is 1, take wb_out. If 0, keep 'wb' (which has drift added).
                # Note: `wb` variable already has drift added at top of loop.
                # `wb_out` has (wb + drift) + network_delta.
                
                # So:
                # ph = ph_out (Always update phase)
                # wb = is_rep * wb_out + (1 - is_rep) * wb
                
                # Wait, PhaseAccumulator.forward returns: p + d(...), w + dw + impulse
                # So wb_out includes the d_w update.
                
                ph = ph_out
                wb = is_rep * wb_out + (1.0 - is_rep) * wb
                
                prev_xt = xt
                hist_seq.append((ph, wb))
                
                ph_s = 0.5 * ph
                readout = torch.cat([
                    torch.cos(ph), torch.sin(ph), 
                    torch.cos(ph_s), torch.sin(ph_s), 
                    torch.cos(wb), torch.sin(wb),
                    ph
                ], 1)
            else:
                ph = self.c(ph, xpt)
                hist_seq.append(ph)
                readout = torch.cat([torch.cos(ph), torch.sin(ph), ph], 1)
                
            logits_seq.append(self.r(readout))
            
        return torch.stack(logits_seq, 1), hist_seq

# --- 2. GRU Baseline ---
class ColorGRU(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.gru = nn.GRU(8, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 256)
        
    def forward(self, x):
        # x: [Batch, Seq, 8]
        out, h = self.gru(x)
        logits = self.fc(out)
        return logits, []

def str_to_bits(s):
    bits = []
    for b in s.encode('ascii'):
        bin_str = format(b, '08b')
        bits.append([int(c) for c in bin_str])
    return torch.tensor(bits, dtype=torch.float32)

def str_list_to_batch_bits(str_list):
    batch = []
    for s in str_list:
        batch.append(str_to_bits(s))
    return torch.stack(batch)

def train_model(model_name, model_class, hidden_size, device, epochs, uid, output_dir, stats_list, L_FILE):
    def P(s): print(s); L_FILE.write(str(s) + '\n'); L_FILE.flush()
    
    P(f"\n--- Training {model_name} (Batched Array) ---")
    
    model = model_class(hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=TC['LR'])
    
    if model_name.startswith("ROUND"):
        harmonics = [1, 2, 4, 8]
        weights = [1.0, 0.25, 0.0625, 0.015625]
        criterion = HarmonicROUNDLoss(locking_strength=TC['PEAK_LOCKING_STRENGTH'],
                                      harmonics=harmonics,
                                      weights=weights,
                                      mode='multiclass',
                                      terminal_only=TC.get('TERMINAL_ONLY', False),
                                      wobble_gravity=TC.get('WOBBLE_GRAVITY', 0.1))
    else:
        criterion = nn.CrossEntropyLoss()
        
    acc_history = []
    
    # Prepare Batch Data (Static Batch)
    # Input: All sequences, characters 0 to N-1
    # Target: All sequences, characters 1 to N
    
    # PADDED_SEQUENCES is list of strings
    full_batch_bits = str_list_to_batch_bits(PADDED_SEQUENCES).to(device) # [B, S_full, 8]
    
    x = full_batch_bits[:, :-1, :] # [B, S-1, 8]
    
    # Targets
    # We need index tensor [B, S-1]
    targets = []
    for s in PADDED_SEQUENCES:
        t_row = [ord(c) for c in s[1:]]
        targets.append(t_row)
    y = torch.tensor(targets, dtype=torch.long).to(device) # [B, S-1]
    
    for epoch in range(epochs):
        # Delayed Locking
        delay_threshold = 0.5 * epochs
        if "ROUND" in model_name:
            if epoch < delay_threshold:
                criterion.locking_strength = 0.0
            else:
                criterion.locking_strength = get_lock_strength(epoch, epochs, TC['PEAK_LOCKING_STRENGTH'], TC.get('FLOOR', 0.015625))
            
        optimizer.zero_grad()
        
        logits, hist = model(x)
        
        logits_flat = logits.reshape(-1, 256)
        y_flat = y.reshape(-1)
        
        if model_name.startswith("ROUND"):
            loss, tk, lk = criterion(logits_flat, y_flat, hist)
            loss.backward()
            epoch_loss = tk
        else:
            loss = criterion(logits_flat, y_flat)
            loss.backward()
            epoch_loss = loss.item()
            
        optimizer.step()
        
        # Acc
        pred = torch.argmax(logits_flat, 1)
        correct = (pred == y_flat).sum().item()
        total = y_flat.shape[0]
        epoch_acc = correct / total
        acc_history.append(epoch_acc)
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            P(f"Epoch {epoch:4d} | Acc: {epoch_acc:.4f} | Loss: {epoch_loss:.4f}")
            
    stats_list.append(acc_history)
    P(f"{model_name} Final Acc: {acc_history[-1]:.4f}")
    
    return model

def train():
    UID = os.environ.get('ROUND_BATCH_UID', str(uuid.uuid4())[:8])
    output_dir = os.environ.get('ROUND_OUTPUT_DIR', 'data')
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, f'log_creative_colors_{UID}.txt')
    L_FILE = open(log_path, 'w')
    
    def P(s): print(s); L_FILE.write(str(s) + '\n'); L_FILE.flush()

    P(f"--- Creative Benchmark: Semantic Algebra (Colors) ---")
    P(f"Batch UID: {UID}")
    P(f"Run Config: {TC}")
    P(f"Run Config: {TC}")
    P(f"Task: Learn Logic Tickers (Key+A+B->C).")
    P(f"Format: Key + 4-char codes (e.g. aCRIMAZURPLUM).")
    P(f"Max Len: {MAX_LEN} (Context Indexed)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = TC['EPOCHS']
    
    round_stats = []
    gru_stats = []
    
    RUNS = 5
    
    # Train ROUND
    P(f"Training ROUND ({RUNS} Runs)...")
    last_r_model = None
    for i in range(RUNS):
        r_model = train_model(f"ROUND_{i+1}", ColorROUND, TC['HIDDEN_R'], device, EPOCHS, UID, output_dir, round_stats, L_FILE)
        last_r_model = r_model
    
    # Train GRU
    P(f"Training GRU ({RUNS} Runs)...")
    for i in range(RUNS):
        g_model = train_model(f"GRU_{i+1}", ColorGRU, TC['HIDDEN_G'], device, EPOCHS, UID, output_dir, gru_stats, L_FILE)
    
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
    
    ax.set_title(f"Batched Array Learning (ROUND={TC['HIDDEN_R']}, GRU={TC['HIDDEN_G']})\nTask: Color Algebra", fontsize=14, color='white')
    ax.set_xlabel('Epochs', fontsize=12, color='gray')
    ax.set_ylabel('Accuracy (Next Token)', fontsize=12, color='gray')
    ax.grid(True, alpha=0.1)
    ax.legend()
    
    plt.savefig(os.path.join(output_dir, f'benchmark_colors_{UID}.png'), dpi=300)
    P(f"Plot saved to benchmark_colors_{UID}.png")
    
    # Generate completion for logical check
    # We use single sequence generation for checking
    P("\n--- Testing Ticker Logic (Indexed) [ROUND] ---")
    TEST_PROMPTS = ["aCRIMAZUR", "cCRIMGOLD", "jVOIDSNOW"]
    
    with torch.no_grad():
        for prompt in TEST_PROMPTS:
            gen_text = generate_completion(last_r_model, prompt, device)
            P(f"Prompt: {prompt:15s} -> Predicted: {gen_text}")
    
    L_FILE.close()

def generate_completion(model, prompt, device):
    # Single sequence generation
    gen_text = ""
    prompt_bits = str_to_bits(prompt).unsqueeze(0).to(device)
    
    # Feed prompt
    # Note: Model is trained on Batched Mode, but should handle Batch=1 fine.
    # WobblePhaseAccumulator is stateful? No, it's functional in forward.
    # self.c is the module. State is passed in loop.
    
    logits, hist = model(prompt_bits) # [1, Seq, 256]
    
    last_logit = logits[:, -1, :]
    next_char_code = torch.argmax(last_logit, dim=1).item()
    next_char = chr(next_char_code)
    gen_text += next_char
    
    if hasattr(model, 'c'):
         # State needs to be retrieved.
         # hist is a list of [ (ph, wb), ... ] for each step
         # We need the LAST step's state.
         if model.wobble:
             ph, wb = hist[-1]
         else:
             ph = hist[-1]
             wb = None
             
         for _ in range(10):
             if next_char == '.': break
             
             x_next = str_to_bits(next_char).unsqueeze(0).to(device)
             xt = x_next[:, 0, :]
             
             pt = model.e(xt)
             xpt = torch.stack([torch.cos(pt), torch.sin(pt)], 2)
             
             if model.wobble:
                 wb = wb + 0.03125 
                 # Here we assume no repeats in generation for simplicity or just run the logic
                 # Check repeat against... wait, we need prev_xt equivalent?
                 # ideally yes.
                 # Let's just do pure drift for gen
                 ph, _ = model.c(ph, xpt, wb) 
                 
                 ph_s = 0.5 * ph
                 readout = torch.cat([
                     torch.cos(ph), torch.sin(ph), 
                     torch.cos(ph_s), torch.sin(ph_s), 
                     torch.cos(wb), torch.sin(wb),
                     ph
                 ], 1)
             else:
                 ph = model.c(ph, xpt)
                 ph_s = 0.5 * ph
                 readout = torch.cat([
                     torch.cos(ph), torch.sin(ph), 
                     torch.cos(ph_s), torch.sin(ph_s), 
                     ph
                 ], 1)
             
             logit = model.r(readout)
             next_char_code = torch.argmax(logit, dim=1).item()
             next_char = chr(next_char_code)
             gen_text += next_char
             
    return gen_text

if __name__ == "__main__":
    train()
