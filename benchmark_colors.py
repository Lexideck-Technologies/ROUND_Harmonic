# version 0.6.3 - "The Density Duel" (Colors)
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

# Formatting as Prompt-Completion pairs
TRAINING_SEQUENCES = [f"{p[0]}={p[1]}." for p in COLOR_ALGEBRA]
# e.g. "RED+BLUE=PURPLE."

class ColorROUND(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.h = hidden_size
        self.e = nn.Linear(8, hidden_size) # 8-bit ASCII
        self.c = PhaseAccumulator(hidden_size)
        self.r = nn.Linear(hidden_size * 3, 256) # ASCII Character Logits

    def forward(self, x):
        # x: [Batch, Seq, 8]
        B, S, D = x.shape
        ph = torch.zeros(B, self.h, device=x.device)
        logits_seq = []
        hist_seq = []
        
        for t in range(S):
            xt = x[:, t, :]
            pt = self.e(xt)
            xpt = torch.stack([torch.cos(pt), torch.sin(pt)], 2)
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

def train_model(model_name, model_class, hidden_size, device, training_sequences, epochs, uid, output_dir, stats_list, L_FILE):
    def P(s): print(s); L_FILE.write(str(s) + '\n'); L_FILE.flush()
    
    P(f"\n--- Training {model_name} ---")
    
    model = model_class(hidden_size).to(device)
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
            
        epoch_loss = 0
        total_correct = 0
        total_tokens = 0
        
        # Shuffle sequences for robustness
        seq_list = list(training_sequences)
        random.shuffle(seq_list)
        
        optimizer.zero_grad()
        
        # Accumulate gradients over sequences (or batch them if lengths matched)
        # Lengths vary ("RED+BLUE=PURPLE." vs "RED+RED=RED.")
        # So we process one by one
        
        for seq in seq_list:
            bytes_seq = list(seq.encode('ascii'))
            input_seq = bytes_seq[:-1] 
            target_seq = bytes_seq[1:] 
            
            x = str_to_bits(seq[:-1]).unsqueeze(0).to(device)
            y = torch.tensor(target_seq, dtype=torch.long).to(device)
            
            logits, hist = model(x)
            
            logits_flat = logits.view(-1, 256)
            y_flat = y
            
            if model_name == "ROUND":
                loss, tk, lk = criterion(logits_flat, y_flat, hist)
                loss.backward()
                epoch_loss += tk
            else:
                loss = criterion(logits_flat, y_flat)
                loss.backward()
                epoch_loss += loss.item()
                
            # Acc
            pred = torch.argmax(logits_flat, 1)
            total_correct += (pred == y_flat).sum().item()
            total_tokens += y_flat.shape[0]
            
        optimizer.step()
        
        epoch_acc = total_correct / total_tokens
        acc_history.append(epoch_acc)
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            P(f"Epoch {epoch:4d} | Acc: {epoch_acc:.4f} | Loss: {epoch_loss:.4f}")
            
    stats_list.append(acc_history)
    P(f"{model_name} Final Acc: {acc_history[-1]:.4f}")
    
    # Final Predictions (Fixed Order)
    all_preds_list = []
    with torch.no_grad():
        for seq in training_sequences: # training_sequences passed in is the fixed global list
             x = str_to_bits(seq[:-1]).unsqueeze(0).to(device)
             # y = seq[1:] (target)
             
             logits, _ = model(x)
             # Logits: [1, Seq, 256]
             pred_indices = torch.argmax(logits, dim=2).cpu().numpy().flatten()
             all_preds_list.append(pred_indices)
             
    flat_preds = np.concatenate(all_preds_list)
    return model, flat_preds

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
    P(f"Task: Learn Symbolic Logic (A+B=C) from 12 examples.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = TC['EPOCHS']
    
    round_stats = []
    gru_stats = []
    round_preds = []
    
    # Generate Ground Truth Flat Vector
    gt_list = []
    for seq in TRAINING_SEQUENCES:
        # Target is seq[1:] (bytes)
        t = list(seq.encode('ascii')[1:])
        gt_list.append(np.array(t))
    gt_flat = np.concatenate(gt_list)
    
    RUNS = 5
    
    # Train ROUND
    P(f"Training ROUND ({RUNS} Runs)...")
    last_r_model = None
    for i in range(RUNS):
        r_model, p = train_model(f"ROUND_{i+1}", ColorROUND, TC['HIDDEN_R'], device, TRAINING_SEQUENCES, EPOCHS, UID, output_dir, round_stats, L_FILE)
        round_preds.append(p)
        last_r_model = r_model
    
    # Train GRU
    P(f"Training GRU ({RUNS} Runs)...")
    for i in range(RUNS):
        g_model, p = train_model(f"GRU_{i+1}", ColorGRU, TC['HIDDEN_G'], device, TRAINING_SEQUENCES, EPOCHS, UID, output_dir, gru_stats, L_FILE)
    
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
    
    ax.set_title(f"Semantic Logic Learning Curve\nTask: Color Algebra", fontsize=14, color='white')
    ax.set_xlabel('Epochs', fontsize=12, color='gray')
    ax.set_ylabel('Accuracy (Next Token)', fontsize=12, color='gray')
    ax.grid(True, alpha=0.1)
    ax.legend()
    
    plt.savefig(os.path.join(output_dir, f'benchmark_colors_{UID}.png'), dpi=300)
    P(f"Plot saved to benchmark_colors_{UID}.png")
    
    # Correlation Plot
    ds = np.vstack([np.stack(round_preds), gt_flat])
    corr = np.corrcoef(ds)
    labels = [f'R{i+1}' for i in range(RUNS)] + ['GT']
    
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, interpolation='nearest', cmap='coolwarm', vmin=0, vmax=1)
    plt.title(f'ROUND Consistency: Colors\nBatch {UID}')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr[i, j]
            if np.isnan(val): val = 0
            text = plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if 0.3 < val < 0.7 else "white")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'correlation_colors_{UID}.png'), dpi=300)
    P(f"Correlation plot saved to correlation_colors_{UID}.png")
    
    # Restore r_model for inference text
    r_model = last_r_model

    P("\n--- Testing Color Algebra (Vector Addition) [ROUND] ---")
    TEST_PROMPTS = ["RED+BLUE=", "YELLOW+BLUE=", "RED+RED=", "BLACK+WHITE="]
    
    with torch.no_grad():
        for prompt in TEST_PROMPTS:
            # Re-implement generation logic for R_MODEL
            gen_text = generate_completion(r_model, prompt, device)
            P(f"Prompt: {prompt:10s} -> Predicted: {gen_text}")
    
    L_FILE.close()

def generate_completion(model, prompt, device):
    gen_text = ""
    prompt_bits = str_to_bits(prompt).unsqueeze(0).to(device)
    
    # Feed prompt
    logits, hist = model(prompt_bits)
    
    # Initial next char
    last_logit = logits[:, -1, :]
    next_char_code = torch.argmax(last_logit, dim=1).item()
    next_char = chr(next_char_code)
    gen_text += next_char
    
    # If model has history state (ROUND), use it
    if hasattr(model, 'c'):
        ph = hist[-1]
        for _ in range(10):
            if next_char == '.': break
            
            x_next = str_to_bits(next_char).unsqueeze(0).to(device)
            
            pt = model.e(x_next[:, 0, :])
            xpt = torch.stack([torch.cos(pt), torch.sin(pt)], 2)
            ph = model.c(ph, xpt)
            
            readout = torch.cat([torch.cos(ph), torch.sin(ph), ph], 1)
            logit = model.r(readout)
            next_char_code = torch.argmax(logit, dim=1).item()
            next_char = chr(next_char_code)
            gen_text += next_char
    else:
        # GRU Autoregression (requires keeping state or re-feeding?)
        # For simplicity, we just autoregress by re-feeding growing sequence
        curr_seq = prompt + gen_text
        for _ in range(10):
            if next_char == '.': break
            
            seq_bits = str_to_bits(curr_seq).unsqueeze(0).to(device)
            out, _ = model.gru(seq_bits)
            logit = model.fc(out[:, -1, :])
            next_char_code = torch.argmax(logit, dim=1).item()
            next_char = chr(next_char_code)
            gen_text += next_char
            curr_seq += next_char
            
    return gen_text

if __name__ == "__main__":
    train()
