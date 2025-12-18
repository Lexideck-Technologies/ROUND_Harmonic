# version 0.6.1 - Harmonic Monism (Permutation Memory Test vs GRU)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import uuid
import itertools
from ROUND import PhaseAccumulator, HarmonicROUNDLoss
from config import PERMS_CONFIG, get_lock_strength

# --- Configuration ---
TC = PERMS_CONFIG
HIDDEN_SIZE = TC['HIDDEN_SIZE']
LR = TC['LR']
# Scaling LR for GRU to give it a fighting chance
GRU_LR = TC['LR']
PEAK_LOCKING_STRENGTH = TC['PEAK_LOCKING_STRENGTH']
EPOCHS = TC['EPOCHS']
RUNS = TC['RUNS']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tokens
WORDS = ["CONC", "RELI", "TOPO", "SYMM"]
SEPARATOR = " | "

# --- Models ---

class PermutationROUNDModel(nn.Module):
    def __init__(self, hidden_size=64, input_dim=8, output_dim=256):
        super().__init__()
        self.h = hidden_size
        self.e = nn.Linear(input_dim, hidden_size) 
        self.c = PhaseAccumulator(hidden_size, spinor=True)
        # Readout: [Cos, Sin, CosS, SinS, Ph]
        self.r = nn.Linear(hidden_size * 5, output_dim)

    def forward(self, x, ph_in=None):
        B, S, D = x.shape
        ph = ph_in if ph_in is not None else torch.zeros(B, self.h, device=x.device)
        logits_seq = []
        hist_seq = []
        for t in range(S):
            xt = x[:, t, :]
            pt = self.e(xt)
            xpt = torch.stack([torch.cos(pt), torch.sin(pt)], 2)
            ph = self.c(ph, xpt)
            hist_seq.append(ph)
            
            ph_s = 0.5 * ph
            readout_features = torch.cat([
                torch.cos(ph), torch.sin(ph), 
                torch.cos(ph_s), torch.sin(ph_s), 
                ph
            ], 1)
            logits = self.r(readout_features)
            logits_seq.append(logits)
        return torch.stack(logits_seq, dim=1), hist_seq

class PermutationGRUModel(nn.Module):
    def __init__(self, hidden_size=64, input_dim=8, output_dim=256):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)
        
    def forward(self, x):
        out, h = self.gru(x)
        logits = self.fc(out)
        return logits, []

# --- Data ---

def str_to_bits(s):
    bits = [[int(c) for c in format(ord(ch), '08b')] for ch in s]
    return torch.tensor(bits, dtype=torch.float32)

def generate_permutation_data(perms):
    data = []
    for p in perms:
        text = SEPARATOR.join(p)
        in_bits = str_to_bits(text).unsqueeze(0).to(DEVICE)
        targets = [ord(c) for c in text[1:] + text[0]]
        targets = torch.tensor(targets, dtype=torch.long).unsqueeze(0).to(DEVICE)
        data.append((text, in_bits, targets))
    return data

# --- Training ---

def train_model(model_name, model_class, perm_data, device, epochs, L_FILE):
    def P(s): print(s); L_FILE.write(str(s) + '\n'); L_FILE.flush()
    P(f"\n--- Training {model_name} ---")
    
    model = model_class(HIDDEN_SIZE).to(device)
    lr = LR if "ROUND" in model_name else GRU_LR
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if "ROUND" in model_name:
        criterion = HarmonicROUNDLoss(locking_strength=PEAK_LOCKING_STRENGTH, harmonics=[1], mode='multiclass')
    else:
        criterion = nn.CrossEntropyLoss()
        
    acc_history = []
    for epoch in range(epochs):
        model.train()
        idx = epoch % len(perm_data)
        text, in_bits, targets = perm_data[idx]
        
        optimizer.zero_grad()
        logits, hist = model(in_bits)
        
        if "ROUND" in model_name:
            criterion.locking_strength = get_lock_strength(epoch, epochs, PEAK_LOCKING_STRENGTH, floor_strength=TC.get('FLOOR', 0.0))
            loss, tk_loss, lk_loss = criterion(logits.view(-1, 256), targets.view(-1), hist)
        else:
            loss = criterion(logits.view(-1, 256), targets.view(-1))
            tk_loss = loss.item()
            
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                total_acc = 0
                for _, ib, tg in perm_data:
                    l, _ = model(ib)
                    acc = (torch.argmax(l, 2) == tg).float().mean().item()
                    total_acc += acc
                avg_acc = total_acc / len(perm_data)
                acc_history.append(avg_acc)
                if epoch % 500 == 0 or epoch == epochs - 1:
                    P(f"E {epoch:5d} | Avg Acc: {avg_acc:.4f} | Loss: {tk_loss:.4f}")

    # Return final predictions on all perms for correlation
    model.eval()
    all_preds = []
    with torch.no_grad():
        for _, ib, tg in perm_data:
            l, _ = model(ib)
            all_preds.append(torch.argmax(l, 2).cpu().numpy().flatten())
            
    return model, acc_history, np.concatenate(all_preds)

def run_benchmark():
    UID = os.environ.get('ROUND_BATCH_UID', str(uuid.uuid4())[:8])
    output_dir = os.environ.get('ROUND_OUTPUT_DIR', 'data')
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    log_path = os.path.join(output_dir, f'log_perms_vs_gru_{UID}.txt')
    L_FILE = open(log_path, 'w')
    def P(s): print(s); L_FILE.write(str(s) + '\n'); L_FILE.flush()

    # Setup 4 permutations
    all_perms = list(itertools.permutations(WORDS))
    selected_perms = []
    for word in WORDS:
        for p in all_perms:
            if p[0] == word: selected_perms.append(p); break
    
    perm_data = generate_permutation_data(selected_perms)
    ground_truth = np.concatenate([tg.cpu().numpy().flatten() for _, _, tg in perm_data])

    P(f"--- [PERMUTATION BENCHMARK: ROUND vs GRU] ---")
    P(f"UID: {UID} | Hidden: {HIDDEN_SIZE} | EPOCHS: {EPOCHS} | RUNS: {RUNS}")

    round_all_acc = []
    gru_all_acc = []
    round_all_preds = []

    for i in range(RUNS):
        _, rac, rpr = train_model(f"ROUND_{i+1}", PermutationROUNDModel, perm_data, DEVICE, EPOCHS, L_FILE)
        round_all_acc.append(rac)
        round_all_preds.append(rpr)

    for i in range(RUNS):
        _, gac, _ = train_model(f"GRU_{i+1}", PermutationGRUModel, perm_data, DEVICE, EPOCHS, L_FILE)
        gru_all_acc.append(gac)

    # --- Plotting ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rm, rs = np.mean(round_all_acc, 0), np.std(round_all_acc, 0)
    gm, gs = np.mean(gru_all_acc, 0), np.std(gru_all_acc, 0)
    ep_axis = np.linspace(0, EPOCHS, len(rm))

    ax.fill_between(ep_axis, rm-rs, rm+rs, color='#FF4B4B', alpha=0.1)
    ax.fill_between(ep_axis, gm-gs, gm+gs, color='#4B4BFF', alpha=0.1)
    ax.plot(ep_axis, rm, color='#FF4B4B', linewidth=2.5, label='ROUND (Harmonic)')
    ax.plot(ep_axis, gm, color='#4B4BFF', linewidth=2.5, label='GRU (Standard)')
    
    ax.set_title(f"Permutation Recall: ROUND vs GRU\n4 Target Shuffles, Hidden={HIDDEN_SIZE}", fontsize=14)
    ax.set_xlabel('Epochs', color='gray')
    ax.set_ylabel('Avg Accuracy (All Perms)', color='gray')
    ax.grid(True, alpha=0.1)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'benchmark_perms_vs_gru_{UID}.png'), dpi=300)
    P(f"Learning curve saved to data/benchmark_perms_vs_gru_{UID}.png")

    # --- Correlation Heatmap ---
    plt.figure(figsize=(8, 6))
    ds = np.vstack([np.stack(round_all_preds), ground_truth])
    corr = np.corrcoef(ds)
    labels = [f'R{i+1}' for i in range(RUNS)] + ['GT']
    plt.imshow(corr, interpolation='nearest', cmap='coolwarm', vmin=0, vmax=1)
    plt.title(f'Normalization Consistency: Permutations\nBatch {UID}')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr[i, j]
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if 0.3 < val < 0.7 else "white")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'correlation_perms_{UID}.png'), dpi=300)
    P(f"Correlation plot saved to data/correlation_perms_{UID}.png")
    
    L_FILE.close()

if __name__ == "__main__":
    run_benchmark()
