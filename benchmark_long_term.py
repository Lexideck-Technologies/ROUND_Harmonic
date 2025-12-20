# version 0.7.3 - "The Hyper-Resolution Basin" (Long-Term Memory)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import uuid
import time
import random
from ROUND import PhaseAccumulator, WobblePhaseAccumulator, HarmonicROUNDLoss
from config import get_lock_strength, LONG_TERM_CONFIG

# --- Configuration ---
TC = LONG_TERM_CONFIG
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Keywords for the Curriculum (6-Word Deep Bake Suite)
WORDS = ["COGITATING", "TOPOLOGY", "MONISM", "RESONANCE", "UNIVERSAL", "SYMMETRY"]

class LongTermROUNDModel(nn.Module):
    def __init__(self, hidden_size=32, input_dim=8, output_dim=256, wobble=True):
        super().__init__()
        self.h = hidden_size
        self.wobble = wobble
        self.e = nn.Linear(input_dim, hidden_size) 
        if wobble:
            self.c = WobblePhaseAccumulator(hidden_size, spinor=True)
            self.c.coupling = -1.0 # The Opposite Direction!
        else:
            self.c = PhaseAccumulator(hidden_size, spinor=True)
            
        # Readout: [Cos, Sin, CosS, SinS, CosW, SinW (if w), Ph]
        num_features = 5 + (2 if wobble else 0)
        self.r = nn.Linear(hidden_size * num_features, output_dim)

    def forward(self, x):
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
                # 1. Constant Mnemonic Drift (The Clock)
                wb = wb + 0.015625 # 2^-6 drift
                
                # 2. Triggered Gemination Deflection
                is_repeat = False
                if prev_xt is not None:
                    is_repeat = torch.all(torch.eq(xt, prev_xt)).item()
                
                if is_repeat:
                    # Accelerate wobble into the Z-axis to break the M-M parity
                    ph, wb = self.c(ph, xpt, wb)
                else:
                    # Planar discovery, keeping the drift-clock steady
                    ph, _ = self.c(ph, xpt, wb)
                
                prev_xt = xt
                hist_seq.append((ph, wb))
                ph_s = 0.5 * ph
                readout_features = torch.cat([
                    torch.cos(ph), torch.sin(ph), 
                    torch.cos(ph_s), torch.sin(ph_s), 
                    torch.cos(wb), torch.sin(wb),
                    ph
                ], 1)
            else:
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

class GRULongTermModel(nn.Module):
    def __init__(self, hidden_size=64, input_dim=8, output_dim=256):
        super().__init__()
        self.h = hidden_size
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.r = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # x: [Batch, Seq, 8]
        out, _ = self.gru(x)
        # out: [Batch, Seq, H]
        logits = self.r(out)
        return logits, None # No phase history for GRU

def str_to_bits(s):
    bits = [[int(c) for c in format(ord(ch), '08b')] for ch in s]
    return torch.tensor(bits, dtype=torch.float32)

def get_word_data(word):
    # Shifted character prediction (Next char)
    input_bits = str_to_bits(word).unsqueeze(0).to(DEVICE)
    targets = [ord(c) for c in word[1:] + word[0]]
    targets = torch.tensor(targets, dtype=torch.long).unsqueeze(0).to(DEVICE)
    return input_bits, targets

def get_stochastic_payload(bits):
    # 1. Gaussian Spreading (Spiritual/Twistor Noise)
    # Using 2^-5 (0.03125) as the 'blur' radius
    noise = torch.randn_like(bits) * 0.03125
    
    # 2. Stochastic Masking (Bit Dropout)
    # 5% chance to lose a bit-signal, force logic redundancy
    mask = (torch.rand_like(bits) > 0.05).float()
    
    return (bits + noise) * mask

def run_long_term_comparison(shuffled_words, epochs=10000, hidden_size_r=64, hidden_size_g=None, p_func=print, output_dir='data'):
    if hidden_size_g is None: hidden_size_g = hidden_size_r
    UID = str(uuid.uuid4())[:8]
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    p_func(f"--- [ROUND vs GRU: LONG-TERM COMPARISON] ---")
    p_func(f"Order: {shuffled_words}")
    p_func(f"Epochs: {epochs} | ROUND Hidden: {hidden_size_r} | GRU Hidden: {hidden_size_g}")

    # 1. Models & Optimizers
    model_r = LongTermROUNDModel(hidden_size_r, wobble=True).to(DEVICE)
    model_g = GRULongTermModel(hidden_size_g).to(DEVICE)
    
    opt_r = optim.Adam(model_r.parameters(), lr=TC['LR'])
    opt_g = optim.Adam(model_g.parameters(), lr=TC['LR'])
    
    harmonics = [1, 2, 4, 8]
    weights = [1.0, 0.25, 0.0625, 0.015625]
    crit_r = HarmonicROUNDLoss(locking_strength=TC['PEAK_LOCKING_STRENGTH'], harmonics=harmonics, weights=weights, mode='multiclass', wobble_gravity=0.1)
    crit_g = nn.CrossEntropyLoss()
    
    word_data = {word: get_word_data(word) for word in shuffled_words}
    hist_r = {word: [] for word in shuffled_words}
    hist_g = {word: [] for word in shuffled_words}
    loss_r, loss_g = [], []
    
    for epoch in range(epochs):
        model_r.train(); model_g.train()
        num_words = len(shuffled_words)
        phase = (epoch / epochs) * num_words
        current_idx = int(phase) if phase < num_words else num_words - 1
        current_word = shuffled_words[current_idx]
        
        if np.random.rand() < 0.4:
            stubborn = [w for w in shuffled_words[:current_idx+1] if len(hist_r[w]) > 0 and hist_r[w][-1] < 1.0]
            train_word = random.choice(stubborn) if stubborn and np.random.rand() < 0.7 else shuffled_words[np.random.randint(0, current_idx + 1)]
        else:
            train_word = current_word
            
        raw_bits, targets = word_data[train_word]
        input_bits = get_stochastic_payload(raw_bits)
        
        # Continuous Learning Protocol:
        # If training the CURRENT word: Use the 50% Fluid/50% Crystalline curve.
        # If training an OLD word (Revision): Use the FLOOR strength immediately.
        cycle_len = epochs // num_words
        if train_word == current_word:
            crit_r.locking_strength = get_lock_strength(epoch % cycle_len, cycle_len, TC['PEAK_LOCKING_STRENGTH'], floor_strength=TC['FLOOR'])
        else:
            crit_r.locking_strength = TC['FLOOR']
        
        # ROUND
        opt_r.zero_grad()
        l_r, h_r = model_r(input_bits)
        ls_r, tk_r, _ = crit_r(l_r.view(-1, 256), targets.view(-1), h_r)
        ls_r.backward(); opt_r.step()
        loss_r.append(tk_r)
        
        # GRU
        opt_g.zero_grad()
        l_g, _ = model_g(input_bits)
        ls_g = crit_g(l_g.view(-1, 256), targets.view(-1))
        ls_g.backward(); opt_g.step()
        loss_g.append(ls_g.item())
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            model_r.eval(); model_g.eval()
            with torch.no_grad():
                for word in shuffled_words:
                    in_b, tgt = word_data[word]
                    # ROUND
                    l_r, _ = model_r(in_b)
                    acc_r = (torch.argmax(l_r, 2) == tgt).float().mean().item()
                    hist_r[word].append(acc_r)
                    # GRU
                    l_g, _ = model_g(in_b)
                    acc_g = (torch.argmax(l_g, 2) == tgt).float().mean().item()
                    hist_g[word].append(acc_g)
            if epoch % 100 == 0:
                p_func(f"E {epoch:4d} | R: {np.mean([hist_r[w][-1] for w in shuffled_words]):.2f} | G: {np.mean([hist_g[w][-1] for w in shuffled_words]):.2f}")

    # Plotting
    plt.style.use('dark_background')
    fig, (ax_r, ax_g) = plt.subplots(2, 1, figsize=(14, 12))
    ep_axis = np.linspace(0, epochs, len(hist_r[shuffled_words[0]]))
    colors = ['#FF4B4B', '#4B4BFF', '#FFFF4B', '#FF4BFF', '#4BFFFF', '#FFA500']
    
    for i, word in enumerate(shuffled_words):
        ax_r.plot(ep_axis, hist_r[word], label=f"R: {word}", color=colors[i % len(colors)], linewidth=2)
        ax_g.plot(ep_axis, hist_g[word], label=f"G: {word}", color=colors[i % len(colors)], linewidth=2, linestyle='--')
    
    ax_r.set_title(f"ROUND - Spinor Monism ({hidden_size_r} Neurons)", color='#FF5555', fontsize=16)
    ax_g.set_title(f"GRU - Standard Gating ({hidden_size_g} Neurons)", color='#5555FF', fontsize=16)
    ax_r.legend(loc='lower left', fontsize=8, ncol=3); ax_g.legend(loc='lower left', fontsize=8, ncol=3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'benchmark_long_term_{UID}.png')
    plt.savefig(plot_path, dpi=300)
    p_func(f"Results saved to {plot_path}")
    
    final_res = {word: (hist_r[word][-1], hist_g[word][-1]) for word in shuffled_words}
    return final_res

def train_long_term():
    UID = os.environ.get('ROUND_BATCH_UID', str(uuid.uuid4())[:8])
    output_dir = os.environ.get('ROUND_OUTPUT_DIR', 'data')
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    log_path = os.path.join(output_dir, f'log_long_term_{UID}.txt')
    L_FILE = open(log_path, 'w')
    def L_P(s): print(s); L_FILE.write(str(s) + '\n'); L_FILE.flush()

    shuffled_words = list(WORDS)
    random.shuffle(shuffled_words)
    results = run_long_term_comparison(shuffled_words, LONG_TERM_CONFIG['EPOCHS'], LONG_TERM_CONFIG['HIDDEN_R'], LONG_TERM_CONFIG['HIDDEN_G'], L_P, output_dir)
    
    P = L_P
    P("\n--- Final Comparative Audit ---")
    header = f"{'Word':12s} | {'ROUND Recall':15s} | {'GRU Recall':15s}"
    P(header); P("-" * len(header))
    for word, (acc_r, acc_g) in results.items():
        P(f"{word:12s} | {acc_r*100:13.1f}% | {acc_g*100:13.1f}%")
    L_FILE.close()

if __name__ == "__main__":
    train_long_term()
