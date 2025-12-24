import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# version 0.8.0 - "The Frozen Basin" (Freezing Mechanism Development)
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

# --- Custom Freezing Mask Generator ---
class FreezingGradientMask(nn.Module):
    """
    Implements the 'Gradient Vault':
    Instead of penalizing movement (stiff spring), we identify solved neurons
    and return a MASK. The optimizer loop will then ZERO the gradients for these
    neurons, physically preventing them from moving regardless of the loss.
    """
    def __init__(self, harmonics=[1], weights=None, solved_threshold=0.001953125, hidden_size=64):
        super().__init__()
        self.h = harmonics
        self.w = weights if weights else [1.0]*len(harmonics)
        self.solved_threshold = solved_threshold
        # Permanent Graduation Registry
        self.register_buffer('permanent_mask', torch.zeros(hidden_size, dtype=torch.bool))

    def forward(self, hist):
        # Unpack History
        if isinstance(hist[0], tuple):
            # Phase is the first element
            st_ph = torch.stack([h[0] for h in hist])
        else:
            st_ph = torch.stack(hist)
            
        # Calculate Per-Neuron Locking Error (Braking Requirement)
        # Shape of st_ph: [Seq, Batch, Hidden]
        
        # Accumulate error across harmonics
        locking_error = torch.zeros_like(st_ph)
        for i, s in enumerate(self.h):
            locking_error += self.w[i] * (torch.sin(s/2.0 * st_ph)**2)
        
        # Normalize by number of harmonics
        locking_error /= len(self.h)
        
        # Collapse time dimension (Seq)
        avg_error = torch.mean(locking_error, dim=0) # [Batch, Hidden]
        
        # Identify CURRENTLY Solved Neurons
        current_solved = avg_error < self.solved_threshold # [Batch, Hidden]
        
        # Since Batch=1 in this test, squeeze to [Hidden]
        current_solved = current_solved.squeeze(0)
        
        # Update Permanent Registry (Logical OR)
        self.permanent_mask = self.permanent_mask | current_solved
        
        return self.permanent_mask # Return the cumulative mask

# --- Models ---
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

    def forward(self, x, pruning_mask=None):
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
                
                # Stack features: [Batch, NumFeatures, Hidden]
                # Order matters to match existing weights?
                # Original was cat([Cos(ph), Sin(ph)...], 1) which is [B, H*F] where H is inner?
                # No, cat([B,H], [B,H]) makes [B, 2H].
                # So indices 0..63 are Cos. 64..127 are Sin.
                # So it IS Feature-Major.
                # Stack [B, H] -> [B, F, H]. Flatten -> [B, F*H].
                # Correct.
                
                features_list = [
                    torch.cos(ph), torch.sin(ph), 
                    torch.cos(ph_s), torch.sin(ph_s), 
                    torch.cos(wb), torch.sin(wb),
                    ph
                ]
            else:
                ph = self.c(ph, xpt)
                hist_seq.append(ph)
                ph_s = 0.5 * ph
                features_list = [
                    torch.cos(ph), torch.sin(ph), 
                    torch.cos(ph_s), torch.sin(ph_s), 
                    ph
                ]
            
            # Form Readout Input
            stacked = torch.stack(features_list, dim=1) # [B, F, H]
            
            # STORM PRUNING
            if pruning_mask is not None:
                # pruning_mask is [Hidden] (True for Keep/Frozen, False for Prune)
                # We want to KEEP True.
                # Ensure mask is float
                mask_tensor = pruning_mask.float().view(1, 1, -1) # [1, 1, H]
                stacked = stacked * mask_tensor
                
            readout_features = stacked.reshape(B, -1) # Flatten to [B, F*H]
                
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

def run_long_term_comparison(shuffled_words, epochs=10500, hidden_size_r=64, hidden_size_g=None, p_func=print, output_dir='data', plot_name=None):
    if hidden_size_g is None: hidden_size_g = hidden_size_r
    UID = str(uuid.uuid4())[:8]
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    p_func(f"--- [ROUND vs GRU: PHASE ANGLE LOCK TEST] ---")
    p_func(f"Order: {shuffled_words}")
    p_func(f"Epochs: {epochs} | ROUND Hidden: {hidden_size_r} | GRU Hidden: {hidden_size_g}")

    # 1. Models & Optimizers
    model_r = LongTermROUNDModel(hidden_size_r, wobble=True).to(DEVICE)
    model_g = GRULongTermModel(hidden_size_g).to(DEVICE)
    
    opt_r = optim.Adam(model_r.parameters(), lr=TC['LR'])
    opt_g = optim.Adam(model_g.parameters(), lr=TC['LR'])
    
    harmonics = [1, 2, 4, 8]
    weights = [1.0, 0.25, 0.0625, 0.015625]
    
    # Use standard harmonic loss for now, we will add the mask logic in the loop
    crit_r = HarmonicROUNDLoss(locking_strength=TC['PEAK_LOCKING_STRENGTH'], harmonics=harmonics, weights=weights, mode='multiclass', wobble_gravity=0.1)
    crit_g = nn.CrossEntropyLoss()
    
    # Freezing Mask Generator
    mask_gen = FreezingGradientMask(harmonics=harmonics, weights=weights, solved_threshold=0.001953125, hidden_size=hidden_size_r).to(DEVICE)
    
    word_data = {word: get_word_data(word) for word in shuffled_words}
    hist_r = {word: [] for word in shuffled_words}
    hist_g = {word: [] for word in shuffled_words}
    loss_r, loss_g = [], []
    
    NOISE_START = epochs - 1500

    # Define layers to freeze (Vertical Crystal)
    # We do this once to avoid logic overhead in the loop
    frozen_layers = [model_r.e, model_r.c.d]
    if hasattr(model_r.c, 'd_w'):
        frozen_layers.append(model_r.c.d_w)
        
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
        
        # --- NOISE STORM INJECTION ---
        if epoch >= NOISE_START:
            if epoch == NOISE_START:
                p_func(f"\n!!! [NOISE STORM INITIATED] Epoch {epoch}: Interference Level 0.5 (Max 50% Corruption) !!!")
            input_bits = get_stochastic_payload(raw_bits) # Apply standard noise first
            # Manually apply storm noise here
            storm_noise = torch.randn_like(input_bits) * 0.5
            storm_mask = (torch.rand_like(input_bits) > 0.4).float() # Keeps 60% approx
            input_bits = (input_bits + storm_noise) * storm_mask
        else:
            # STANDARD TRAINING
            input_bits = get_stochastic_payload(raw_bits)

        
        # Continuous Learning Protocol:
        # If training the CURRENT word: Use the 50% Fluid/50% Crystalline curve.
        # If training an OLD word (Revision): Use the FLOOR strength immediately.
        cycle_len = epochs // num_words
        if train_word == current_word:
            crit_r.locking_strength = get_lock_strength(epoch % cycle_len, cycle_len, TC['PEAK_LOCKING_STRENGTH'], floor_strength=TC['FLOOR'])
        else:
            # Strengthen the lock for revisited words!
            crit_r.locking_strength = TC['FLOOR'] * 2.0
        
        # ROUND
        if epoch < NOISE_START:
            model_r.train()
            opt_r.zero_grad()
            l_r, h_r = model_r(input_bits)
            
            # Determine if we freeze based on global locked status or local mask?
            # We stick to the Gradient Vault Mask (local neuron state)
            ls_r, tk_r, _ = crit_r(l_r.view(-1, 256), targets.view(-1), h_r)
            
            # --- GRADIENT VAULT: APPLY FREEZING ---
            ls_r.backward()
            
            frozen_mask = mask_gen(h_r) # [Hidden]
            if frozen_mask.any():
                # Apply to all registered frozen layers (Vertical Crystal)
                for layer in frozen_layers:
                    if layer.weight.grad is not None:
                        layer.weight.grad[frozen_mask] = 0.0
                    if layer.bias is not None and layer.bias.grad is not None:
                        layer.bias.grad[frozen_mask] = 0.0
                
            opt_r.step()
            loss_r.append(tk_r)
        else:
            # ROUND CRYOSTASIS (Storm Mode)
            # The model is frozen. It faces the storm with fixed weights.
            model_r.eval()
            with torch.no_grad():
                l_r, h_r = model_r(input_bits)
                ls_r, tk_r, _ = crit_r(l_r.view(-1, 256), targets.view(-1), h_r)
            loss_r.append(tk_r)
        
        # GRU (Continues to struggle/learn/overfit)
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

    # Plotting with Seaborn
    from visualization_utils import setup_seaborn_theme, plot_multi_word_comparison

    palette = setup_seaborn_theme(style='darkgrid', palette='classic')
    ep_axis = np.linspace(0, epochs, len(hist_r[shuffled_words[0]]))
    colors = ['#FF4B4B', '#4B4BFF', '#FFFF4B', '#FF4BFF', '#4BFFFF', '#FFA500']

    if plot_name:
        plot_path = os.path.join(output_dir, plot_name)
    else:
        plot_path = os.path.join(output_dir, f'benchmark_phase_lock_{UID}.png')

    plot_multi_word_comparison(
        hist_r=hist_r,
        hist_g=hist_g,
        words=shuffled_words,
        ep_axis=ep_axis,
        hidden_size_r=hidden_size_r,
        hidden_size_g=hidden_size_g,
        output_path=plot_path,
        word_colors=colors
    )
    p_func(f"Results saved to {plot_path}")
    
    final_res = {word: (hist_r[word][-1], hist_g[word][-1]) for word in shuffled_words}
    return final_res

def train_long_term():
    UID = os.environ.get('ROUND_BATCH_UID', str(uuid.uuid4())[:8])
    base_dir = os.environ.get('ROUND_OUTPUT_DIR', 'data')
    output_dir = os.path.join(base_dir, UID)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    log_path = os.path.join(output_dir, f'log_phase_lock_{UID}.txt')
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
