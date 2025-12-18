# version 0.6.1 - Harmonic Monism (Long-Term Memory Curriculum)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import uuid
import time
from ROUND import PhaseAccumulator, HarmonicROUNDLoss
from config import get_lock_strength

# --- Configuration ---
HIDDEN_SIZE = 32
LR = 0.001953125 # 2^-9
PEAK_LOCKING_STRENGTH = 0.0625
EPOCHS = 2000 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Keywords for the Curriculum
WORDS = ["CONCENTRATING", "RELIABILITY", "TOPOLOGY", "SYMMETRY", "MONISM"]

class LongTermROUNDModel(nn.Module):
    def __init__(self, hidden_size=32, input_dim=8, output_dim=256):
        super().__init__()
        self.h = hidden_size
        self.e = nn.Linear(input_dim, hidden_size) 
        self.c = PhaseAccumulator(hidden_size, spinor=True)
        # Readout: [Cos, Sin, CosS, SinS, Ph]
        self.r = nn.Linear(hidden_size * 5, output_dim)

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
            
            ph_s = 0.5 * ph
            readout_features = torch.cat([
                torch.cos(ph), torch.sin(ph), 
                torch.cos(ph_s), torch.sin(ph_s), 
                ph
            ], 1)
            logits = self.r(readout_features)
            logits_seq.append(logits)

        return torch.stack(logits_seq, dim=1), hist_seq

def str_to_bits(s):
    bits = [[int(c) for c in format(ord(ch), '08b')] for ch in s]
    return torch.tensor(bits, dtype=torch.float32)

def get_word_data(word):
    # Shifted character prediction (Next char)
    input_bits = str_to_bits(word).unsqueeze(0).to(DEVICE)
    targets = [ord(c) for c in word[1:] + word[0]]
    targets = torch.tensor(targets, dtype=torch.long).unsqueeze(0).to(DEVICE)
    return input_bits, targets

def train_long_term():
    UID = str(uuid.uuid4())[:8]
    output_dir = 'data'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    log_path = os.path.join(output_dir, f'log_long_term_{UID}.txt')
    L_FILE = open(log_path, 'w')
    
    def P(s): print(s); L_FILE.write(str(s) + '\n'); L_FILE.flush()

    P(f"--- [LONG-TERM MEMORY CURRICULUM v0.6.1] ---")
    P(f"Keywords: {WORDS}")
    P(f"Epochs: {EPOCHS} | Hidden: {HIDDEN_SIZE}")

    model = LongTermROUNDModel(HIDDEN_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion_lock = HarmonicROUNDLoss(locking_strength=PEAK_LOCKING_STRENGTH, harmonics=[1], mode='multiclass')
    
    # Store data for all words
    word_data = {word: get_word_data(word) for word in WORDS}
    
    # History for plotting
    history = {word: [] for word in WORDS}
    loss_history = []
    
    print("Beginning Artificial Memory Gradient...")
    
    for epoch in range(EPOCHS):
        model.train()
        
        # 1. Determine Current Dominant Word (The Curriculum Gradient)
        # We shift focus from WORDS[0] to WORDS[-1] over time
        # Curve: Use a sigmoid or simple linear split
        num_words = len(WORDS)
        phase = (epoch / EPOCHS) * num_words
        current_idx = int(phase) if phase < num_words else num_words - 1
        current_word = WORDS[current_idx]
        
        # 2. Leaky Sampling: Small chance to train on any word seen so far
        if np.random.rand() < 0.2:
            train_idx = np.random.randint(0, current_idx + 1)
            train_word = WORDS[train_idx]
        else:
            train_word = current_word
            
        # 3. Training Step
        input_bits, targets = word_data[train_word]
        criterion_lock.locking_strength = get_lock_strength(epoch % (EPOCHS // num_words), EPOCHS // num_words, PEAK_LOCKING_STRENGTH)
        
        optimizer.zero_grad()
        logits, hist = model(input_bits)
        loss, tk_loss, lk_loss = criterion_lock(logits.view(-1, 256), targets.view(-1), hist)
        loss.backward()
        optimizer.step()
        loss_history.append(tk_loss)
        
        # 4. Periodic "Leaky Blind Test" (Evaluate ALL words)
        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                for word in WORDS:
                    in_b, tgt = word_data[word]
                    l, _ = model(in_b)
                    acc = (torch.argmax(l, 2) == tgt).float().mean().item()
                    history[word].append(acc)
            
            p_str = f"E {epoch:4d} | [Active: {train_word[:4]}] | "
            p_str += " ".join([f"{word[0]}:{history[word][-1]:.1f}" for word in WORDS])
            if epoch % 100 == 0: P(p_str)

    # --- Plotting Results ---
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Accuracy Plot
    ep_axis = np.linspace(0, EPOCHS, len(history[WORDS[0]]))
    colors = ['#FF4B4B', '#4BFF4B', '#4B4BFF', '#FFFF4B', '#FF4BFF']
    
    for i, word in enumerate(WORDS):
        ax1.plot(ep_axis, history[word], label=word, color=colors[i % len(colors)], linewidth=2)
    
    # Vertical bars for curriculum stages
    for i in range(1, num_words):
        ax1.axvline(x=(i/num_words)*EPOCHS, color='white', linestyle='--', alpha=0.2)
    
    ax1.set_title("Long-Term Memory Retention (32 Neurons)", fontsize=14)
    ax1.set_ylabel("Recall Accuracy", fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.1)
    
    # Loss Plot
    ax2.plot(loss_history, color='gray', alpha=0.5)
    ax2.set_title("Learning Dynamics", fontsize=14)
    ax2.set_xlabel("Epochs", fontsize=12)
    ax2.set_ylabel("CrossEntropy Loss", fontsize=12)
    ax2.grid(True, alpha=0.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'benchmark_long_term_{UID}.png'), dpi=300)
    P(f"Results saved to data/benchmark_long_term_{UID}.png")
    
    # Final Analysis
    P("\n--- Final Recall Audit ---")
    for word in WORDS:
        final_acc = history[word][-1]
        status = "CLAMPED" if final_acc > 0.9 else "FORGOTTEN"
        P(f"{word:12s}: {final_acc*100:6.1f}% [{status}]")
    
    L_FILE.close()

if __name__ == "__main__":
    train_long_term()
