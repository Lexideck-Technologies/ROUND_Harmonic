import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Ensure root is in path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path: sys.path.append(root_dir)
from UIT_ROUND import UITModel

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default=".")
parser.add_argument("--uid", type=str, default="prism_restored")
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--crystal_path", type=str, default=None)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = args.output_dir
UID = args.uid
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- CONFIGURATION ---
BATCH_SIZE = 64
EPOCHS = 2000 # budget for logic crystallization
LEARNING_RATE = args.lr if args.lr is not None else 0.02 
HIDDEN_SIZE = 18  # Match modular space (like Color Algebra's 64/64)

class PrismROUND(nn.Module):
    def __init__(self):
        super().__init__()
        # Match Color Algebra: 1-layer, direct cell access for Phasic Identity preservation
        # quantization_strength=0.0 for Mod-18 compatibility
        # persistence=0.5 for phase decay (critical for learning!)
        self.uit = UITModel(input_size=18, hidden_size=HIDDEN_SIZE, output_size=18, num_layers=1, quantization_strength=0.0, persistence=0.5)
        self.classifier = nn.Linear(HIDDEN_SIZE * 3, 18)
        
    def forward(self, xl, xp):
        # Explicit Phase Passing (ColorROUND Pattern)
        h = torch.zeros(xl.size(0), HIDDEN_SIZE).to(xl.device)
        # Step 1: Process xl (the Lens), accumulate phase into h
        _, h, _, _, _ = self.uit.layers[0](xl[:, 0, :], h)
        # Step 2: Process xp (the Light) through the prism state h
        feat, h, _, h_cos, h_sin = self.uit.layers[0](xp[:, 0, :], h)
        # Readout: Combine standard output with harmonic features
        combined = torch.cat([feat, h_cos, h_sin], dim=-1)
        return self.classifier(combined)

class PrismGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(18, HIDDEN_SIZE, num_layers=2, batch_first=True)
        self.readout = nn.Linear(HIDDEN_SIZE, 18)
    def forward(self, xl, xp):
        x_seq = torch.cat([xl, xp], dim=1)
        _, h = self.gru(x_seq)
        return self.readout(h[-1])

def run_benchmark():
    print(f"--- [v1.3.11 PRISM STACK RESTORATION | UID: {UID}] ---")
    r_model = PrismROUND().to(DEVICE)
    g_model = PrismGRU().to(DEVICE)
    
    # Using 0.02 LR to force movement in weights
    r_opt = optim.Adam(r_model.parameters(), lr=LEARNING_RATE)
    g_opt = optim.Adam(g_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    history = {"round": [], "gru": []}
    
    for epoch in range(EPOCHS):
        xl_idx = torch.randint(0, 18, (BATCH_SIZE,)).to(DEVICE)
        xp_idx = torch.randint(0, 18, (BATCH_SIZE,)).to(DEVICE)
        
        xl = torch.zeros(BATCH_SIZE, 1, 18).to(DEVICE).scatter_(2, xl_idx.view(-1, 1, 1), 1.0)
        xp = torch.zeros(BATCH_SIZE, 1, 18).to(DEVICE).scatter_(2, xp_idx.view(-1, 1, 1), 1.0)
        y = (xl_idx + xp_idx) % 18
        
        # Training
        r_model.train(); g_model.train()
        r_opt.zero_grad(); r_loss = criterion(r_model(xl, xp), y); r_loss.backward(); r_opt.step()
        g_opt.zero_grad(); g_loss = criterion(g_model(xl, xp), y); g_loss.backward(); g_opt.step()
        
        if epoch % 100 == 0:
            history["round"].append(r_loss.item())
            history["gru"].append(g_loss.item())
            print(f"Epoch {epoch:4d} | ROUND: {r_loss.item():.4f} | GRU: {g_loss.item():.4f}")
            if r_loss < 0.001 and g_loss < 0.001: break

    # --- VISUALIZATION ---
    try:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history["round"], color='#00FF00', linewidth=3, label='ROUND (Green)')
        ax.plot(history["gru"], color='#4B4BFF', linewidth=3, label='GRU (Blue)')
        ax.set_title(f"Prism Stack Duel: Learning Convergence | UID: {UID}", color='white', fontsize=14)
        ax.set_xlabel("Epochs (x100)", color='white')
        ax.set_ylabel("Cross Entropy Loss", color='white')
        ax.grid(True, alpha=0.3)
        ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
        
        plot_path = os.path.join(OUTPUT_DIR, f"prism_stack_duel_{UID}.png")
        plt.savefig(plot_path, facecolor='black', edgecolor='none')
        print(f"Plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"Visualization Failed: {e}")

if __name__ == "__main__":
    run_benchmark()
