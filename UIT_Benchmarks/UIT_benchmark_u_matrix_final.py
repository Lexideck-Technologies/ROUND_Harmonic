"""
U-Matrix 7-Octave Benchmark
Testing the U-Matrix architecture with the 7-octave harmonic spectrum and spin multiplier 0.5.
Task: Continuous Sine Wave Tracking
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
import argparse
import matplotlib.pyplot as plt

# Ensure root in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from UIT_u_matrix import UMatrix
from UIT_ROUND import HARMONICS_7OCTAVE

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default=".")
parser.add_argument("--uid", type=str, default="um_7oct")
parser.add_argument("--lr", type=float, default=0.0078125) # 2^-7 default
parser.add_argument("--crystal_path", type=str, default=None) # Included for battery compatibility
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = args.output_dir
UID = args.uid
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- CONFIGURATION ---
BATCH_SIZE = 32
SEQ_LEN = 40 
HIDDEN_DIM = 8          # Optimal for sine wave
SPIN_MULTIPLIER = 0.5   # Optimal phase range (π)
HARMONICS = HARMONICS_7OCTAVE # [0.125 ... 8]

class GRUBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, input_dim)
    def forward(self, x, context_signal=None, h_states=None):
        out, h_new = self.gru(x, h_states)
        return self.readout(out), h_new
    def init_states(self, batch_size, device):
        return torch.zeros(1, batch_size, HIDDEN_DIM).to(device)

def train_and_capture(agent, lr=0.01, epochs=500):
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_history = []
    final_preds = None
    final_truth = None
    
    for epoch in range(epochs):
        t_steps = torch.linspace(0, 4*np.pi, SEQ_LEN, device=DEVICE).view(1, SEQ_LEN, 1)
        x = torch.sin(t_steps + torch.rand(BATCH_SIZE, 1, 1, device=DEVICE)*2*np.pi)
        
        h = agent.init_states(BATCH_SIZE, DEVICE)
        preds = []
        for t in range(SEQ_LEN):
            y_p, h = agent(x[:, t:t+1, :], h_states=h)
            if isinstance(y_p, tuple): y_p = y_p[0] 
            if y_p.dim()==2: y_p = y_p.unsqueeze(1)
            preds.append(y_p)
        
        pred_seq = torch.cat(preds, dim=1)
        loss = criterion(pred_seq, x)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        loss_history.append(loss.item())
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

        if epoch == epochs - 1:
            final_preds = pred_seq[0].detach().cpu().numpy()
            final_truth = x[0].detach().cpu().numpy()
    
    return loss.item(), loss_history, final_preds, final_truth

def run_benchmark():
    print(f"--- [U-MATRIX 7-OCTAVE SPECTRUM | UID: {UID}] ---")
    print(f"Config: Hidden={HIDDEN_DIM}, Spin={SPIN_MULTIPLIER}, Harmonics=7-Octave")
    lr = args.lr
    
    print("Training ROUND U-Matrix...")
    um = UMatrix(sensor_dim=1, logic_dim=1, hidden_dim=HIDDEN_DIM, 
                 persistence_sensor=0.5, persistence_logic=0.9, 
                 quantization_strength=0.0,
                 harmonics=HARMONICS,
                 spin_multiplier=SPIN_MULTIPLIER).to(DEVICE)
                 
    um_loss, um_hist, um_preds, truth = train_and_capture(um, lr=lr)
    
    print("Training GRU Baseline...")
    gru = GRUBaseline(1, HIDDEN_DIM).to(DEVICE)
    gru_loss, gru_hist, gru_preds, _ = train_and_capture(gru, lr=lr)
    
    print(f"Final ROUND Loss: {um_loss:.6f} | GRU Loss: {gru_loss:.6f}")
    
    # --- VISUALIZATION ---
    try:
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Panel A: Loss
        ax1.plot(um_hist, color='#00FF00', linewidth=2, label='ROUND (Green)')
        ax1.plot(gru_hist, color='#4B4BFF', linewidth=2, label='GRU (Blue)')
        ax1.set_title("A. Harmonic Resonance Convergence", color='white')
        ax1.set_xlabel("Epochs", color='white')
        ax1.set_ylabel("MSE Loss", color='white')
        ax1.legend(facecolor='black', edgecolor='white', labelcolor='white')
        ax1.grid(True, alpha=0.3)
        
        # Panel B: Prediction Overlay
        ax2.plot(truth, color='white', linestyle='-', linewidth=2, label='Ground Truth')
        ax2.plot(um_preds, color='#00FF00', linestyle='--', linewidth=2, label='ROUND Pred')
        ax2.plot(gru_preds, color='#FF4B4B', linestyle=':', linewidth=2, label='GRU Pred')
        ax2.set_title(f"B. Sequence Lock (Hidden={HIDDEN_DIM})", color='white')
        ax2.legend(facecolor='black', edgecolor='white', labelcolor='white')
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f"U-Matrix 7-Octave Spectrum: ROUND vs GRU | UID: {UID}", color='white', fontsize=14)
        
        path = os.path.join(OUTPUT_DIR, f"u_matrix_7octave_{UID}.png")
        plt.savefig(path, facecolor='black', edgecolor='none')
        print(f"Plot saved to: {path}")
        
    except Exception as e:
        print(f"Visualization Failed: {e}")

if __name__ == "__main__":
    run_benchmark()
