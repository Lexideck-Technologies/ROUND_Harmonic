import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from UIT_ROUND import UITModel

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default=".")
parser.add_argument("--uid", type=str, default="color_duel")
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--crystal_path", type=str, default=None) # Included for battery compatibility
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = args.output_dir
UID = args.uid
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- CONFIGURATION ---
BATCH_SIZE = 64
EPOCHS = 1000 
LEARNING_RATE = args.lr if args.lr is not None else 0.0078125 # Default 2^-7
HIDDEN_SIZE = 64 
NUM_COLORS = 64 

# --- DATA GENERATION ---
def get_color_phase(idx):
    return (idx / NUM_COLORS) * 2 * np.pi

def get_ground_truth_mixture(idx_a, idx_b):
    phi_a = get_color_phase(idx_a)
    phi_b = get_color_phase(idx_b)
    vec_a = np.array([np.cos(phi_a), np.sin(phi_a)])
    vec_b = np.array([np.cos(phi_b), np.sin(phi_b)])
    vec_mid = (vec_a + vec_b) / 2.0
    norm = np.linalg.norm(vec_mid)
    if norm < 1e-6: return (idx_a + idx_b) // 2 
    vec_mid = vec_mid / norm
    phi_mid = np.arctan2(vec_mid[1], vec_mid[0])
    if phi_mid < 0: phi_mid += 2 * np.pi
    return np.argmin([np.abs(get_color_phase(i) - phi_mid) for i in range(NUM_COLORS)])

def generate_color_data(batch_size):
    idx_a = torch.randint(0, NUM_COLORS, (batch_size,))
    idx_b = torch.randint(0, NUM_COLORS, (batch_size,))
    x = torch.zeros(batch_size, 2, NUM_COLORS)
    x.scatter_(2, idx_a.unsqueeze(1).unsqueeze(2), 1.0)
    x.scatter_(2, idx_b.unsqueeze(1).unsqueeze(2), 1.0)
    targets = [get_ground_truth_mixture(idx_a[i].item(), idx_b[i].item()) for i in range(batch_size)]
    return x.to(DEVICE), torch.tensor(targets).long().to(DEVICE)

# --- MODELS ---
class ColorROUND(nn.Module):
    def __init__(self):
        super().__init__()
        self.uit = UITModel(input_size=NUM_COLORS, hidden_size=HIDDEN_SIZE, output_size=NUM_COLORS, num_layers=1, persistence=0.5)
        self.classifier = nn.Linear(HIDDEN_SIZE * 3, NUM_COLORS)
    def forward(self, x):
        h = torch.zeros(x.size(0), HIDDEN_SIZE).to(DEVICE)
        _, h, _, _, _ = self.uit.layers[0](x[:, 0, :], h)
        feat_2, h, _, h_cos_2, h_sin_2 = self.uit.layers[0](x[:, 1, :], h)
        combined = torch.cat([feat_2, h_cos_2, h_sin_2], dim=-1)
        return self.classifier(combined)

class ColorGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(NUM_COLORS, HIDDEN_SIZE, batch_first=True)
        self.classifier = nn.Linear(HIDDEN_SIZE, NUM_COLORS)
    def forward(self, x):
        _, h = self.gru(x) # h: [1, B, H]
        return self.classifier(h.squeeze(0))

def run_benchmark():
    print(f"--- [UIT-ROUND vs GRU: COLOR ALGEBRA DUEL | UID: {UID}] ---")
    round_model = ColorROUND().to(DEVICE)
    gru_model = ColorGRU().to(DEVICE)
    
    r_opt = optim.Adam(round_model.parameters(), lr=LEARNING_RATE)
    g_opt = optim.Adam(gru_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    history = {"round": [], "gru": []}
    
    for epoch in range(EPOCHS):
        # Training
        round_model.train(); gru_model.train()
        x, y = generate_color_data(BATCH_SIZE)
        
        # ROUND
        r_opt.zero_grad(); r_loss = criterion(round_model(x), y); r_loss.backward(); r_opt.step()
        # GRU
        g_opt.zero_grad(); g_loss = criterion(gru_model(x), y); g_loss.backward(); g_opt.step()
        
        if epoch % 50 == 0:
            round_model.eval(); gru_model.eval()
            with torch.no_grad():
                vx, vy = generate_color_data(100)
                r_acc = (torch.argmax(round_model(vx), dim=1) == vy).float().mean().item()
                g_acc = (torch.argmax(gru_model(vx), dim=1) == vy).float().mean().item()
                history["round"].append(r_acc)
                history["gru"].append(g_acc)
                print(f"Epoch {epoch:4d} | ROUND: {r_acc:7.2%} | GRU: {g_acc:7.2%}")
            if r_acc > 0.99 and g_acc > 0.99: break

    # --- VISUALIZATION ---
    try:
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Panel A: Learning Curves
        ax1.plot(history["round"], color='#00FF00', linewidth=3, label='ROUND (Green)')
        ax1.plot(history["gru"], color='#4B4BFF', linewidth=3, label='GRU (Blue)')
        ax1.set_title("A. Learning Curves (Crayola-64 Mixture)", color='white', fontsize=12)
        ax1.set_xlabel("Epochs (x50)", color='white')
        ax1.set_ylabel("Accuracy", color='white')
        ax1.grid(True, alpha=0.3)
        ax1.legend(facecolor='black', edgecolor='white', labelcolor='white')
        
        # Panel B: Final Performance
        labels = ['ROUND', 'GRU']
        accs = [history["round"][-1], history["gru"][-1]]
        ax2.bar(labels, accs, color=['#00FF00', '#4B4BFF'], alpha=0.8)
        ax2.set_ylim(0, 1.1)
        ax2.set_title("B. Final Accuracy Duel", color='white', fontsize=12)
        ax2.set_ylabel("Accuracy", color='white')
        for i, v in enumerate(accs):
            ax2.text(i, v + 0.02, f"{v:.2%}", ha='center', color='white', fontweight='bold')

        fig.suptitle(f"Color Algebra Head-to-Head Duel | UID: {UID}", color='white', fontsize=14)
        
        plot_path = os.path.join(OUTPUT_DIR, f"color_algebra_duel_{UID}.png")
        plt.savefig(plot_path, facecolor='black', edgecolor='none')
        print(f"Plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"Visualization Failed: {e}")

    with open(os.path.join(OUTPUT_DIR, f"color_log_{UID}.txt"), "w") as f:
        f.write(f"Final ROUND Acc: {history['round'][-1]:.4f}\n")
        f.write(f"Final GRU Acc: {history['gru'][-1]:.4f}\n")

if __name__ == "__main__":
    run_benchmark()
