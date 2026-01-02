import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from UIT_ROUND import UITModel

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default=".")
parser.add_argument("--uid", type=str, default="trans_duel")
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--crystal_path", type=str, default=None)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = args.output_dir
UID = args.uid
os.makedirs(OUTPUT_DIR, exist_ok=True)

CRYSTAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../crystals'))
DECODER_PATH = args.crystal_path if args.crystal_path else os.path.join(CRYSTAL_DIR, f"uit_dec_{UID}.pt")
# Fallback for GRU if needed (optional)
GRU_PATH = DECODER_PATH.replace("uit_dec", "gru_dec") if "uit_dec" in DECODER_PATH else None

HIDDEN_SIZE = 512
EPOCHS = 500

class GRUDecoderWrapper(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gru = nn.GRU(1, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, 256)
    def forward(self, x):
        _, h = self.gru(x)
        return self.readout(h.squeeze(0)), torch.ones(x.size(0), 1).to(x.device)

class PhasicPrismModel(nn.Module):
    def __init__(self, sanctuary):
        super().__init__()
        self.warp = nn.Linear(1, 1)
        self.sanctuary = sanctuary
        for p in self.sanctuary.parameters(): p.requires_grad = False
    def forward(self, x):
        w = torch.sigmoid(self.warp(x))
        out, conf = self.sanctuary(w)
        return out, conf, w

def run_transparency():
    print(f"--- [v1.3.11 PHASIC TRANSPARENCY DUEL | UID: {UID}] ---")
    
    # 1. SETUP ROUND
    sanctuary_r = UITModel(input_size=1, hidden_size=HIDDEN_SIZE, output_size=256, use_binary_alignment=True).to(DEVICE)
    try:
        sanctuary_r.load_crystal(DECODER_PATH)
        print(f"Loaded ROUND Sanctuary: {DECODER_PATH}")
    except:
        print("Warning: ROUND Sanctuary not found. Using random init.")
    sanctuary_r.eval()
    
    model_r = PhasicPrismModel(sanctuary_r).to(DEVICE)
    opt_r = optim.Adam(model_r.warp.parameters(), lr=0.01)
    
    # 2. SETUP GRU
    sanctuary_g = GRUDecoderWrapper(HIDDEN_SIZE).to(DEVICE)
    if GRU_PATH and os.path.exists(GRU_PATH):
        try:
            sanctuary_g.load_state_dict(torch.load(GRU_PATH))
            print(f"Loaded GRU Sanctuary: {GRU_PATH}")
        except: pass
    sanctuary_g.eval()
    
    model_g = PhasicPrismModel(sanctuary_g).to(DEVICE)
    opt_g = optim.Adam(model_g.warp.parameters(), lr=0.01)
    
    history = {"round": [], "gru": []}
    
    # cuDNN Fix: RNNs must be in train mode to store activations for backward pass, even if weights are frozen.
    model_r.train(); model_g.train()
    
    for epoch in range(EPOCHS):
        x = torch.randn(64, 8, 1).to(DEVICE)
        y = torch.randint(0, 256, (64,)).to(DEVICE)
        
        # ROUND
        opt_r.zero_grad(); r_out, _, _ = model_r(x); r_loss = nn.CrossEntropyLoss()(r_out, y); r_loss.backward(); opt_r.step()
        # GRU
        opt_g.zero_grad(); g_out, _, _ = model_g(x); g_loss = nn.CrossEntropyLoss()(g_out, y); g_loss.backward(); opt_g.step()
        
        if epoch % 100 == 0:
            history["round"].append(r_loss.item())
            history["gru"].append(g_loss.item())
            print(f"Epoch {epoch} | ROUND Loss: {r_loss.item():.4f} | GRU Loss: {g_loss.item():.4f}")

    # --- VISUALIZATION ---
    try:
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Panel A: Loss Convergence
        ax1.plot(history["round"], color='#00FF00', linewidth=3, label='ROUND (Green)')
        ax1.plot(history["gru"], color='#4B4BFF', linewidth=3, label='GRU (Blue)')
        ax1.set_title("A. Transparency Duel: Inverse Mapping Convergence", color='white')
        ax1.set_xlabel("Epochs (x100)", color='white')
        ax1.set_ylabel("Cross Entropy Loss", color='white')
        ax1.legend(facecolor='black', edgecolor='white', labelcolor='white')
        ax1.grid(True, alpha=0.3)
        
        # Panel B: Warp Distribution (ROUND only for clarity)
        with torch.no_grad():
             test_x = torch.randn(256, 8, 1).to(DEVICE)
             _, _, final_w = model_r(test_x)
             flat_w = final_w.cpu().numpy().flatten()
        sns.histplot(flat_w, ax=ax2, color='#00FF00', bins=50, kde=True, label='ROUND Warp')
        ax2.set_title("B. Learned Warp Distribution (ROUND)", color='white')
        ax2.set_xlabel("Warp Value (0-1)", color='white')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(f"UIT Transparency Experiment: Head-to-Head Duel | UID: {UID}", color='white', fontsize=14)
        
        output_path = os.path.join(OUTPUT_DIR, f"transparency_duel_{UID}.png")
        plt.savefig(output_path, facecolor='black', edgecolor='none')
        print(f"Plot saved to: {output_path}")
        
    except Exception as e:
        print(f"Visualization Failed: {e}")

if __name__ == "__main__":
    run_transparency()
