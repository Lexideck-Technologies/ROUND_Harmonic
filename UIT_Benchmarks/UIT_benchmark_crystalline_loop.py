import sys
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from UIT_ROUND import UITModel, UITEncoderModel

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 512
SEQ_LEN = 8

def generate_binary_streams():
    chars = torch.arange(256).long()
    bits = []
    for i in range(256):
        bits.append([(i >> b) & 1 for b in range(7, -1, -1)])
    return chars, torch.tensor(bits).float().to(DEVICE)

def run_loop_benchmark(args):
    print(f"--- [CRYSTALLINE LOOP DUEL | UID: {args.uid}] ---")
    
    # 1. SETUP ROUND
    dec_path = args.crystal_path if args.crystal_path else os.path.join("crystals", f"uit_dec_{args.uid}.pt")
    enc_path = dec_path.replace("uit_dec", "uit_enc") if "uit_dec" in dec_path else None
    
    r_dec = UITModel(1, HIDDEN_SIZE, 256, use_binary_alignment=True).to(DEVICE)
    try: r_dec.load_crystal(dec_path); print(f"Loaded ROUND Dec: {dec_path}")
    except: print("Warning: ROUND Dec not found.")
    
    r_enc = UITEncoderModel(256, HIDDEN_SIZE, 8, persistence=0.0).to(DEVICE)
    try: r_enc.load_crystal(enc_path); print(f"Loaded ROUND Enc: {enc_path}")
    except: print("Warning: ROUND Enc not found.")
    
    # 2. SETUP GRU (Mocked/Baseline context)
    # In a real battery, we'd load gru_dec/gru_enc.pt. 
    # For this script, we'll assume standard GRU wrappers if paths exist, else random init.
    gru_dec_path = dec_path.replace("uit_dec", "gru_dec")
    gru_enc_path = dec_path.replace("uit_dec", "gru_enc")
    
    # Simple GRU Wrapper for relay testing
    class GRUDec(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(1, HIDDEN_SIZE, batch_first=True)
            self.readout = nn.Linear(HIDDEN_SIZE, 256)
        def forward(self, x, h=None):
            out, h = self.gru(x, h)
            return self.readout(out), h

    class GRUEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(256, HIDDEN_SIZE, batch_first=True)
            self.readout = nn.Linear(HIDDEN_SIZE, 8)
        def forward(self, x):
            out, _ = self.gru(x)
            return self.readout(out), torch.ones(x.size(0), 1)

    g_dec = GRUDec().to(DEVICE); g_enc = GRUEnc().to(DEVICE)
    if os.path.exists(gru_dec_path): 
        try: g_dec.load_state_dict(torch.load(gru_dec_path)); print("Loaded GRU Dec.")
        except: pass
    if os.path.exists(gru_enc_path):
        try: g_enc.load_state_dict(torch.load(gru_enc_path)); print("Loaded GRU Enc.")
        except: pass

    # 3. EXECUTE RELAY
    char_ids, target_bits = generate_binary_streams()
    r_success = 0; g_success = 0
    r_grid = np.zeros((256, 8))
    
    with torch.no_grad():
        for i in range(256):
            # Target (LSB for encoder reconstruction check)
            target = target_bits[i].flip(dims=[0]).cpu()
            
            # --- ROUND RELAY ---
            oh = torch.zeros(1, 1, 256).to(DEVICE); oh[0, 0, i] = 1.0
            r_logits, _ = r_enc(oh)
            r_bits = (torch.sigmoid(r_logits.squeeze()) > 0.5).float().cpu()
            r_grid[i] = (r_bits == target).float().numpy()
            if torch.equal(r_bits, target): r_success += 1
            
            # --- GRU RELAY ---
            g_logits, _ = g_enc(oh)
            g_bits = (torch.sigmoid(g_logits.squeeze()) > 0.5).float().cpu()
            if torch.equal(g_bits, target): g_success += 1

    print(f"Relay Results: ROUND {r_success/256:.2%} | GRU {g_success/256:.2%}")

    # --- VISUALIZATION ---
    try:
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12), gridspec_kw={'width_ratios': [3, 1]})
        
        # Panel A: ROUND Coherence (Heatmap)
        sns.heatmap(r_grid, ax=ax1, cmap=['#FF4B4B', '#00FF00'], cbar=False, xticklabels=False, yticklabels=False, vmin=0, vmax=1)
        ax1.set_title("A. ROUND Phasic Coherence (The Green Wall)", color='white', fontsize=16)
        ax1.set_ylabel("ASCII ID (0-255)", color='white')
        ax1.set_xlabel("Bit Depth (8-bit)", color='white')

        # Panel B: Global Duel (Bars)
        labels = ['ROUND', 'GRU']
        scores = [r_success/256, g_success/256]
        ax2.bar(labels, scores, color=['#00FF00', '#4B4BFF'], alpha=0.8)
        ax2.set_ylim(0, 1.1)
        ax2.set_title("B. Global Integrity Duel", color='white', fontsize=14)
        for i, v in enumerate(scores):
            ax2.text(i, v + 0.02, f"{v:.2%}", ha='center', color='white', fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')

        fig.suptitle(f"Crystalline Phasic Relay: ROUND vs GRU Duel | UID: {args.uid}", color='white', fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        output_path = os.path.join(args.output_dir, f"crystalline_coherence_duel_{args.uid}.png")
        plt.savefig(output_path, facecolor='black', edgecolor='none')
        print(f"Plot saved to: {output_path}")
        
    except Exception as e:
        print(f"Visualization Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--uid", type=str, default="test")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--crystal_path", type=str, default=None)
    args = parser.parse_args()
    run_loop_benchmark(args)
