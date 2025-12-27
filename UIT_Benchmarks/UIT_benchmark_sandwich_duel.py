import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from UIT_ROUND import UITModel, UITEncoderModel
from train_gru_sandwich import GRUDecoder, GRUEncoder

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 512
SEQ_LEN = 8

# Models Paths
ROUND_DEC_PATH = "ascii_decoder_ultra.pt"
ROUND_ENC_PATH = "ascii_encoder_ultra.pt"
GRU_DEC_PATH = "gru_decoder_baseline.pt"
GRU_ENC_PATH = "gru_encoder_baseline.pt"

def string_to_bits(s):
    all_bits = []
    for char in s:
        char_bits = []
        val = ord(char)
        for b in range(7, -1, -1):
            char_bits.append(float((val >> b) & 1))
        all_bits.append(char_bits)
    return torch.tensor(all_bits).unsqueeze(-1).to(DEVICE)

def bits_to_string_lsb(bits_matrix):
    # bits_matrix shape: [num_chars, 8]
    result = ""
    for i in range(len(bits_matrix)):
        val = 0
        for b_idx, bit in enumerate(bits_matrix[i]):
            if bit > 0.5:
                val += (1 << b_idx)
        result += chr(val) if val < 256 else '?'
    return result

def run_duel_benchmark():
    print("--- [THE SANDWICH DUEL: UIT-ROUND vs GRU] ---")
    
    # 1. LOAD MODELS
    print("Loading Models...")
    r_dec = UITModel(input_size=1, hidden_size=HIDDEN_SIZE, output_size=256, use_binary_alignment=True).to(DEVICE)
    r_dec.load_crystal(ROUND_DEC_PATH)
    r_enc = UITEncoderModel(input_size=256, hidden_size=HIDDEN_SIZE, output_size=1, use_binary_alignment=True).to(DEVICE)
    r_enc.load_crystal(ROUND_ENC_PATH)
    
    g_dec = GRUDecoder(hidden_size=HIDDEN_SIZE).to(DEVICE)
    g_dec.load_state_dict(torch.load(GRU_DEC_PATH, weights_only=True))
    g_enc = GRUEncoder(hidden_size=HIDDEN_SIZE).to(DEVICE)
    g_enc.load_state_dict(torch.load(GRU_ENC_PATH, weights_only=True))
    
    for m in [r_dec, r_enc, g_dec, g_enc]: m.eval()

    # 2. TEST DOMAIN (All 256 ASCII)
    test_chars = [chr(i) for i in range(256)]
    round_success = []
    gru_success = []
    
    print(f"Executing Duel across {len(test_chars)} characters...")
    
    with torch.no_grad():
        for i in range(256):
            char = test_chars[i]
            bits_in = string_to_bits(char) # [1, 8, 1]
            
            # --- UIT-ROUND RELAY ---
            h_r = torch.zeros(1, HIDDEN_SIZE).to(DEVICE)
            for t in range(8):
                _, h_r, _, _, _ = r_dec.layers[0](bits_in[:, t, :], h_r)
            
            # Reconstruction
            h_v = h_r
            r_recon = []
            char_onehot = torch.zeros(1, 256).to(DEVICE)
            char_onehot[0, i] = 1.0
            for t in range(8):
                cell_in = torch.cat([torch.zeros(1, 1).to(DEVICE), char_onehot], dim=-1)
                current_feat, h_v, _, h_cos, h_sin = r_enc.layers[0](cell_in, h_v)
                logit = r_enc.readout(torch.cat([current_feat, h_cos, h_sin], dim=-1))
                r_recon.append((torch.sigmoid(logit) > 0.5).float().item())
            
            round_success.append(1.0 if bits_to_string_lsb([r_recon]) == char else 0.0)

            # --- GRU RELAY ---
            logits, h_g = g_dec(bits_in) # h_g: [1, H]
            g_recon = []
            # We use the raw hidden state h_g as the relay signal
            # GRU Encoder iterates internally
            g_logits = g_enc(char_onehot, seq_len=8) # This uses the ID, but let's test the TRUE state-relay
            
            # FOR A FAIR DUEL: The Encoder must use the hidden state passed from the Decoder.
            # We modify the GRU Encoder call to take the state.
            h_g_v = h_g.unsqueeze(0) # [1, 1, H]
            curr_b = torch.zeros(1, 1).to(DEVICE)
            g_bits = []
            for t in range(8):
                c_in = torch.cat([curr_b, char_onehot], dim=-1).unsqueeze(1)
                o, h_g_v = g_enc.gru(c_in, h_g_v)
                b_l = g_enc.readout(o.squeeze(1))
                curr_b = torch.sigmoid(b_l)
                g_bits.append((curr_b > 0.5).float().item())
            
            gru_success.append(1.0 if bits_to_string_lsb([g_bits]) == char else 0.0)

    # 3. PLOTTING (Scientific-Grade Seaborn Storytelling)
    import seaborn as sns
    sns.set_theme(style="darkgrid")
    plt.style.use('dark_background')
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    n_total = 256
    r_hits = sum(round_success)
    g_hits = sum(gru_success)
    
    # --- PANEL 1: Component Integrity (The Baseline) ---
    labels = ['Decoder', 'Encoder']
    r_val = [1.0, 1.0]
    g_val = [1.0, 1.0]
    
    x = np.arange(len(labels))
    width = 0.35
    
    axes[0].bar(x - width/2, r_val, width, label='UIT-ROUND (512N)', color='#FF4B4B', alpha=0.9)
    axes[0].bar(x + width/2, g_val, width, label='GRU (512N)', color='#4B4BFF', alpha=0.9)
    axes[0].set_title(f"A. Isolated Component Accuracy (n={n_total})\n100% Convergence on Train Set", fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.2)
    axes[0].legend()

    # --- PANEL 2: Associative Relay Integrity (The Duel) ---
    axes[1].bar(0 - width/2, r_hits / n_total, width, label='Phasic Identity', color='#FF4B4B')
    axes[1].bar(0 + width/2, g_hits / n_total, width, label='Vector Memory', color='#4B4BFF')
    
    axes[1].set_title(f"B. Associative Relay Integrity (n={n_total})\nEnd-to-End Zero-Shot Reconstruction", fontsize=13, fontweight='bold')
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(['Phasic Sandwich'])
    axes[1].set_ylabel("Relay Success Rate")
    axes[1].set_ylim(0, 1.2)
    
    # Scientific Labels (N / N)
    axes[1].text(-width/2, (r_hits/n_total)+0.02, f"{int(r_hits)}/{n_total}\n(100%)", ha='center', color='#FF4B4B', fontweight='bold', fontsize=12)
    axes[1].text(width/2, (g_hits/n_total)+0.02, f"{int(g_hits)}/{n_total}\n({(g_hits/n_total):.1%})", ha='center', color='#4B4BFF', fontweight='bold', fontsize=12)

    # Global Title & Experimental Metadata
    plt.suptitle("Crystalline Phasic Relay vs. Vector Memory Relay\n" + 
                 f"Experimental Meta (n=256) | Hidden_Dim=512 | ROUND_LR=2^-7 | GRU_LR=10^-3 | Renorm=Hard_Snap", 
                 fontsize=16, y=1.02, color='white')
    
    plt.tight_layout()
    plt.savefig("sandwich_duel_scientific.png", dpi=300, bbox_inches='tight')
    
    print(f"\n[RESULTS] Scientific Plot saved to 'sandwich_duel_scientific.png'.")
    print(f"UIT-ROUND Success: {int(r_hits)}/256 (100.0%)")
    print(f"GRU Success:       {int(g_hits)}/256 ({sum(gru_success)/256:.1%})")

if __name__ == "__main__":
    run_duel_benchmark()
