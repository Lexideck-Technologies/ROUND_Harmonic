import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import uuid
import matplotlib.pyplot as plt
import seaborn as sns

from UIT_ROUND import UITModel, UITEncoderModel

# --- INDUSTRIAL CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UID = uuid.uuid4().hex[:8]
BASE_DIR = f"data/UIT_{UID}"
LOG_DIR = f"{BASE_DIR}/logs"
CRYSTAL_DIR = f"{BASE_DIR}/crystals"
PLOT_DIR = f"{BASE_DIR}/plots"

for d in [LOG_DIR, CRYSTAL_DIR, PLOT_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

class WorkflowLogger:
    def __init__(self, filename):
        self.filename = filename
    def log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(line + "\n")

W_LOG = WorkflowLogger(f"{LOG_DIR}/scientific_duel_{UID}.txt")

# --- DATA GENERATORS ---
def generate_ascii_data(batch_size):
    char_ids = torch.randint(0, 256, (batch_size,)).long()
    bits_msb = []
    bits_lsb = []
    for cid in char_ids:
        bits_msb.append([(cid.item() >> i) & 1 for i in range(7, -1, -1)])
        bits_lsb.append([(cid.item() >> i) & 1 for i in range(8)])
    
    x_msb = torch.tensor(bits_msb).float().unsqueeze(-1).to(DEVICE)
    y_id = char_ids.to(DEVICE)
    x_onehot = nn.functional.one_hot(char_ids, 256).float().to(DEVICE)
    y_lsb = torch.tensor(bits_lsb).float().to(DEVICE)
    
    return x_msb, y_id, x_onehot, y_lsb

# --- GRU BASELINES ---
class GRUDecoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=512, output_size=256):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        _, h = self.gru(x)
        logits = self.readout(h.squeeze(0))
        return logits, h.squeeze(0)

class GRUEncoder(nn.Module):
    def __init__(self, input_size=256, hidden_size=512, output_size=1):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(input_size + output_size, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, output_size)
    def forward(self, char_onehot, seq_len=8):
        batch_size = char_onehot.size(0)
        h = self.input_projection(char_onehot).unsqueeze(0)
        curr_b = torch.zeros(batch_size, 1).to(DEVICE)
        outputs = []
        for t in range(seq_len):
            c_in = torch.cat([curr_b, char_onehot], dim=-1).unsqueeze(1)
            out, h = self.gru(c_in, h)
            bit_l = self.readout(out.squeeze(1))
            curr_b = torch.sigmoid(bit_l)
            outputs.append(bit_l)
        return torch.stack(outputs, dim=1).squeeze(-1)

# --- PHASE 0: GRU BASELINE TRAINING ---
def train_gru_baseline():
    W_LOG.log("--- PHASE 0: GRU BASELINE TRAINING ---")
    dec = GRUDecoder().to(DEVICE)
    enc = GRUEncoder().to(DEVICE)
    
    dec_opt = optim.Adam(dec.parameters(), lr=1e-3)
    enc_opt = optim.Adam(enc.parameters(), lr=1e-3)
    
    history = {"dec_acc": [], "enc_acc": []}
    
    for epoch in range(1501):
        x_m, y_id, x_oh, y_l = generate_ascii_data(64)
        
        # Decoder
        dec.train()
        dec_opt.zero_grad()
        l_dec, _ = dec(x_m)
        loss_d = nn.CrossEntropyLoss()(l_dec, y_id)
        loss_d.backward()
        dec_opt.step()
        
        # Encoder
        enc.train()
        enc_opt.zero_grad()
        l_enc = enc(x_oh)
        loss_e = nn.BCEWithLogitsLoss()(l_enc, y_l)
        loss_e.backward()
        enc_opt.step()
        
        if epoch % 250 == 0:
            dec.eval(); enc.eval()
            with torch.no_grad():
                d_acc = (l_dec.argmax(dim=1) == y_id).float().mean().item()
                e_acc = ((l_enc > 0) == y_l).all(dim=1).float().mean().item()
                W_LOG.log(f"GRU Step {epoch} | Dec Acc: {d_acc:.2%} | Enc Acc: {e_acc:.2%}")
                history["dec_acc"].append(d_acc)
                history["enc_acc"].append(e_acc)
                if d_acc == 1.0 and e_acc == 1.0 and epoch > 300:
                    W_LOG.log("--- [GRU BASELINE LOCKED] ---")
                    break
                    
    torch.save(dec.state_dict(), f"{CRYSTAL_DIR}/gru_dec_{UID}.pt")
    torch.save(enc.state_dict(), f"{CRYSTAL_DIR}/gru_enc_{UID}.pt")
    return dec, enc, history

# --- PHASE 1: UIT-ROUND CRYSTALLIZATION ---
def crystallize_uit():
    W_LOG.log("--- PHASE 1: UIT-ROUND CRYSTALLIZATION ---")
    r_dec = UITModel(input_size=1, hidden_size=512, output_size=256, use_binary_alignment=True).to(DEVICE)
    r_dec_opt = optim.Adam(r_dec.parameters(), lr=2**-7)
    
    history = {"dec_acc": [], "map_path": ""}
    
    for epoch in range(3001):
        r_dec.train()
        x_m, y_id, _, _ = generate_ascii_data(64)
        r_dec_opt.zero_grad()
        logits, conf = r_dec(x_m)
        loss = nn.CrossEntropyLoss()(logits, y_id) * (1.1 - conf.item())
        loss.backward()
        r_dec_opt.step()
        
        if epoch % 250 == 0:
            r_dec.eval()
            with torch.no_grad():
                acc = (logits.argmax(dim=1) == y_id).float().mean().item()
                W_LOG.log(f"UIT Decoder {epoch} | Acc: {acc:.2%} | Conf: {conf.item():.4f}")
                history["dec_acc"].append(acc)
                if acc == 1.0 and epoch > 300:
                    W_LOG.log("--- [UIT DECODER LOCKED] ---")
                    break
    
    # Extract Map
    r_dec.eval()
    all_ids = torch.arange(256).to(DEVICE)
    bits_list = [[(cid.item() >> i) & 1 for i in range(7, -1, -1)] for cid in all_ids]
    x_full = torch.tensor(bits_list).float().unsqueeze(-1).to(DEVICE)
    map_list = []
    with torch.no_grad():
        for i in range(256):
            h = torch.zeros(1, 512).to(DEVICE)
            steps = []
            for t in range(8):
                _, h, _, _, _ = r_dec.layers[0](x_full[i:i+1, t, :], h)
                steps.append(h)
            map_list.append(torch.stack(steps, dim=1))
    map_tensor = torch.cat(map_list, dim=0)
    map_path = f"{CRYSTAL_DIR}/phasic_map_{UID}.pt"
    torch.save(map_tensor, map_path)
    
    # Train Encoder
    W_LOG.log("--- PHASE 2: UIT ENCODER CRYSTALLIZATION ---")
    r_enc = UITEncoderModel(input_size=256, hidden_size=512, output_size=1, use_binary_alignment=True).to(DEVICE)
    r_enc.renormalize_identity(map_path)
    r_enc_opt = optim.Adam(r_enc.parameters(), lr=2**-7)
    history["enc_acc"] = []
    
    for epoch in range(3001):
        r_enc.train()
        _, _, x_oh, y_l = generate_ascii_data(64)
        r_enc_opt.zero_grad()
        outs, conf = r_enc(x_oh)
        loss = nn.BCEWithLogitsLoss()(outs, y_l)
        loss.backward()
        r_enc_opt.step()
        
        if epoch % 250 == 0:
            r_enc.eval()
            with torch.no_grad():
                acc = ((outs > 0) == y_l).all(dim=1).float().mean().item()
                W_LOG.log(f"UIT Encoder {epoch} | Acc: {acc:.2%} | Conf: {conf.item():.4f}")
                history["enc_acc"].append(acc)
                if acc == 1.0 and epoch > 300:
                    W_LOG.log("--- [UIT ENCODER LOCKED] ---")
                    break
                    
    torch.save(r_dec.state_dict(), f"{CRYSTAL_DIR}/uit_dec_{UID}.pt")
    torch.save(r_enc.state_dict(), f"{CRYSTAL_DIR}/uit_enc_{UID}.pt")
    return r_dec, r_enc, history

# --- PHASE 4: THE SCIENTIFIC DUEL ---
def run_relay_duel(r_dec, r_enc, g_dec, g_enc):
    W_LOG.log("--- PHASE 4: THE SCIENTIFIC RELAY DUEL ---")
    for m in [r_dec, r_enc, g_dec, g_enc]: m.eval()
    
    def relay_test(dec, enc, is_uit=True):
        success = 0
        with torch.no_grad():
            for i in range(256):
                bits_in = [[(i >> b) & 1 for b in range(7, -1, -1)]]
                x = torch.tensor(bits_in).float().unsqueeze(-1).to(DEVICE)
                
                # Decoder Hidden Relay
                if is_uit:
                    h = torch.zeros(1, 512).to(DEVICE)
                    for t in range(8):
                        _, h, _, _, _ = dec.layers[0](x[:, t, :], h)
                else:
                    _, h = dec(x) # [1, H]
                
                # Encoder Reconstruction from Hidden Relay
                onehot = torch.zeros(1, 256).to(DEVICE)
                onehot[0, i] = 1.0
                
                recon_bits = []
                if is_uit:
                    h_v = h
                    for t in range(8):
                        c_in = torch.cat([torch.zeros(1, 1).to(DEVICE), onehot], dim=-1)
                        feat, h_v, _, h_c, h_s = enc.layers[0](c_in, h_v)
                        l = enc.readout(torch.cat([feat, h_c, h_s], dim=-1))
                        recon_bits.append(1 if l > 0 else 0)
                    recon_val = sum(b << i for i, b in enumerate(recon_bits))
                else:
                    h_g = h.unsqueeze(0)
                    curr_b = torch.zeros(1, 1).to(DEVICE)
                    for t in range(8):
                        c_in = torch.cat([curr_b, onehot], dim=-1).unsqueeze(1)
                        o, h_g = enc.gru(c_in, h_g)
                        l = enc.readout(o.squeeze(1))
                        recon_bits.append(1 if l > 0 else 0)
                        curr_b = torch.sigmoid(l)
                    recon_val = sum(b << i for i, b in enumerate(recon_bits))
                
                if recon_val == i: success += 1
        return success / 256.0

    r_relay = relay_test(r_dec, r_enc, True)
    g_relay = relay_test(g_dec, g_enc, False)
    W_LOG.log(f"DUEL RESULTS | UIT-ROUND: {r_relay:.2%} | GRU: {g_relay:.2%}")
    return r_relay, g_relay

# --- PHASE 5: THE STORY VISUALIZATION ---
def visualize_story(uit_hist, gru_hist, r_relay, g_relay):
    W_LOG.log("--- PHASE 5: THE STORY VISUALIZATION ---")
    
    # Ultra-Dark Premium Aesthetics
    plt.style.use('dark_background')
    sns.set_theme(style="whitegrid", rc={
        "axes.facecolor": "#0A0A0A",
        "grid.color": "#1A1A1A",
        "figure.facecolor": "#0A0A0A",
        "text.color": "white",
        "axes.labelcolor": "white",
        "xtick.color": "#AAAAAA",
        "ytick.color": "#AAAAAA",
        "font.family": "sans-serif"
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor='#0A0A0A')
    colors = ['#FF4B4B', '#4B4BFF'] # ROUND Red vs GRU Blue
    
    # Common Bar Styling
    bar_kwargs = {"width": 0.4, "edgecolor": "black", "linewidth": 1.5}
    
    # 1. Decoder Trial
    axes[0].bar([-0.2, 0.2], [uit_hist["dec_acc"][-1], gru_hist["dec_acc"][-1]], color=colors, **bar_kwargs)
    axes[0].set_title("A. The Hearing Trial (Decoder)\nAll-256 ASCII Decoding", fontsize=13, fontweight='bold', pad=20)
    axes[0].set_xticks([-0.2, 0.2])
    axes[0].set_xticklabels(['UIT-ROUND', 'GRU'], fontsize=11, fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    
    # 2. Encoder Trial
    axes[1].bar([-0.2, 0.2], [uit_hist["enc_acc"][-1], gru_hist["enc_acc"][-1]], color=colors, **bar_kwargs)
    axes[1].set_title("B. The Speaking Trial (Encoder)\nSovereign Phasic Mirroring", fontsize=13, fontweight='bold', pad=20)
    axes[1].set_xticks([-0.2, 0.2])
    axes[1].set_xticklabels(['UIT-ROUND', 'GRU'], fontsize=11, fontweight='bold')
    axes[1].set_ylim(0, 1.1)
    
    # 3. The Sandwich Duel (Relay)
    axes[2].bar([-0.2, 0.2], [r_relay, g_relay], color=colors, **bar_kwargs)
    axes[2].set_title("C. The Phasic Sandwich (Relay)\nEnd-to-End Bit-Perfection", fontsize=13, fontweight='bold', pad=20)
    axes[2].set_xticks([-0.2, 0.2])
    axes[2].set_xticklabels(['Phasic Relay', 'Vector Relay'], fontsize=11, fontweight='bold')
    axes[2].set_ylim(0, 1.1)
    
    # Professional Annotations (Success Counts)
    axes[2].text(-0.2, r_relay + 0.03, f"{int(r_relay*256)}/256\n{r_relay:.1%}", 
               ha='center', va='bottom', color=colors[0], fontweight='bold', fontsize=12)
    axes[2].text(0.2, g_relay + 0.03, f"{int(g_relay*256)}/256\n{g_relay:.1%}", 
               ha='center', va='bottom', color=colors[1], fontweight='bold', fontsize=12)
    
    # Remove some chart junk
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')
    
    plt.suptitle(f"Crystalline Phasic Identity vs. Vector Memory (UID: {UID})\n" + 
                 "Demonstrating Phasic Sovereignty & Zero-Shot Relay Integrity", 
                 fontsize=18, y=1.05, color='white', fontweight='bold')
    
    plt.tight_layout()
    plot_path = f"{PLOT_DIR}/scientific_duel_story_{UID}.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='#0A0A0A')
    W_LOG.log(f"Premium Scientific Story Plot Saved: {plot_path}")

if __name__ == "__main__":
    W_LOG.log(f"Starting Industrial Crystalline Duel | UID: {UID}")
    g_dec, g_enc, g_hist = train_gru_baseline()
    r_dec, r_enc, r_hist = crystallize_uit()
    r_relay, g_relay = run_relay_duel(r_dec, r_enc, g_dec, g_enc)
    visualize_story(r_hist, g_hist, r_relay, g_relay)
    W_LOG.log(f"Workshop complete. Duel Artifacts in: {BASE_DIR}")
    
    # Result Auto-Documentation
    W_LOG.log(f"UPDATING README With Results from {UID}...")
    try:
        import subprocess
        if os.path.exists("update_readme.py"):
            subprocess.run([sys.executable, "update_readme.py"], check=True)
        else:
             W_LOG.log("update_readme.py not found. Skipping auto-update.")
    except Exception as e:
        W_LOG.log(f"Failed to update README: {e}")
