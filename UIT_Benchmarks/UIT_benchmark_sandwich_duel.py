import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from UIT_ROUND import UITModel, UITEncoderModel

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 512
SEQ_LEN = 8

# --- GRU BASELINES (Embedded for Portability) ---
class GRUDecoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=512, output_size=256):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        _, h = self.gru(x); return self.readout(h.squeeze(0)), h.squeeze(0)

class GRUEncoder(nn.Module):
    def __init__(self, input_size=256, hidden_size=512, output_size=1):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(input_size + output_size, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, output_size)
    def forward(self, char_onehot, seq_len=8):
        if char_onehot.dim() == 3: char_onehot = char_onehot.squeeze(1)
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

# --- DATA UTILS ---
def get_all_chars_data():
    # Returns: (bits_input, id_target, onehot_input, bits_target)
    # bits_input: [256, 8, 1] (MB First) for Decoder
    # id_target: [256]
    # onehot_input: [256, 1, 256] for Encoder
    # bits_target: [256, 8] (LSB) for Encoder Check
    
    ids = torch.arange(256).long().to(DEVICE)
    
    # MSB bits for Decoder Input
    bits_msb = []
    for i in range(256):
        bits_msb.append([(i >> b) & 1 for b in range(7, -1, -1)])
    x_dec = torch.tensor(bits_msb).float().unsqueeze(-1).to(DEVICE)
    
    # LSB bits for Encoder Target
    bits_lsb = []
    for i in range(256):
        bits_lsb.append([(i >> b) & 1 for b in range(8)])
    y_enc_bits = torch.tensor(bits_lsb).float().to(DEVICE)
    
    # OneHot for Encoder Input
    x_enc = nn.functional.one_hot(ids, 256).float().unsqueeze(1).to(DEVICE)
    
    return x_dec, ids, x_enc, y_enc_bits

def verify_component(model, x, y, is_decoder=True, is_uit=True):
    model.eval()
    with torch.no_grad():
        if is_decoder:
            # Decoder: X=[256,8,1] -> Y=[256]
            if is_uit:
                logits, _ = model(x)
            else:
                logits, _ = model(x)
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean().item()
        else:
            # Encoder: X=[256,1,256] -> Y=[256, 8]
            if is_uit:
                # UIT Encoder Morning State: onehot -> [1, 1, 8]
                # x is [256, 1, 256]
                out, _ = model(x) # [256, 1, 8]
                out = out.squeeze(1) # [256, 8]
            else:
                # GRU: Expects [Batch, 256]
                out = model(x.squeeze(1)) # [256, 8]
            
            # Bits Check
            pred_bits = (out > 0).float()
            # Row-wise match
            matches = (pred_bits == y).all(dim=1).float()
            acc = matches.mean().item()
    return acc

# --- VISUALIZATION ---
def plot_sandwich_story(results, output_dir, uid):
    # results = {
    #   "r_dec_acc": float, "r_enc_acc": float,
    #   "g_dec_acc": float, "g_enc_acc": float,
    #   "r_relay": float, "g_relay": float
    # }
    
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Colors
    c_round = '#FF4B4B' # Poppin Red
    c_gru = '#4B4BFF'   # Sad Blue
    
    # --- PANEL A: ISOLATED COMPONENTS ---
    ax1.set_title("A. Isolated Component Accuracy (n=256)\n100% Convergence on Train Set", fontsize=12, color='white', fontweight='bold')
    
    labels_a = ['Decoder', 'Encoder']
    x_a = np.arange(len(labels_a))
    width = 0.35
    
    r_vals_a = [results['r_dec_acc'], results['r_enc_acc']]
    g_vals_a = [results['g_dec_acc'], results['g_enc_acc']]
    
    rects1 = ax1.bar(x_a - width/2, r_vals_a, width, label='UIT-ROUND (512N)', color=c_round, edgecolor='white', linewidth=1)
    rects2 = ax1.bar(x_a + width/2, g_vals_a, width, label='GRU (512N)', color=c_gru, edgecolor='white', linewidth=1)
    
    ax1.set_ylabel('Accuracy', color='white')
    ax1.set_xticks(x_a)
    ax1.set_xticklabels(labels_a, color='white')
    ax1.legend(facecolor='black', edgecolor='white', labelcolor='white')
    ax1.set_ylim(0, 1.2)
    ax1.grid(True, axis='y', alpha=0.3, color='gray')
    ax1.axhline(1.0, color='white', linestyle='-', alpha=0.5)

    # --- PANEL B: RELAY INTEGRITY ---
    ax2.set_title("B. Associative Relay Integrity (n=256)\nEnd-to-End Zero-Shot Reconstruction", fontsize=12, color='white', fontweight='bold')
    
    labels_b = ['Phasic Sandwich']
    x_b = np.arange(len(labels_b))
    
    r_val_b = results['r_relay']
    g_val_b = results['g_relay']
    
    rects3 = ax2.bar(x_b - width/2, [r_val_b], width, label='UIT-ROUND', color=c_round, edgecolor='white', linewidth=1)
    rects4 = ax2.bar(x_b + width/2, [g_val_b], width, label='GRU', color=c_gru, edgecolor='white', linewidth=1)
    
    ax2.set_ylabel('Relay Success Rate', color='white')
    ax2.set_xticks(x_b)
    ax2.set_xticklabels(labels_b, color='white')
    ax2.set_ylim(0, 1.2)
    ax2.grid(True, axis='y', alpha=0.3, color='gray')
    ax2.axhline(1.0, color='white', linestyle='-', alpha=0.5)
    
    # Annotations
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1%}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', color='white', fontweight='bold')

    autolabel(rects1, ax1)
    autolabel(rects2, ax1)
    autolabel(rects3, ax2)
    autolabel(rects4, ax2)
    
    # Global Title
    fig.suptitle(f"Crystalline Phasic Relay vs. Vector Memory Relay\nExperimental Meta (n=256) | Hidden=512 | ROUND_LR=2^-7 | UID: {uid}", fontsize=14, color='white')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(output_dir, f"sandwich_duel_story_{uid}.png")
    plt.savefig(path, facecolor='black', edgecolor='none')
    print(f"Plot saved to: {path}")

# --- MAIN ---
def run_duel(args):
    print(f"--- [v1.3.0 SANDWICH DUEL | UID: {args.uid}] ---")
    
    # 1. LOAD MODEL PATHS
    if args.crystal_path:
        # Robust logic: Check existence, ignore case for "dec"/"enc" swap logic
        R_DEC_P = args.crystal_path
        
        # Try to infer Encoder path by swapping 'decoder' -> 'encoder' or 'dec' -> 'enc'
        # Check case-insensitive
        lower_path = args.crystal_path.lower()
        if "decoder" in lower_path:
            # Case-preserving swap if possible, simple otherwise
            if "UIT_ascii_decoder" in args.crystal_path:
                 R_ENC_P = args.crystal_path.replace("decoder", "encoder")
            else:
                 R_ENC_P = args.crystal_path.lower().replace("decoder", "encoder") # Fallback
        elif "dec" in lower_path:
             if "uit_dec" in lower_path:
                 R_ENC_P = args.crystal_path.replace("uit_dec", "uit_enc")
             elif "UIT_dec" in args.crystal_path:
                 R_ENC_P = args.crystal_path.replace("UIT_dec", "UIT_enc")
             else:
                 R_ENC_P = args.crystal_path.lower().replace("dec", "enc")
        else:
             # Assume user pointed to something else, default encoder?
             R_ENC_P = "ascii_encoder_ultra.pt"

        # Find GRU
        c_dir = os.path.dirname(args.crystal_path)
        G_DEC_P = os.path.join(c_dir, f"gru_decoder_baseline_{args.uid}.pt")
        G_ENC_P = os.path.join(c_dir, f"gru_encoder_baseline_{args.uid}.pt")
    else:
        # Default or Fallback
        c_dir = os.path.join(args.output_dir, "crystals") 
        if not os.path.exists(c_dir): c_dir = "."
        # Try to find something reasonable
        R_DEC_P = "ascii_decoder_ultra.pt"
        R_ENC_P = "ascii_encoder_ultra.pt" 
        G_DEC_P = None
        
    print(f"Targeting UIT Crystals:\n D: {R_DEC_P}\n E: {R_ENC_P}")
    
    # 2. INIT MODELS
    r_dec = UITModel(1, HIDDEN_SIZE, 256, use_binary_alignment=True).to(DEVICE)
    r_enc = UITEncoderModel(256, HIDDEN_SIZE, 8, use_binary_alignment=False).to(DEVICE)
    
    g_dec = GRUDecoder().to(DEVICE)
    g_enc = GRUEncoder().to(DEVICE)

    # 3. LOAD WEIGHTS
    try:
        r_dec.load_crystal(R_DEC_P); r_dec.eval()
        r_enc.load_crystal(R_ENC_P); r_enc.eval()
        print("UIT Models Loaded.")
    except Exception as e:
        print(f"Error loading UIT: {e}")
        return

    has_gru = False
    if G_DEC_P and os.path.exists(G_DEC_P) and os.path.exists(G_ENC_P):
        try:
            g_dec.load_state_dict(torch.load(G_DEC_P, map_location=DEVICE)); g_dec.eval()
            g_enc.load_state_dict(torch.load(G_ENC_P, map_location=DEVICE)); g_enc.eval()
            has_gru = True
            print("GRU Models Loaded.")
        except Exception as e:
            print(f"Error loading GRU: {e}. Comparing vs Untrained GRU.")
    else:
        print("GRU Crystals not found. Comparing vs Untrained GRU.")

    # 4. MEASURE ISOLATED COMPONENTS
    print("\n[PHASE A] Verifying Components...")
    x_d, y_d, x_e, y_e = get_all_chars_data()
    
    r_dec_acc = verify_component(r_dec, x_d, y_d, True, True)
    r_enc_acc = verify_component(r_enc, x_e, y_e, False, True)
    print(f"UIT | Dec: {r_dec_acc:.1%} | Enc: {r_enc_acc:.1%}")
    
    if has_gru:
        g_dec_acc = verify_component(g_dec, x_d, y_d, True, False)
        g_enc_acc = verify_component(g_enc, x_e, y_e, False, False)
        print(f"GRU | Dec: {g_dec_acc:.1%} | Enc: {g_enc_acc:.1%}")
    else:
        g_dec_acc = 0.0; g_enc_acc = 0.0

    # 5. MEASURE RELAY (THE DUEL)
    print("\n[PHASE B] The Sandwich Duel...")
    
    def test_relay(dec, enc, is_uit):
        success = 0
        with torch.no_grad():
            for i in range(256):
                # 1. Decode Bitstream -> H
                bits = [(i >> b) & 1 for b in range(7, -1, -1)]
                
                if is_uit:
                    h = torch.zeros(1, HIDDEN_SIZE).to(DEVICE)
                    for b in bits:
                        _, h, _, _, _ = dec.layers[0](torch.tensor([[b]]).float().to(DEVICE), h)
                else:
                    x_seq = torch.tensor(bits).float().unsqueeze(0).unsqueeze(-1).to(DEVICE)
                    _, h = dec(x_seq)
                
                # 2. Encode Identity (OneHot) -> Bitstream (Reconstruction)
                # Note: The relay 'message' is the Identity 'i'.
                # The Receiver (Encoder) tries to reconstruct the 'features' of 'i'.
                onehot = torch.zeros(1, 256).to(DEVICE); onehot[0, i] = 1.0
                
                recon_bits = []
                if is_uit:
                     # One Shot
                     l, _ = enc(onehot.unsqueeze(1))
                     bits_raw = l.squeeze().tolist()
                     if type(bits_raw) != list: bits_raw = [bits_raw]
                     recon_bits = [1 if b > 0 else 0 for b in bits_raw]
                else:
                     # GRU Recurrent Gen
                     h_g = h.unsqueeze(0)
                     curr_b = torch.zeros(1, 1).to(DEVICE)
                     for t in range(8):
                         c_in = torch.cat([curr_b, onehot], dim=-1).unsqueeze(1)
                         o, h_g = enc.gru(c_in, h_g)
                         l = enc.readout(o.squeeze(1))
                         recon_bits.append(1 if l > 0 else 0)
                         curr_b = torch.sigmoid(l)
                
                # Check
                # Encoder target was LSB.
                # 'bits' (Input) was MSB.
                target_lsb = bits[::-1] 
                if recon_bits == target_lsb: success += 1
        return success / 256.0

    r_relay = test_relay(r_dec, r_enc, True)
    g_relay = test_relay(g_dec, g_enc, False) if has_gru else 0.0
    
    print(f"Results | UIT Relay: {r_relay:.1%} | GRU Relay: {g_relay:.1%}")

    # 6. PLOT
    results = {
        "r_dec_acc": r_dec_acc, "r_enc_acc": r_enc_acc,
        "g_dec_acc": g_dec_acc, "g_enc_acc": g_enc_acc,
        "r_relay": r_relay, "g_relay": g_relay
    }
    plot_sandwich_story(results, args.output_dir, args.uid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--uid", type=str, default="test")
    parser.add_argument("--crystal_path", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    
    args = parser.parse_args()
    run_duel(args)
