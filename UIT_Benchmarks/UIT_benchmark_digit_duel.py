"""
U-Matrix Text/Number Benchmark
Task: Decode ASCII digit characters ('0'-'9') from their bit representations.
This is a discrete symbolic task - perfect for ROUND's phase encoding.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
import argparse
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from UIT_ROUND import UITModel

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default=".")
parser.add_argument("--uid", type=str, default="digit_duel")
parser.add_argument("--lr", type=float, default=None)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = args.output_dir
UID = args.uid
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- CONFIGURATION ---
BATCH_SIZE = 64
EPOCHS = 1000
LEARNING_RATE = args.lr if args.lr is not None else 0.0078125  # 2^-7
HIDDEN_SIZE = 10  # Match the 10 digit classes (0-9)
SEQ_LEN = 8  # 8 bits per ASCII character

# --- DATA GENERATION ---
def generate_digit_data(batch_size):
    """Generate ASCII bit sequences for digits '0'-'9' and their numeric labels."""
    # Digits '0'-'9' have ASCII codes 48-57
    digit_chars = torch.randint(0, 10, (batch_size,))  # 0-9
    ascii_codes = digit_chars + 48  # 48-57 ('0'-'9')
    
    # Convert to bit sequences (MSB first, 8 bits)
    bits = []
    for code in ascii_codes:
        bit_seq = [(code.item() >> i) & 1 for i in range(7, -1, -1)]
        bits.append(bit_seq)
    
    x = torch.tensor(bits).float().unsqueeze(-1).to(DEVICE)  # [B, 8, 1]
    y = digit_chars.to(DEVICE)  # [B] (labels 0-9)
    
    return x, y

# --- MODELS ---
class DigitROUND(nn.Module):
    def __init__(self):
        super().__init__()
        # 1-layer, hidden size matches class count, direct cell access
        self.uit = UITModel(
            input_size=1, 
            hidden_size=HIDDEN_SIZE, 
            output_size=10, 
            num_layers=1, 
            persistence=0.5, 
            quantization_strength=0.0
        )
        self.classifier = nn.Linear(HIDDEN_SIZE * 3, 10)
        
    def forward(self, x):
        batch_size = x.size(0)
        h = torch.zeros(batch_size, HIDDEN_SIZE).to(x.device)
        
        # Process bit sequence through the cell
        for t in range(SEQ_LEN):
            out, h, _, h_cos, h_sin = self.uit.layers[0](x[:, t, :], h)
        
        # Final readout uses accumulated phase
        combined = torch.cat([out, h_cos, h_sin], dim=-1)
        return self.classifier(combined)

class DigitGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(1, HIDDEN_SIZE, batch_first=True)
        self.classifier = nn.Linear(HIDDEN_SIZE, 10)
        
    def forward(self, x):
        _, h = self.gru(x)  # h: [1, B, H]
        return self.classifier(h.squeeze(0))

def run_benchmark():
    print(f"--- [U-MATRIX DIGIT DUEL | UID: {UID}] ---")
    print(f"Task: Decode ASCII digit chars ('0'-'9') from 8-bit sequences")
    
    round_model = DigitROUND().to(DEVICE)
    gru_model = DigitGRU().to(DEVICE)
    
    r_opt = optim.Adam(round_model.parameters(), lr=LEARNING_RATE)
    g_opt = optim.Adam(gru_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    history = {"round": [], "gru": []}
    
    for epoch in range(EPOCHS):
        # Training
        round_model.train(); gru_model.train()
        x, y = generate_digit_data(BATCH_SIZE)
        
        r_opt.zero_grad()
        r_loss = criterion(round_model(x), y)
        r_loss.backward(); r_opt.step()
        
        g_opt.zero_grad()
        g_loss = criterion(gru_model(x), y)
        g_loss.backward(); g_opt.step()
        
        if epoch % 50 == 0:
            round_model.eval(); gru_model.eval()
            with torch.no_grad():
                vx, vy = generate_digit_data(100)
                r_acc = (torch.argmax(round_model(vx), dim=1) == vy).float().mean().item()
                g_acc = (torch.argmax(gru_model(vx), dim=1) == vy).float().mean().item()
                history["round"].append(r_acc)
                history["gru"].append(g_acc)
                print(f"Epoch {epoch:4d} | ROUND: {r_acc:7.2%} | GRU: {g_acc:7.2%}")
            if r_acc > 0.99 and g_acc > 0.99:
                print("Both models converged!")
                break
    
    # --- VISUALIZATION ---
    try:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history["round"], color='#00FF00', linewidth=3, label='ROUND (Green)')
        ax.plot(history["gru"], color='#4B4BFF', linewidth=3, label='GRU (Blue)')
        ax.set_title(f"Digit Recognition Duel (ASCII Bits → Digit) | UID: {UID}", color='white', fontsize=14)
        ax.set_xlabel("Epochs (x50)", color='white')
        ax.set_ylabel("Accuracy", color='white')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
        
        plot_path = os.path.join(OUTPUT_DIR, f"digit_duel_{UID}.png")
        plt.savefig(plot_path, facecolor='black', edgecolor='none')
        print(f"Plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"Visualization Failed: {e}")

if __name__ == "__main__":
    run_benchmark()
