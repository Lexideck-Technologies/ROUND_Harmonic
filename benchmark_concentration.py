# version 0.6.1 - Harmonic Monism (Concentration Terminal)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import uuid
import time
from ROUND import PhaseAccumulator, HarmonicROUNDLoss
from config import get_lock_strength

# --- Configuration ---
HIDDEN_SIZE = 32
LR = 0.001953125 # 2^-9
PEAK_LOCKING_STRENGTH = 0.0625
EPOCHS = 1000 # Let it train well
TEXT = "CONCENTRATING"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConcentrationNeuron(nn.Module):
    def __init__(self, hidden_size=32, input_dim=8, output_dim=256):
        super().__init__()
        self.h = hidden_size
        self.e = nn.Linear(input_dim, hidden_size) 
        self.c = PhaseAccumulator(hidden_size, spinor=True)
        # Readout: [Cos, Sin, CosS, SinS, Ph] -> Total H * 5
        self.r = nn.Linear(hidden_size * 5, output_dim)
        # Parity Readout (Dual Task): XOR of all bits seen
        self.p_readout = nn.Linear(hidden_size * 5, 1)

    def forward(self, x, ph_in=None):
        # x: [Batch, Seq, 8]
        B, S, D = x.shape
        ph = ph_in if ph_in is not None else torch.zeros(B, self.h, device=x.device)
        
        logits_seq = []
        parity_seq = []
        hist_seq = []

        for t in range(S):
            xt = x[:, t, :]
            pt = self.e(xt)
            xpt = torch.stack([torch.cos(pt), torch.sin(pt)], 2)
            ph = self.c(ph, xpt)
            hist_seq.append(ph)
            
            # Spinor features
            ph_s = 0.5 * ph
            readout_features = torch.cat([
                torch.cos(ph), torch.sin(ph), 
                torch.cos(ph_s), torch.sin(ph_s), 
                ph
            ], 1)
            
            logits = self.r(readout_features)
            parity = self.p_readout(readout_features)
            
            logits_seq.append(logits)
            parity_seq.append(parity)

        return torch.stack(logits_seq, dim=1), torch.stack(parity_seq, dim=1), hist_seq

def str_to_bits(s):
    bits = [[int(c) for c in format(ord(ch), '08b')] for ch in s]
    return torch.tensor(bits, dtype=torch.float32)

def parity_of_bits(bits_tensor):
    # bits_tensor: [Seq, 8]
    # We want cumulative parity
    flat = bits_tensor.view(-1)
    cum_sum = torch.cumsum(flat, dim=0)
    # This is slightly complex for cumulative char-level parity
    # Let's just do parity per char for simplicity or cumulative bit parity?
    # User said "parity and hello world", usually parity test is recursive XOR.
    # Let's do cumulative XOR of all bits.
    parities = []
    current = 0
    for b in bits_tensor:
        for bit in b:
            current = (current + int(bit)) % 2
        parities.append(float(current))
    return torch.tensor(parities).unsqueeze(1)

def train_concentration():
    print(f"\n--- [CONCENTRATION TERMINAL v0.6.1] ---")
    print(f"Target: '{TEXT}' | Hidden: {HIDDEN_SIZE} | LR: {LR}")
    
    model = ConcentrationNeuron(HIDDEN_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion_char = nn.CrossEntropyLoss()
    criterion_parity = nn.BCEWithLogitsLoss()
    criterion_lock = HarmonicROUNDLoss(locking_strength=PEAK_LOCKING_STRENGTH, harmonics=[1], mode='multiclass')

    # Data
    input_bits = str_to_bits(TEXT).unsqueeze(0).to(DEVICE)
    # Target char is shifted
    targets_char = [ord(c) for c in TEXT[1:] + TEXT[0]]
    targets_char = torch.tensor(targets_char, dtype=torch.long).unsqueeze(0).to(DEVICE)
    # Target parity
    targets_parity = parity_of_bits(str_to_bits(TEXT)).unsqueeze(0).to(DEVICE)

    print("Training phase... (May the U-Neuron guide you)")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        criterion_lock.locking_strength = get_lock_strength(epoch, EPOCHS, PEAK_LOCKING_STRENGTH)
        optimizer.zero_grad()
        
        logits, parities, hist = model(input_bits)
        
        loss_char = criterion_char(logits.view(-1, 256), targets_char.view(-1))
        loss_parity = criterion_parity(parities.view(-1, 1), targets_parity.view(-1, 1))
        
        # Harmonic Locking on chars
        loss_lock, _, _ = criterion_lock(logits.view(-1, 256), targets_char.view(-1), hist)
        
        # Combined Loss
        total_loss = loss_char + loss_parity + (loss_lock - loss_char)
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            acc = (torch.argmax(logits, 2) == targets_char).float().mean().item()
            print(f"E {epoch:4d} | Acc: {acc:.2f} | Loss: {total_loss.item():.4f}")
            if acc == 1.0 and epoch > 300: break

    print(f"Locked in {time.time() - start_time:.2f}s.")
    
    # --- Interactive Mode ---
    print("\n--- ENTERING TERMINAL MODE ---")
    print("Commands: 'run <n>', 'disrupt', 'prompt <char>', 'status', 'help', 'exit'")
    
    current_ph = torch.zeros(1, HIDDEN_SIZE, device=DEVICE)
    last_char = TEXT[-1]
    
    while True:
        try:
            cmd = input(f"\n[{last_char}]> ").strip().lower()
            if not cmd: continue
            
            if cmd == 'exit' or cmd == 'quit':
                print("Exiting Concentration...")
                break
            
            elif cmd.startswith('run'):
                parts = cmd.split()
                n = int(parts[1]) if len(parts) > 1 else 13
                model.eval()
                with torch.no_grad():
                    output_str = ""
                    for _ in range(n):
                        bits = str_to_bits(last_char).unsqueeze(0).to(DEVICE)
                        logits, _, hist = model(bits, ph_in=current_ph)
                        current_ph = hist[-1]
                        idx = torch.argmax(logits, 2).item()
                        last_char = chr(idx)
                        output_str += last_char
                print(f"Output: {output_str}")
                
            elif cmd == 'disrupt':
                noise = torch.randn_like(current_ph) * 2.0
                current_ph += noise
                print("Phase disrupted. State jittered.")
                
            elif cmd.startswith('prompt'):
                parts = cmd.split()
                if len(parts) > 1:
                    last_char = parts[1][0]
                    bits = str_to_bits(last_char).unsqueeze(0).to(DEVICE)
                    _, _, hist = model(bits, ph_in=current_ph)
                    current_ph = hist[-1]
                    print(f"Prompted with '{last_char}'. State updated.")
                else:
                    print("Usage: prompt <char>")
                    
            elif cmd == 'status':
                # Show phase mean
                print(f"Phase State Mean: {current_ph.mean().item():.4f}")
                print(f"Last Char: {last_char}")
                
            elif cmd == 'help':
                print("run <n>      : Generate n characters")
                print("disrupt      : Inject noise into the phase state")
                print("prompt <c>   : Force feed a character and update memory")
                print("status       : Inspect the neuron's internal state")
                print("exit         : Terminate the session")
            
            else:
                print(f"Unknown command: {cmd}")
                
        except KeyboardInterrupt:
            print("\nInterrupt. Type 'exit' to quit.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    train_concentration()
