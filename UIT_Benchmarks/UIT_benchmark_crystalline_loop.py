import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import numpy as np
from UIT_ROUND import UITModel, UITEncoderModel

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DECODER_PATH = "ascii_decoder_ultra.pt"
ENCODER_PATH = "ascii_encoder_ultra.pt"
HIDDEN_SIZE = 512
SEQ_LEN = 8

def generate_binary_streams():
    """Returns all 256 ASCII characters as 8-bit tensors."""
    chars = torch.arange(256).long()
    bits = []
    for i in range(256):
        char_bits = []
        for b in range(7, -1, -1):
            char_bits.append((i >> b) & 1)
        bits.append(char_bits)
    return chars, torch.tensor(bits).float().to(DEVICE)

def run_loop_benchmark():
    print("--- [CRYSTALLINE LOOP BENCHMARK] ---")
    
    # 1. Load the Crystalline Identities
    print(f"Loading Decoder: {DECODER_PATH}")
    decoder = UITModel(input_size=1, hidden_size=HIDDEN_SIZE, output_size=256, num_layers=1, use_binary_alignment=True)
    try:
        decoder.load_crystal(DECODER_PATH)
    except Exception as e:
        print(f"Warning: Could not load decoder crystal. Proceeding with random init.")
    decoder.to(DEVICE)
    decoder.eval()
    
    print(f"Loading Encoder: {ENCODER_PATH}")
    encoder = UITEncoderModel(input_size=256, hidden_size=HIDDEN_SIZE, output_size=1, num_layers=1, use_binary_alignment=True)
    try:
        encoder.load_crystal(ENCODER_PATH)
    except Exception as e:
        print(f"Warning: Could not load encoder crystal. Proceeding with random init.")
    encoder.to(DEVICE)
    encoder.eval()
    
    # 2. Generate Data
    char_ids, target_bits = generate_binary_streams()
    
    print("\nExecuting Phase Relay Loop...")
    success_count = 0
    
    with torch.no_grad():
        for i in range(256):
            # A. THE EAR (Decoding Bitstream -> Phase)
            bitstream = target_bits[i].unsqueeze(0).unsqueeze(-1)
            h = torch.zeros(1, HIDDEN_SIZE).to(DEVICE)
            
            if i < 5:
                print(f"\n[CHAR {i} ({chr(i) if i < 128 else '?'})] EAR JOURNEY:")
            
            for t in range(SEQ_LEN):
                in_bit = bitstream[:, t, :]
                _, h, _, _, _ = decoder.layers[0](in_bit, h)
                if i < 5:
                    print(f"  Step {t}: In={in_bit.item()} -> Phi={h[0, 0].item():.4f}")
            
            phase_identity = h 
            
            # B. THE VOICE (Phase -> Bitstream)
            # Seed the encoder with the terminal phase from the decoder
            h_v = phase_identity
            
            # Context for encoder (the same ID)
            char_onehot = torch.zeros(1, 256).to(DEVICE)
            char_onehot[0, char_ids[i]] = 1.0
            
            # Run the Encoder (Topological Mirror)
            # We bypass the internal 'input_projection' and set h_v directly
            # encoder.forward(x, seq_len) usually calls input_projection.
            # We'll use a modified loop that calls the model's layer and readout.
            
            recon_bits = []
            for t in range(SEQ_LEN):
                # Context bit 0 (Autonomous Unwinding)
                cell_in = torch.cat([torch.zeros(1, 1).to(DEVICE), char_onehot], dim=-1)
                
                # This calls the model's layer (Renormalization included!)
                current_feat, h_v, conf, h_cos, h_sin = encoder.layers[0](cell_in, h_v)
                
                # Apply readout
                combined = torch.cat([current_feat, h_cos, h_sin], dim=-1)
                logit = encoder.readout(combined)
                
                # Bit extraction
                bit = (torch.sigmoid(logit) > 0.5).float()
                recon_bits.append(bit.item())
            
            recon_bits_binary = torch.tensor(recon_bits).float()
            target_bits_rev = target_bits[i].flip(dims=[0]).cpu()
            
            if torch.equal(recon_bits_binary, target_bits_rev):
                success_count += 1
            else:
                if i < 10: 
                    print(f"\nFAILED Char {i} ({chr(i) if i < 128 else '?'})")
                    print(f"  Target (LSB-first): {target_bits_rev.tolist()}")
                    print(f"  Recov  (LSB-first): {recon_bits_binary.tolist()}")
    
    print(f"\n[RESULTS] Relay Success: {success_count}/256 ({success_count/256:.2%})")
    if success_count == 256:
        print("[CONCLUSION] The Crystalline Loop is 100% Air-Tight.")
    else:
        print("[CONCLUSION] The Relay has Phasic Leakage.")

if __name__ == "__main__":
    run_loop_benchmark()
