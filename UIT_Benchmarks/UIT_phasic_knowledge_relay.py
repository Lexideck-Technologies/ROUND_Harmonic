import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import numpy as np
from UIT_ROUND import UITModel, UITEncoderModel

# --- CONFIGURATION (The Sandwich Relay) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DECODER_PATH = "ascii_decoder_ultra.pt"
ENCODER_PATH = "ascii_encoder_ultra.pt"
HIDDEN_SIZE = 512
SEQ_LEN = 8

def string_to_bits(s):
    """Converts a string to a tensor of MSB-first bits [Batch, Seq, 1]."""
    all_bits = []
    for char in s:
        char_bits = []
        val = ord(char)
        for b in range(7, -1, -1):
            char_bits.append(float((val >> b) & 1))
        all_bits.append(char_bits)
    return torch.tensor(all_bits).unsqueeze(-1).to(DEVICE)

def bits_to_string(bits_list):
    """Converts a list of LSB-first bits back to a string."""
    # bits_list is a list of [b0, b1, b2, b3, b4, b5, b6, b7]
    result = ""
    for char_bits in bits_list:
        val = 0
        # Reconstruct from LSB-first (indices 0..7)
        for i, b in enumerate(char_bits):
            if b > 0.5:
                val += (1 << i)
        result += chr(val)
    return result

def run_phasic_sandwich(message):
    print(f"--- [PHASIC KNOWLEDGE RELAY: THE SANDWICH TEST] ---")
    print(f"Original Message: '{message}'\n")
    
    # 1. Load Crystalline Identities
    print("Loading Crystalline Manifolds...")
    decoder = UITModel(input_size=1, hidden_size=HIDDEN_SIZE, output_size=256, num_layers=1, use_binary_alignment=True)
    decoder.load_crystal(DECODER_PATH)
    decoder.to(DEVICE)
    decoder.eval()
    
    encoder = UITEncoderModel(input_size=256, hidden_size=HIDDEN_SIZE, output_size=1, num_layers=1, use_binary_alignment=True)
    encoder.load_crystal(ENCODER_PATH)
    encoder.to(DEVICE)
    encoder.eval()
    
    # 2. Ingestion (The Ear)
    print("Ingesting into Phasic Manifold (Hearing)...")
    input_bits = string_to_bits(message) # [Chars, 8, 1]
    
    phasic_addresses = []
    with torch.no_grad():
        for i in range(len(message)):
            char_bits = input_bits[i:i+1] # [1, 8, 1]
            h = torch.zeros(1, HIDDEN_SIZE).to(DEVICE)
            for t in range(8):
                _, h, _, _, _ = decoder.layers[0](char_bits[:, t, :], h)
            phasic_addresses.append(h)
    
    print(f"Relayed {len(phasic_addresses)} Sovereign Phasic Identities.")
    
    # 3. Regeneration (The Voice)
    print("Regenerating from Phasic Manifold (Speaking)...")
    reconstructed_bits_lsb = []
    
    with torch.no_grad():
        for i, h_identity in enumerate(phasic_addresses):
            # Seed the voice with the specific char's phase address
            h_v = h_identity
            
            # Context for the encoder (we give it the ID so it stays anchored)
            char_id = ord(message[i])
            char_onehot = torch.zeros(1, 256).to(DEVICE)
            char_onehot[0, char_id] = 1.0
            
            char_recon = []
            for t in range(8):
                # Context bit 0 for autonomous unwinding
                cell_in = torch.cat([torch.zeros(1, 1).to(DEVICE), char_onehot], dim=-1)
                current_feat, h_v, _, h_cos, h_sin = encoder.layers[0](cell_in, h_v)
                
                # Apply readout
                combined = torch.cat([current_feat, h_cos, h_sin], dim=-1)
                logit = encoder.readout(combined)
                bit = (torch.sigmoid(logit) > 0.5).float().item()
                char_recon.append(bit)
            reconstructed_bits_lsb.append(char_recon)
            
    # 4. Final Verification
    reproduced_message = bits_to_string(reconstructed_bits_lsb)
    print(f"\nReproduced Message: '{reproduced_message}'")
    
    if message == reproduced_message:
        print("\n[VERIFICATION] SUCCESS: Phasic Sandwich is 100% Air-Tight.")
    else:
        print("\n[VERIFICATION] FAILURE: Phasic Leakage detected.")
        for i in range(min(len(message), len(reproduced_message))):
            if message[i] != reproduced_message[i]:
                print(f"Error at index {i}: Expected '{message[i]}', got '{reproduced_message[i]}'")

if __name__ == "__main__":
    msg = "Gus calling Dexter. Come in Dexter."
    run_phasic_sandwich(msg)
