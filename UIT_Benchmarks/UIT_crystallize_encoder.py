import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from UIT_ROUND import UITEncoderModel, landauer_loss

# --- CONFIGURATION (v18: The Crystalline Hammer) ---
SEQ_LEN = 8 
BATCH_SIZE = 32
HIDDEN_SIZE = 512 
EPOCHS = 2000 
LR = 0.0078125 # 2^-7 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAP_PATH = "ascii_topological_map.pt"

def generate_encoder_data(batch_size, sovereign_map):
    char_ints = torch.randint(0, 256, (batch_size,)).long()
    char_onehot = torch.zeros(batch_size, 256)
    char_onehot.scatter_(1, char_ints.unsqueeze(-1), 1.0)
    
    # Target bits: LSB-first [b0, b1, b2, b3, b4, b5, b6, b7]
    # This matches the natural Bernoulli Unwinding after MSB-first winding.
    bits_msb = []
    for i in range(7, -1, -1):
        bits_msb.append((char_ints >> i) & 1)
    target_bits = torch.stack(bits_msb[::-1], dim=1).float().to(DEVICE) 
    
    return char_onehot.to(DEVICE), target_bits

def resonance_loss_fn(phi_v, grid_size=256):
    return torch.mean(torch.sin((grid_size/2.0) * phi_v)**2)

def crystallize():
    print(f"--- [ENCODER v18: THE CRYSTALLINE HAMMER] ---")
    sovereign_map = torch.load(MAP_PATH, map_location=torch.device('cpu'), weights_only=True)
    
    model = UITEncoderModel(input_size=256, hidden_size=HIDDEN_SIZE, output_size=1, num_layers=1, use_binary_alignment=True)
    
    # SEED THE IDENTITY (Renormalization Strike)
    model.renormalize_identity(MAP_PATH)
    
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(EPOCHS):
        model.train()
        char_onehot, y_bits = generate_encoder_data(BATCH_SIZE, sovereign_map)
        optimizer.zero_grad()
        
        batch_size = char_onehot.size(0)
        h = model.input_projection(char_onehot)
        
        output_logits = []
        phi_hist = []
        conf_hist = []
        
        for t in range(SEQ_LEN):
            cell_in = torch.cat([torch.zeros(batch_size, 1).to(DEVICE), char_onehot], dim=-1)
            bit_out_vec, h, conf, h_cos, h_sin = model.layers[0](cell_in, h)
            
            combined = torch.cat([bit_out_vec, h_cos, h_sin], dim=-1)
            logits = model.readout(combined)
            
            phi_hist.append(h)
            conf_hist.append(conf)
            output_logits.append(logits)
            
        output_logits = torch.stack(output_logits, dim=1).squeeze(-1)
        avg_conf = torch.stack(conf_hist).mean()
        all_phis = torch.stack(phi_hist)
        
        l_res = resonance_loss_fn(all_phis, grid_size=256)
        loss_bce = criterion(output_logits, y_bits)
        
        # INCREASED RESONANCE STRENGTH (0.5) to deepen the grid basins
        loss = (loss_bce + 0.5 * l_res + landauer_loss(model, beta=0.0)) * (1.0 - avg_conf.detach())
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                vx, vy = generate_encoder_data(256, sovereign_map)
                v_out_logits, v_conf = model(vx, seq_len=SEQ_LEN)
                preds = (torch.sigmoid(v_out_logits) > 0.5).float()
                acc = (preds == vy).all(dim=1).float().mean().item()
                print(f"Epoch {epoch+1} | Acc: {acc:.2%} | Conf: {v_conf.item():.4f} | Res: {l_res.item():.6f}")
                
                if acc >= 1.0:
                    print(f"--- [ENCODER v12 CRYSTAL SECURED] ---")
                    model.save_crystal("ascii_encoder_ultra.pt")
                    return

    print("Error: Encoder v12 failed to snap.")

if __name__ == "__main__":
    crystallize()
