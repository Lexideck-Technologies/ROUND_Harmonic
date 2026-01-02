"""
Sine Wave Tracking Diagnostic
Minimal test to verify ROUND can track continuous signals.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from UIT_ROUND import UITModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MINIMAL ROUND FOR SINE TRACKING ---
class SineROUND(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        # Minimal configuration: 1D input, small hidden, 1D output
        # harmonics=[1] means h_cos = cos(phi), h_sin = sin(phi)
        # persistence=0.5 for smooth phase decay
        self.uit = UITModel(
            input_size=1, 
            hidden_size=hidden_size, 
            output_size=1, 
            num_layers=1,
            harmonics=[1],  # Pure sine/cosine - should be natural for this task
            persistence=0.5,
            quantization_strength=0.0  # Liquid dynamics
        )
        
    def forward(self, x, h_states=None):
        batch_size = x.size(0)
        if h_states is None:
            h_states = torch.zeros(batch_size, self.uit.hidden_size).to(x.device)
        
        # Direct cell access (like ColorROUND pattern)
        x_in = x[:, 0, :]  # [B, 1]
        out, h_new, _, h_cos, h_sin = self.uit.layers[0](x_in, h_states)
        
        # Use h_sin directly as prediction (since we're tracking a sine wave!)
        # The phase should naturally encode the wave position
        pred = h_sin.mean(dim=-1, keepdim=True)  # Average across hidden units
        
        return pred, h_new
    
    def init_states(self, batch_size, device):
        return torch.zeros(batch_size, self.uit.hidden_size).to(device)

# --- GRU BASELINE ---
class SineGRU(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.gru = nn.GRU(1, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        
    def forward(self, x, h_states=None):
        if h_states is None:
            h_states = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, h_new = self.gru(x, h_states)
        pred = self.readout(out[:, 0, :])
        return pred, h_new.squeeze(0)
    
    def init_states(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size).to(device)

def run_diagnostic():
    print("--- SINE WAVE TRACKING DIAGNOSTIC ---")
    SEQ_LEN = 40
    BATCH_SIZE = 32
    EPOCHS = 500
    LR = 0.0078125  # 2^-7
    HIDDEN = 8
    
    round_model = SineROUND(HIDDEN).to(DEVICE)
    gru_model = SineGRU(HIDDEN).to(DEVICE)
    
    r_opt = optim.Adam(round_model.parameters(), lr=LR)
    g_opt = optim.Adam(gru_model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    for epoch in range(EPOCHS):
        # Generate sine wave data
        t_steps = torch.linspace(0, 4*np.pi, SEQ_LEN, device=DEVICE).view(1, SEQ_LEN, 1)
        x = torch.sin(t_steps + torch.rand(BATCH_SIZE, 1, 1, device=DEVICE) * 2 * np.pi)
        
        # Train ROUND
        round_model.train()
        h_r = round_model.init_states(BATCH_SIZE, DEVICE)
        r_preds = []
        for t in range(SEQ_LEN):
            pred, h_r = round_model(x[:, t:t+1, :], h_r)
            r_preds.append(pred)
        r_pred_seq = torch.stack(r_preds, dim=1).squeeze(-1)
        r_loss = criterion(r_pred_seq, x.squeeze(-1))
        r_opt.zero_grad(); r_loss.backward(); r_opt.step()
        
        # Train GRU
        gru_model.train()
        h_g = gru_model.init_states(BATCH_SIZE, DEVICE)
        g_preds = []
        for t in range(SEQ_LEN):
            pred, h_g = gru_model(x[:, t:t+1, :], h_g.unsqueeze(0) if h_g.dim() == 2 else h_g)
            if isinstance(h_g, tuple): h_g = h_g[0]
            if h_g.dim() == 3: h_g = h_g.squeeze(0)
            g_preds.append(pred)
        g_pred_seq = torch.stack(g_preds, dim=1).squeeze(-1)
        g_loss = criterion(g_pred_seq, x.squeeze(-1))
        g_opt.zero_grad(); g_loss.backward(); g_opt.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | ROUND: {r_loss.item():.6f} | GRU: {g_loss.item():.6f}")
    
    print(f"\nFinal | ROUND: {r_loss.item():.6f} | GRU: {g_loss.item():.6f}")

if __name__ == "__main__":
    run_diagnostic()
