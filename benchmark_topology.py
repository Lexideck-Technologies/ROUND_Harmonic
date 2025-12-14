
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import random


from ROUND import ROUNDModel, ROUNDLoss, ROUNDTopologyLoss, PhaseAccumulator, HarmonicROUNDLoss

# ==============================================================================
# Customized Model for Winding (Unwrapped Phase)
# ==============================================================================
class ROUNDTopologyModel(ROUNDModel):
    """
    Modified ROUND model for Topological Winding Tasks.
    Crucially, this model exposes the RAW PHASE (phi) to the readout,
    allowing it to distinguish between 0 and 2pi (Winding Number),
    which are identical in the projected Cos/Sin space.
    """
    def __init__(self, hidden_size=64, input_dim=16):
        super().__init__(hidden_size, input_dim)
        # Re-initialize readout to accept (Cos, Sin, Phi) -> Hidden*3
        self.readout = nn.Linear(hidden_size * 3, 1)

    def forward(self, x, steps=12):
        batch_size = x.size(0)
        device = x.device
        
        # 1. Encode
        phi_in = self.encoder(x)
        x_phasors_cos = torch.cos(phi_in)
        x_phasors_sin = torch.sin(phi_in)
        x_phasors = torch.stack([x_phasors_cos, x_phasors_sin], dim=2)
        
        # 2. Init State
        phi_h = torch.zeros(batch_size, self.hidden_size).to(device)
        phi_history = []
        
        # 3. Spin
        for _ in range(steps):
             phi_h = self.cell(phi_h, x_phasors)
             phi_history.append(phi_h)
             
        # 4. Readout (Topology Aware)
        # We concatenate Cos, Sin, AND Raw Phi.
        # This allows the "Manifold Depth" to be read.
        final_cos = torch.cos(phi_h)
        final_sin = torch.sin(phi_h)
        features = torch.cat([final_cos, final_sin, phi_h], dim=1)
        
        return self.readout(features), phi_history

# ==============================================================================
# Configuration
# ==============================================================================
CONFIG = {
    'task': 'winding',    
    'seq_len': 30,        
    'hidden_size': 32,    
    'steps': 30,          
    'epochs': 1000, 
    'batch_size': 64,
    'dataset_size': 3000, 
    'runs': 5,            
    'lr': 0.005,          
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ==============================================================================
# Dataset: Winding
# ==============================================================================
def generate_winding_data(n_samples, seq_len):
    X_list = []
    Y_list = []
    
    for _ in range(n_samples):
        label = 0 if random.random() < 0.5 else 1
        target_delta = 2 * np.pi if label == 1 else 0
        theta_start = random.uniform(0, 2*np.pi)
        t = np.linspace(0, 1, seq_len)
        theta_path = theta_start + t * target_delta
        perturbation = 0.5 * np.sin(t * np.pi * 2 * random.random() * 5)
        theta_path += perturbation
        x_cos = np.cos(theta_path)
        x_sin = np.sin(theta_path)
        x_seq = np.stack([x_cos, x_sin], axis=1) 
        x_flat = x_seq.flatten() 
        X_list.append(torch.tensor(x_flat).float())
        Y_list.append(float(label))

    X = torch.stack(X_list)
    Y = torch.tensor(Y_list).unsqueeze(1)
    return X, Y

# ==============================================================================
# GRU Baseline
# ==============================================================================
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim=1):
        super().__init__()
        self.gru = nn.GRU(2, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # Flattened input (Batch, 60) -> (Batch, 30, 2)
        x_seq = x.view(x.size(0), -1, 2)
        output, hn = self.gru(x_seq)
        final_h = hn[-1]
        return self.fc(final_h)

# ==============================================================================
# Training Engine
# ==============================================================================
def train_round(run_id, X, Y, device):
    model = ROUNDTopologyModel(hidden_size=CONFIG['hidden_size'], input_dim=CONFIG['seq_len']*2).to(device)
    criterion = HarmonicROUNDLoss(locking_strength=0.1, harmonics=[1, 2, 4, 8], mode='binary', terminal_only=True) 
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    accuracy_history = []
    for epoch in range(CONFIG['epochs']):
        optimizer.zero_grad()
        output, history = model(X, steps=CONFIG['steps'])
        loss, mse, locking = criterion(output, Y, history)
        loss.backward()
        optimizer.step()
        preds = (torch.sigmoid(output) > 0.5).float()
        acc = (preds == Y).float().mean().item()
        accuracy_history.append(acc)
        if epoch % 100 == 0:
            print(f"[ROUND Run {run_id}] Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc:.2f}")
    return accuracy_history, preds, Y

def train_gru(run_id, X, Y, device):
    model = GRUModel(input_dim=2, hidden_size=CONFIG['hidden_size']).to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    accuracy_history = []
    for epoch in range(CONFIG['epochs']):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
        preds = (torch.sigmoid(output) > 0.5).float()
        acc = (preds == Y).float().mean().item()
        accuracy_history.append(acc)
        if epoch % 100 == 0:
            print(f"[GRU Run {run_id}] Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc:.2f}")
    return accuracy_history, preds, Y

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    device = torch.device(CONFIG['device'])
    print(f"Running on {device} - Topology Benchmark")
    
    X, Y = generate_winding_data(CONFIG['dataset_size'], CONFIG['seq_len'])
    X, Y = X.to(device), Y.to(device)
    
    round_results = []
    gru_results = []
    all_round_final_preds = []
    final_targets = Y.cpu().numpy()
    
    print("\n--- Training ROUND ---")
    for i in range(CONFIG['runs']):
        acc, preds, _ = train_round(i+1, X, Y, device)
        round_results.append(acc)
        all_round_final_preds.append(preds.detach().cpu().numpy().flatten())

    print("\n--- Training GRU ---")
    for i in range(CONFIG['runs']):
        acc, preds, _ = train_gru(i+1, X, Y, device)
        gru_results.append(acc)

    print("\nGenerating Graphs...")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    round_arr = np.array(round_results)
    gru_arr = np.array(gru_results)
    round_mean = np.mean(round_arr, axis=0)
    round_std = np.std(round_arr, axis=0)
    gru_mean = np.mean(gru_arr, axis=0)
    gru_std = np.std(gru_arr, axis=0)
    epochs = np.arange(CONFIG['epochs'])
    
    ax.fill_between(epochs, round_mean - round_std, round_mean + round_std, color='#FF4B4B', alpha=0.1)
    ax.fill_between(epochs, gru_mean - gru_std, gru_mean + gru_std, color='#4B4BFF', alpha=0.1)
    
    for run in round_results: ax.plot(run, color='#FF4B4B', alpha=0.15, linewidth=1)
    for run in gru_results: ax.plot(run, color='#4B4BFF', alpha=0.15, linewidth=1)
        
    ax.plot(round_mean, color='#FF4B4B', linewidth=2.5, label=f'ROUND (n={CONFIG["hidden_size"]})')
    ax.plot(gru_mean, color='#4B4BFF', linewidth=2.5, label=f'GRU (n={CONFIG["hidden_size"]})')
    
    ax.set_title(f'Continuous Topology: ROUND vs GRU (Winding)\nResults over {CONFIG["runs"]} Runs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.1)
    ax.legend()
    plt.tight_layout()
    plt.savefig('benchmark_topology.png')
    
    data_stack = np.stack(all_round_final_preds) 
    target_flat = final_targets.flatten()
    data_stack = np.vstack([data_stack, target_flat]) 
    labels = [f'Run {i+1}' for i in range(CONFIG['runs'])] + ['Target']
    corr = np.corrcoef(data_stack)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, interpolation='nearest', cmap='coolwarm', vmin=0, vmax=1)
    plt.title('ROUND Topology: Inter-Run Consistency')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = plt.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color="black" if 0.3 < corr[i, j] < 0.7 else "white")
    plt.tight_layout()
    plt.savefig('correlation_topology.png')
    
    print("Done. Saved benchmark_topology.png and correlation_topology.png")
