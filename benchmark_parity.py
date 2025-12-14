
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os


from ROUND import ROUNDModel, ROUNDLoss, HarmonicROUNDLoss

# ==============================================================================
# Configuration
# ==============================================================================
CONFIG = {
    'task': 'parity_16',
    'input_dim': 16,
    'hidden_size': 32,
    'steps': 20,          
    'epochs': 1000,        # Standardize to 1k
    'batch_size': 64,
    'dataset_size': 2000, 
    'runs': 5,            
    'lr': 0.005,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ==============================================================================
# Dataset: N-bit Parity
# ==============================================================================
def generate_parity_data(n_samples, n_bits):
    X = torch.randint(0, 2, (n_samples, n_bits)).float()
    Y = X.sum(dim=1) % 2
    Y = Y.unsqueeze(1).float()
    return X, Y

# ==============================================================================
# GRU Baseline
# ==============================================================================
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x_seq = x.unsqueeze(-1) 
        output, hn = self.gru(x_seq)
        final_h = hn[-1]
        return self.fc(final_h)

# ==============================================================================
# Training Engine
# ==============================================================================
def train_round(run_id, X, Y, device):
    model = ROUNDModel(hidden_size=CONFIG['hidden_size'], input_dim=CONFIG['input_dim']).to(device)
    criterion = HarmonicROUNDLoss(locking_strength=0.05, harmonics=[1, 2], mode='binary', terminal_only=True) 
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    accuracy_history = []
    
    for epoch in range(CONFIG['epochs']):
        optimizer.zero_grad()
        output, history = model(X, steps=CONFIG['steps'])
        loss, mse, locking = criterion(output, Y, history)
        loss.backward()
        optimizer.step()
        
        # Sigmoid for accuracy check
        preds = (torch.sigmoid(output) > 0.5).float()
        acc = (preds == Y).float().mean().item()
        accuracy_history.append(acc)
        
        if epoch % 100 == 0:
            print(f"[ROUND Run {run_id}] Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc:.2f}")
            
    return accuracy_history, preds, Y

def train_gru(run_id, X, Y, device):
    model = GRUModel(input_dim=1, hidden_size=CONFIG['hidden_size']).to(device)
    # GRU output is logits, use BCEWithLogitsLoss
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
    print(f"Running on {device} - Parity Benchmark")
    
    # 1. Generate Data
    print("Generating 16-bit Parity Data...")
    X, Y = generate_parity_data(CONFIG['dataset_size'], CONFIG['input_dim'])
    X, Y = X.to(device), Y.to(device)
    
    # 2. Run Benchmarks
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

    # 3. Plotting
    print("\nGenerating Graphs...")
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate Mean and Std
    round_arr = np.array(round_results)
    gru_arr = np.array(gru_results)
    
    round_mean = np.mean(round_arr, axis=0)
    round_std = np.std(round_arr, axis=0)
    gru_mean = np.mean(gru_arr, axis=0)
    gru_std = np.std(gru_arr, axis=0)
    
    epochs = np.arange(CONFIG['epochs'])
    
    # Plot Confidence Intervals
    ax.fill_between(epochs, round_mean - round_std, round_mean + round_std, color='#FF4B4B', alpha=0.1)
    ax.fill_between(epochs, gru_mean - gru_std, gru_mean + gru_std, color='#4B4BFF', alpha=0.1)
    
    # Plot Individual Runs (Thin)
    for run in round_results:
        ax.plot(run, color='#FF4B4B', alpha=0.15, linewidth=1)
    for run in gru_results:
        ax.plot(run, color='#4B4BFF', alpha=0.15, linewidth=1)
        
    # Plot Averages
    ax.plot(round_mean, color='#FF4B4B', linewidth=2.5, label=f'ROUND (n={CONFIG["hidden_size"]})')
    ax.plot(gru_mean, color='#4B4BFF', linewidth=2.5, label=f'GRU (n={CONFIG["hidden_size"]})')
    
    ax.set_title(f'Discrete Logic: ROUND vs GRU ({CONFIG["input_dim"]} bit Parity)\nResults over {CONFIG["runs"]} Runs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.1)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('benchmark_parity.png')
    
    # 4. Correlation Matrix
    data_stack = np.stack(all_round_final_preds)
    target_flat = final_targets.flatten()
    data_stack = np.vstack([data_stack, target_flat])
    labels = [f'Run {i+1}' for i in range(CONFIG['runs'])] + ['Target']
    corr = np.corrcoef(data_stack)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, interpolation='nearest', cmap='coolwarm', vmin=0, vmax=1)
    plt.title('ROUND Parity: Inter-Run Consistency')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = plt.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color="black" if 0.3 < corr[i, j] < 0.7 else "white")
    plt.tight_layout()
    plt.savefig('correlation_parity.png')
    
    print("Done. Saved benchmark_parity.png and correlation_parity.png")
