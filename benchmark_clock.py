
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os


from ROUND import ROUNDClockModel, ROUNDClockLoss, HarmonicROUNDLoss

# ==============================================================================
# Configuration
# ==============================================================================
CONFIG = {
    'task': 'modulo_8',   
    'seq_len': 20,        
    'classes': 8,         # Modulo 8 (0-7)
    'hidden_size': 32,    
    'steps': 20,          
    'epochs': 1000,       # 1000 epochs
    'batch_size': 64,
    'dataset_size': 4000, 
    'runs': 5,            
    'lr': 0.005,          
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ==============================================================================
# GRU Baseline for Classification
# ==============================================================================
class GRUClockModel(nn.Module):
    def __init__(self, input_dim, hidden_size, output_classes=8):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_classes)

    def forward(self, x):
        x_seq = x.unsqueeze(-1)
        output, hn = self.gru(x_seq)
        final_h = hn[-1]
        return self.fc(final_h)

# ==============================================================================
# Dataset: Modulo-8 Sum
# ==============================================================================
def generate_clock_data(n_samples, seq_len, mod=8):
    X_list = []
    Y_list = []
    
    for _ in range(n_samples):
        # Integers 0..7
        seq = torch.randint(0, mod, (seq_len,)).float()
        total = seq.sum().long()
        label = total % mod
        seq_norm = seq / float(mod) 
        X_list.append(seq_norm)
        Y_list.append(label)

    X = torch.stack(X_list)
    Y = torch.tensor(Y_list).unsqueeze(1) 
    return X, Y

# ==============================================================================
# Training Engine
# ==============================================================================
def train_round(run_id, X, Y, device):
    model = ROUNDClockModel(hidden_size=CONFIG['hidden_size'], 
                          input_dim=CONFIG['seq_len'], 
                          output_classes=CONFIG['classes']).to(device)
                          
    # criterion = ROUNDClockLoss(locking_strength=0.05, states=CONFIG['classes']) 
    criterion = HarmonicROUNDLoss(locking_strength=0.1, harmonics=[2, 4, 8], terminal_only=True) 
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    accuracy_history = []
    
    for epoch in range(CONFIG['epochs']):
        optimizer.zero_grad()
        output, history = model(X, steps=CONFIG['steps'])
        loss, ce, locking = criterion(output, Y, history)
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(output, dim=1)
        acc = (preds == Y.squeeze()).float().mean().item()
        accuracy_history.append(acc)
        
        if epoch % 100 == 0:
            print(f"[ROUND Run {run_id}] Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc:.2f}")
            
    return accuracy_history, preds, Y

def train_gru(run_id, X, Y, device):
    model = GRUClockModel(input_dim=1, 
                        hidden_size=CONFIG['hidden_size'], 
                        output_classes=CONFIG['classes']).to(device)
                        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    accuracy_history = []
    
    for epoch in range(CONFIG['epochs']):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y.squeeze().long())
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(output, dim=1)
        acc = (preds == Y.squeeze()).float().mean().item()
        accuracy_history.append(acc)
        
        if epoch % 100 == 0:
            print(f"[GRU Run {run_id}] Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc:.2f}")

    return accuracy_history, preds, Y

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    device = torch.device(CONFIG['device'])
    print(f"Running on {device} - Clock Benchmark")
    
    X, Y = generate_clock_data(CONFIG['dataset_size'], CONFIG['seq_len'], CONFIG['classes'])
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
    
    ax.set_title(f'Cyclic Logic: ROUND vs GRU (Modulo-{CONFIG["classes"]})\nResults over {CONFIG["runs"]} Runs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.1)
    ax.legend()
    plt.tight_layout()
    plt.savefig('benchmark_clock.png')

    data_stack = np.stack(all_round_final_preds) 
    target_flat = final_targets.flatten()
    data_stack = np.vstack([data_stack, target_flat]) 
    labels = [f'Run {i+1}' for i in range(CONFIG['runs'])] + ['Target']
    corr = np.corrcoef(data_stack)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, interpolation='nearest', cmap='coolwarm', vmin=0, vmax=1)
    plt.title('ROUND Clock: Inter-Run Consistency')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = plt.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color="black" if 0.3 < corr[i, j] < 0.7 else "white")
    plt.tight_layout()
    plt.savefig('correlation_clock.png')
    
    print("Done. Saved benchmark_clock.png and correlation_clock.png")
