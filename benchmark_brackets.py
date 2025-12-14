
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import random


from ROUND import ROUNDModel, ROUNDLoss, HarmonicROUNDLoss

# ==============================================================================
# Configuration
# ==============================================================================
CONFIG = {
    'task': 'brackets',
    'input_dim': 20, 
    'hidden_size': 64, 
    'steps': 30,          
    'epochs': 1000, 
    'batch_size': 64,
    'dataset_size': 4000, 
    'runs': 5,            
    'lr': 0.002,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ==============================================================================
# Dataset: Dyck-1
# ==============================================================================
def is_balanced(seq):
    balance = 0
    for char in seq:
        if char == 0: balance += 1
        else: balance -= 1
        if balance < 0: return False
    return balance == 0

def generate_dyck_data(n_samples, length):
    X_list = []
    Y_list = []
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos
    count_pos = 0
    count_neg = 0
    
    while count_pos < n_pos or count_neg < n_neg:
        batch_size = 100
        x = torch.randint(0, 2, (batch_size, length)).float()
        for i in range(batch_size):
            seq = x[i].tolist()
            if is_balanced(seq):
                if count_pos < n_pos:
                    X_list.append(x[i])
                    Y_list.append(1.0)
                    count_pos += 1
            else:
                if count_neg < n_neg:
                    X_list.append(x[i])
                    Y_list.append(0.0)
                    count_neg += 1
        if count_pos < n_pos:
            k = length // 2
            seq = []
            open_rem = k
            close_rem = k
            balance = 0
            while len(seq) < length:
                if open_rem > 0 and (balance == 0 or random.random() > 0.5 or close_rem == 0):
                    seq.append(0); open_rem -= 1; balance += 1
                elif close_rem > 0 and balance > 0:
                    seq.append(1); close_rem -= 1; balance -= 1
                elif open_rem > 0:
                    seq.append(0); open_rem -= 1; balance += 1
                else: seq.append(1) 
            if is_balanced(seq) and count_pos < n_pos:
                 X_list.append(torch.tensor(seq).float())
                 Y_list.append(1.0)
                 count_pos += 1

    idx = list(range(n_samples))
    random.shuffle(idx)
    X = torch.stack(X_list)[idx]
    Y = torch.tensor(Y_list)[idx].unsqueeze(1)
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
    criterion = HarmonicROUNDLoss(locking_strength=0.1, harmonics=[2, 4, 8], mode='binary', terminal_only=True) 
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
    model = GRUModel(input_dim=1, hidden_size=CONFIG['hidden_size']).to(device)
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
    print(f"Running on {device} - Brackets Benchmark")
    
    X, Y = generate_dyck_data(CONFIG['dataset_size'], CONFIG['input_dim'])
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
    
    ax.set_title(f'Ordered Logic: ROUND vs GRU (Brackets)\nResults over {CONFIG["runs"]} Runs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.1)
    ax.legend()
    plt.tight_layout()
    plt.savefig('benchmark_brackets.png')
    
    data_stack = np.stack(all_round_final_preds) 
    target_flat = final_targets.flatten()
    data_stack = np.vstack([data_stack, target_flat]) 
    labels = [f'Run {i+1}' for i in range(CONFIG['runs'])] + ['Target']
    corr = np.corrcoef(data_stack)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, interpolation='nearest', cmap='coolwarm', vmin=0, vmax=1)
    plt.title('ROUND Brackets: Inter-Run Consistency')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = plt.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color="black" if 0.3 < corr[i, j] < 0.7 else "white")
    plt.tight_layout()
    plt.savefig('correlation_brackets.png')
    
    print("Done. Saved benchmark_brackets.png and correlation_brackets.png")
