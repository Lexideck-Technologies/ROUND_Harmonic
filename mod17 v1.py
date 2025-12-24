import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from ROUND import PhaseAccumulator  # Component import

# --- STRATEGY: Curriculum Learning & One-Hot Encoding ---
# Mod 17 is difficult because the state space is discrete and cyclic.
# Standard RNNs often fail to maintain precise counting over long sequences.
# Strategy 1: One-Hot Inputs. Instead of a scalar, give the network clear distinct inputs for 0-16.
# Strategy 2: Curriculum Learning. Start with short sequences and expand only when accurate.
# Strategy 3: High Capacity. Use a larger hidden state to ensure enough "frequency slots" for Mod 17.

class GenerativeROUND(nn.Module):
    def __init__(self, input_dim, hidden_size, output_classes, spinor=True, harmonic_init=False, phase_well_strength=0.0):
        super().__init__()
        self.h = hidden_size
        self.e = nn.Linear(input_dim, hidden_size)
        self.c = PhaseAccumulator(hidden_size, spinor=spinor)
        self.spinor = spinor
        self.phase_well_strength = phase_well_strength
        # Readout: [Cos, Sin] + [CosS, SinS if spinor]
        num_features = 2 + (2 if spinor else 0)
        readout_dim = hidden_size * num_features
        self.r = nn.Linear(readout_dim, output_classes)
        
        if harmonic_init and input_dim == output_classes: 
            print("Applying Harmonic Initialization (2pi/17)...")
            with torch.no_grad():
                modulus = input_dim
                angles = torch.linspace(0, 2*np.pi * (modulus-1)/modulus, modulus)
                self.e.weight.data = angles.unsqueeze(0).repeat(hidden_size, 1)
                self.e.bias.data.fill_(0.0)
                
    def forward(self, x, targets=None, teacher_forcing_ratio=0.0):
        # x: [Batch, Seq, InputDim]
        B, S, D = x.shape
        ph = torch.zeros(B, self.h, device=x.device)
        H = []
        logits_seq = []
        
        for t in range(S):
            xt = x[:, t, :]
            pt = self.e(xt)
            xpt = torch.stack([torch.cos(pt), torch.sin(pt)], 2)
            
            ph = self.c(ph, xpt)
            
            # Phase Well (The Snap): Discrete Attractors
            if self.phase_well_strength > 0:
                # Force phi towards k * 2pi/17
                # Potential V = -cos(17 * phi)
                # Force F = -dV/dphi = -17 * sin(17 * phi)
                # Update: phi += strength * F
                # We use a simplified snap: phi -= strength * sin(17 * phi)
                # Note: sin(17*phi) is 0 at k*pi/17. 
                # We want attractors at k*2pi/17 (even multiples of pi/17).
                # sin(17*phi) is 0 at these points.
                snap = torch.sin(17.0 * ph)
                ph = ph - self.phase_well_strength * snap
                
            # Teacher Forcing (The Correction)
            if targets is not None and teacher_forcing_ratio > 0:
                if torch.rand(1).item() < teacher_forcing_ratio:
                    # Reset phase to ideal target phase
                    # Target [Batch] -> [Batch, 1] -> [Batch, Hidden]
                    current_target = targets[:, t] # [Batch]
                    ideal_phase = current_target.float() * (2 * np.pi / 17.0)
                    # Broadcast to hidden size
                    ph = ideal_phase.unsqueeze(1).repeat(1, self.h)
            
            H.append(ph)
            
            readout_features_list = [torch.cos(ph), torch.sin(ph)]
            if self.spinor:
                ph_s = 0.5 * ph
                readout_features_list.extend([torch.cos(ph_s), torch.sin(ph_s)])
                
            readout_features = torch.cat(readout_features_list, 1)
            
            logits_seq.append(self.r(readout_features))
            
        return torch.stack(logits_seq, dim=1), H

class Mod17Generator:
    def __init__(self, batch_size, modulus=17):
        self.batch_size = batch_size
        self.modulus = modulus

    def generate(self, seq_len, device='cpu', one_hot=True, jitter=False):
        if jitter and seq_len > 1:
            # Jitter Curriculum: Pick random length in [1, seq_len]
            actual_len = torch.randint(1, seq_len + 1, (1,)).item()
        else:
            actual_len = seq_len
            
        # Generate random increments in range [0, modulus-1]
        increments = torch.randint(0, self.modulus, (self.batch_size, actual_len), device=device)
        
        # Calculate cumulative sum modulo 17
        targets = torch.cumsum(increments, dim=1) % self.modulus
        
        if one_hot:
            inputs = torch.nn.functional.one_hot(increments, num_classes=self.modulus).float()
        else:
            inputs = (increments.float().unsqueeze(2) / (self.modulus - 1)) * 2.0 - 1.0
        
        return inputs, targets, actual_len

def run_experiment(config):
    # Unpack Config
    name = config['name']
    use_one_hot = config['one_hot']
    use_curriculum = config['curriculum']
    use_spinor = config.get('spinor', True)
    harmonic_init = config.get('harmonic_init', False)
    resonance_loss = config.get('resonance_loss', False)
    use_jitter = config.get('jitter', False)
    
    # New Hybrids
    frozen_harmonic = config.get('frozen_harmonic', False)
    phase_well_strength = config.get('phase_well', 0.0)
    teacher_forcing_ratio = config.get('teacher_forcing', 0.0)
    clip_grad = config.get('clip_grad', 1.0)
    
    # Constants
    MODULUS = 17
    HIDDEN_SIZE = 128
    LEARNING_RATE = 0.002
    MAX_SEQ_LEN = 200
    START_SEQ_LEN = 10 if use_curriculum else MAX_SEQ_LEN
    BATCH_SIZE = 64
    EPOCHS = 3000
    ACCURACY_THRESHOLD = 0.99
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n--- Starting Experiment: {name} ---")
    print(f"One-Hot:{use_one_hot} | Curr:{use_curriculum} | Spin:{use_spinor} | Harm:{harmonic_init} | Res:{resonance_loss} | Jit:{use_jitter} | Frz:{frozen_harmonic} | Well:{phase_well_strength} | TF:{teacher_forcing_ratio}")
    
    # Model Setup
    input_dim = MODULUS if use_one_hot else 1
    model = GenerativeROUND(input_dim=input_dim, hidden_size=HIDDEN_SIZE, output_classes=MODULUS, spinor=use_spinor, harmonic_init=(harmonic_init or frozen_harmonic), phase_well_strength=phase_well_strength).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    generator = Mod17Generator(BATCH_SIZE, MODULUS)
    
    loss_history = []
    acc_history = []
    seq_len_history = []
    current_seq_len = START_SEQ_LEN
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        
        # Frozen Harmonic Logic: Freeze oscillator for first 30%
        if frozen_harmonic:
            freeze = epoch < (EPOCHS * 0.3)
            # Freeze/Unfreeze PhaseAccumulator params
            for param in model.c.parameters():
                param.requires_grad = not freeze
                
        optimizer.zero_grad()
        
        # Jitter: always pass current_seq_len as the max cap
        inputs, targets, actual_len = generator.generate(current_seq_len, DEVICE, one_hot=use_one_hot, jitter=use_jitter)
        
        outputs, hist = model(inputs, targets=targets, teacher_forcing_ratio=teacher_forcing_ratio)
        
        outputs_flat = outputs.view(-1, MODULUS)
        targets_flat = targets.view(-1)
        
        loss = criterion(outputs_flat, targets_flat)
        
        if resonance_loss:
            all_ph = torch.stack(hist) 
            l_lock = torch.mean(torch.sin((MODULUS / 2.0) * all_ph)**2)
            loss += 0.1 * l_lock 
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        
        preds = torch.argmax(outputs, dim=2)
        accuracy = (preds == targets).float().mean().item()
        
        loss_history.append(loss.item())
        acc_history.append(accuracy)
        seq_len_history.append(actual_len) 
        
        if use_curriculum:
            if accuracy > ACCURACY_THRESHOLD:
                if current_seq_len < MAX_SEQ_LEN:
                    current_seq_len += 5
                
        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Len: {actual_len} | Acc: {accuracy:.4f} | Loss: {loss.item():.4f}")

    total_time = time.time() - start_time
    print(f"Finished {name} in {total_time:.2f}s | Final Acc: {accuracy:.4f} | Max Len: {current_seq_len}")
    
    return {
        'loss': loss_history,
        'acc': acc_history,
        'seq': seq_len_history,
        'config': config
    }

def train_mod17():
    timestamp = f"{int(time.time()):x}"
    save_dir = f"data/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    strategies = [
        # classic
        {"name": "Baseline", "one_hot": False, "curriculum": False},
        {"name": "Curriculum", "one_hot": False, "curriculum": True},
        {"name": "One-Hot", "one_hot": True, "curriculum": False},
        {"name": "Combined", "one_hot": True, "curriculum": True},
        {"name": "No-Spinor", "one_hot": True, "curriculum": True, "spinor": False},
        # Scrappy
        {"name": "Harmonic Init", "one_hot": True, "curriculum": True, "harmonic_init": True},
        {"name": "Resonance Loss", "one_hot": True, "curriculum": True, "resonance_loss": True},
        {"name": "Jitter Curriculum", "one_hot": True, "curriculum": True, "jitter": True},
        # Hybrids
        {"name": "Frozen Harmonic", "one_hot": True, "curriculum": True, "frozen_harmonic": True},
        {"name": "Phase Well", "one_hot": True, "curriculum": True, "phase_well": 0.1},
        {"name": "Teacher Forcing", "one_hot": True, "curriculum": True, "teacher_forcing": 0.1},
        {"name": "Robust Harmonic", "one_hot": True, "curriculum": True, "harmonic_init": True, "jitter": True, "clip_grad": 0.1}
    ]
    
    results = {}
    
    for strategy in strategies:
        results[strategy['name']] = run_experiment(strategy)
        
    # --- Plotting Comparison ---
    plt.figure(figsize=(15, 10))
    
    # Plot Accuracy with Seaborn
    import pandas as pd
    import seaborn as sns
    from visualization_utils import setup_seaborn_theme

    palette = setup_seaborn_theme(style='darkgrid', palette='classic')

    # Prepare accuracy data
    acc_records = []
    for name, res in results.items():
        for epoch_idx, acc in enumerate(res['acc']):
            acc_records.append({'Epoch': epoch_idx, 'Accuracy': acc, 'Strategy': name})
    df_acc = pd.DataFrame(acc_records)

    plt.subplot(2, 1, 1)
    sns.lineplot(data=df_acc, x='Epoch', y='Accuracy', hue='Strategy', alpha=0.8)
    plt.title("Accuracy per Epoch", fontsize=14)
    plt.legend(loc='lower right', fontsize='small', ncol=2)

    # Plot Curriculum (keep matplotlib)
    plt.subplot(2, 1, 2)
    for name, res in results.items():
        # Smoothed line for jitter for readability
        data = res['seq']
        if res['config'].get('jitter', False):
             # Moving average for jitter
             data = np.convolve(data, np.ones(50)/50, mode='valid')
        plt.plot(data, label=name)

    plt.title("Sequence Length (Curriculum)", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Seq Length", fontsize=12)
    plt.legend(loc='lower right', fontsize='small', ncol=2)
    plt.grid(True, alpha=0.1)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/benchmark_mod17_ablation_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Results saved to {save_dir}")
    
    with open(f"{save_dir}/log_mod17_{timestamp}.txt", "w") as f:
        for name, res in results.items():
            f.write(f"{name}: Final Acc={res['acc'][-1]:.4f}, Max Len={res['seq'][-1]}\n")

if __name__ == "__main__":
    train_mod17()