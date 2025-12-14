
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==============================================================================
# ROUND (Riemannian Optimized Unified Neural Dynamo)
# The "Sphere that contains the Cube".
# ==============================================================================

class PhaseAccumulator(nn.Module):
    """
    The Core Engine of ROUND.
    Operates purely in the Phase Domain via Accumulation (Addition).
    No complex multiplication. No matrix multiplication in the recurrence (conceptually).
    
    Formula:
        Phi_new = Phi_old + Delta_Phi
        Delta_Phi = NN(Cos(Phi_old), Sin(Phi_old), Cos(Phi_in), Sin(Phi_in))
    """
    def __init__(self, size):
        super().__init__()
        # 4 inputs: State(Cos,Sin) + Input(Cos,Sin)
        # Using a simple Linear layer to calculate the drift.
        # This allows the "Manifold" to be learned.
        self.drift_computer = nn.Linear(size * 4, size)

    def forward(self, phi_state, x_phasors):
        # 1. Project Topology (Polar -> Cartesian features)
        # We need Cos/Sin to avoid the 0/2PI discontinuity issue for the neural net.
        state_cos = torch.cos(phi_state)
        state_sin = torch.sin(phi_state)
        
        # x_phasors is already (Batch, Size, 2) -> (Cos, Sin)
        combined = torch.cat([state_cos, state_sin, x_phasors[:,:,0], x_phasors[:,:,1]], dim=1)
        
        # 2. Compute Drift (Rotation)
        delta_phi = self.drift_computer(combined)
        
        # 3. Accumulate (Spin)
        phi_new = phi_state + delta_phi
        
        return phi_new

class ROUNDModel(nn.Module):
    """
    Riemannian Optimized Unified Neural Dynamo
    Applicable for both Topological (Continuous) and Logical (Discrete) tasks.
    """
    def __init__(self, hidden_size=64, input_dim=16):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Encoder: Maps arbitrary inputs to Initial Phases
        self.encoder = nn.Linear(input_dim, hidden_size)
        
        # The Dynamo
        self.cell = PhaseAccumulator(hidden_size)
        
        # Readout: Interprets the final Phase Interference Pattern
        self.readout = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, steps=12):
        batch_size = x.size(0)
        device = x.device
        
        # 1. Encode Input -> Input Phasors (Stationary Waves)
        phi_in = self.encoder(x)
        x_phasors_cos = torch.cos(phi_in)
        x_phasors_sin = torch.sin(phi_in)
        x_phasors = torch.stack([x_phasors_cos, x_phasors_sin], dim=2)
        
        # 2. Initialize State (Vacuum / Zero Phase)
        phi_h = torch.zeros(batch_size, self.hidden_size).to(device)
        
        phi_history = []
        
        # 3. Spin the Dynamo (Recurrence)
        for _ in range(steps):
             phi_h = self.cell(phi_h, x_phasors)
             phi_history.append(phi_h)
             
        # 4. Readout
        # We read the final Cos/Sin components.
        final_cos = torch.cos(phi_h)
        final_sin = torch.sin(phi_h)
        features = torch.cat([final_cos, final_sin], dim=1)
        
        # Return Raw Logits, not Probabilities
        return self.readout(features), phi_history

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

class ROUNDClockModel(nn.Module):
    """
    Extension of ROUND for Multi-State (Clock) tasks.
    Reuses the exact same PhaseAccumulator 'Neuron' from the original file.
    """
    def __init__(self, hidden_size=64, input_dim=1, output_classes=8):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Encoder: Map Scalar -> Phase
        self.encoder = nn.Linear(input_dim, hidden_size)
        
        # The Core Dynamo (Reuse from ROUND.py)
        self.cell = PhaseAccumulator(hidden_size)
        
        # Readout: Map Phase State -> 8 Class Logits
        self.readout = nn.Linear(hidden_size * 2, output_classes)

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
             
        # 4. Readout
        final_cos = torch.cos(phi_h)
        final_sin = torch.sin(phi_h)
        features = torch.cat([final_cos, final_sin], dim=1)
        
        return self.readout(features), phi_history

class ROUNDLoss(nn.Module):
    """
    The Physics Engine.
    Combines Task Error (BCEWithLogitsLoss) with Quantum Locking (Riemannian Potential).
    """
    def __init__(self, locking_strength=0.1, terminal_only=False):
        super().__init__()
        # Use BCEWithLogitsLoss for numerical stability and correctness with raw logits
        self.bce = nn.BCEWithLogitsLoss()
        self.locking_strength = locking_strength
        self.terminal_only = terminal_only
        
    def forward(self, prediction, target, phi_history):
        # 1. Task Error
        task_loss = self.bce(prediction, target)
        
        # 2. Quantum Locking
        if self.terminal_only:
             states_to_lock = phi_history[-1]
        else:
             states_to_lock = torch.stack(phi_history)
             
        locking_potential = torch.mean(torch.sin(states_to_lock) ** 2)
        
        total_loss = task_loss + self.locking_strength * locking_potential
        return total_loss, task_loss.item(), locking_potential.item()

class ROUNDTopologyLoss(nn.Module):
    """
    Specialized Loss for Continuous Topology Tasks.
    Only locks the TERMINAL state, allowing free rotation during the sequence.
    """
    def __init__(self, locking_strength=0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.locking_strength = locking_strength
        
    def forward(self, prediction, target, phi_history):
        # 1. Task Error
        task_loss = self.bce(prediction, target)
        
        # 2. Terminal Locking
        final_phis = phi_history[-1]
        locking_potential = torch.mean(torch.sin(final_phis) ** 2)
        
        total_loss = task_loss + self.locking_strength * locking_potential
        return total_loss, task_loss.item(), locking_potential.item()

class ROUNDClockLoss(nn.Module):
    """
    The Physics Engine adapted for an 8-hour Clock.
    Instead of locking to 0, PI (2 states), we lock to 0, PI/4, PI/2... (8 states).
    """
    def __init__(self, locking_strength=0.1, states=8, terminal_only=False):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.locking_strength = locking_strength
        self.states = states
        self.terminal_only = terminal_only
        
    def forward(self, prediction, target, phi_history):
        # 1. Task Error
        task_loss = self.ce(prediction, target.squeeze().long())
        
        # 2. Multi-State Quantum Locking
        if self.terminal_only:
             states_to_lock = phi_history[-1]
        else:
             states_to_lock = torch.stack(phi_history)
             
        locking_freq = self.states / 2
        locking_potential = torch.mean(torch.sin(locking_freq * states_to_lock) ** 2)
        
        total_loss = task_loss + self.locking_strength * locking_potential
        return total_loss, task_loss.item(), locking_potential.item()

class HarmonicROUNDLoss(nn.Module):
    """
    Experimental: Harmonic Potential.
    Instead of locking to a single frequency (e.g. 8 states), this sums potentials
    across a spectrum of harmonics (e.g. 2, 4, 8).
    """
    def __init__(self, locking_strength=0.1, harmonics=[2, 4, 8], weights=None, mode='multiclass', terminal_only=False):
        super().__init__()
        self.mode = mode
        if mode == 'binary':
            self.task_loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.task_loss_fn = nn.CrossEntropyLoss()
            
        self.locking_strength = locking_strength
        self.harmonics = harmonics
        self.terminal_only = terminal_only
        
        if weights is None:
            self.weights = [1.0] * len(harmonics)
        else:
            self.weights = weights
            
    def forward(self, prediction, target, phi_history):
        # 1. Task Error
        if self.mode == 'binary':
             task_loss = self.task_loss_fn(prediction, target)
        else:
             task_loss = self.task_loss_fn(prediction, target.squeeze().long())
        
        # 2. Harmonic Quantum Locking
        if self.terminal_only:
             states_to_lock = phi_history[-1]
        else:
             states_to_lock = torch.stack(phi_history)
        
        total_locking = 0.0
        
        for i, states in enumerate(self.harmonics):
            freq = states / 2.0
            # V = sin^2(freq * phi)
            potential = torch.mean(torch.sin(freq * states_to_lock) ** 2)
            total_locking += self.weights[i] * potential
            
        total_locking /= len(self.harmonics)
        
        total_loss = task_loss + self.locking_strength * total_locking
        return total_loss, task_loss.item(), total_locking.item()
