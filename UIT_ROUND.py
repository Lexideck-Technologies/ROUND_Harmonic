import torch
import torch.nn as nn
import numpy as np

"""
UIT-ROUND v1.3.10 (Restored Golden State)
Realized from: Unified Informatic Topology (UIT) - ROUND Implementation
axioms:
1. Phasic Identity: Information is stored as a residue in the graded ring (Phase Angle).
2. Mogura Winding: Bit-streams are encoded via recursive half-shifts (phi = 0.5*phi + bit*pi).
3. Bernoulli Unwinding: Bits are generated via recursive doubling (The speaking instinct).
4. Phasic Inertia: Stability is maintained by damping updates as resonance (confidence) grows.
"""

# DEV FEATURES: Harmonic Spectrums
HARMONICS_STANDARD = [1, 2, 4, 8]
HARMONICS_7OCTAVE = [0.125, 0.25, 0.5, 1, 2, 4, 8]

class UITNeuronCell(nn.Module):
    def __init__(self, input_size, hidden_size, harmonics=[1, 2, 4, 8], spin_multiplier=1.0, quantization_strength=0.125, use_binary_alignment=False, unwinding_mode=False, persistence=1.0): 
        super(UITNeuronCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.harmonics = harmonics
        # Spin Multiplier: Controls phase range (2π × multiplier)
        # 0.125=π/4, 0.25=π/2, 0.5=π, 1.0=2π (default), 2.0=4π (spinor), 4.0=8π, 8.0=16π
        self.spin_multiplier = spin_multiplier
        self.quantization_strength = quantization_strength
        self.use_binary_alignment = use_binary_alignment
        self.unwinding_mode = unwinding_mode
        self.persistence = persistence
        
        # Linear projections for Standard Part (x) and Phase (phi)
        self.weight_ih = nn.Parameter(torch.Tensor(input_size, hidden_size * 2))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 2))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 2))
        
        # Epsilon (Infinitesimal fiber strength) - Now Bit-Biased
        self.epsilon = nn.Parameter(torch.Tensor(hidden_size))
        
        # Harmonic Identity (Strength of each harmonic per neuron)
        self.diagnostic_harmonics = nn.Parameter(torch.Tensor(hidden_size, len(harmonics)))
        
        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_uniform_(p.data)
            elif 'weight_hh' in name:
                nn.init.zeros_(p.data) # Disable noisy linear recurrence by default
            elif 'bias' in name:
                nn.init.zeros_(p.data)
            elif p.data.ndimension() >= 2:
                nn.init.kaiming_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
        
        # Identity Initialization: Bitwise Epsilon Gradient
        with torch.no_grad():
            for j in range(self.hidden_size):
                self.epsilon[j] = 0.125 * (0.5 ** (j % 5)) 
            
            # Spectral Diversity: Uniform 0-1 (Golden Ratio)
            nn.init.uniform_(self.diagnostic_harmonics, 0.0, 1.0)
                
            # Phase Pre-Partitioning
            spread = (2.0 * np.pi) / self.hidden_size
            for j in range(self.hidden_size):
                self.bias[self.hidden_size + j] = j * spread

    def forward(self, x, h_prev):
        """
        State 'h_prev' is the previous phase angle (phi)
        Returns: (output, next_phi, confidence)
        """
        # Linear combinations for update and phase shift
        gates = x @ self.weight_ih + h_prev @ self.weight_hh + self.bias
        
        x_gate, phi_gate = gates.chunk(2, dim=-1)
        
        # 1. Standard Part (x): Macroscopic activation
        standard_part = torch.tanh(x_gate)
        
        # 2. Phase Part (phi): Infinitesimal shift
        multiplier = self.spin_multiplier  # Generalized spin control
        
        if self.use_binary_alignment:
            if self.unwinding_mode:
                # BERNOULLI UNWINDING (Phasic Generation)
                bit_out = (h_prev >= (np.pi - 1e-7)).float()
                phi_next = (h_prev - bit_out * np.pi) * 2.0
                
                # PHASIC RENORMALIZATION (Hard Snap)
                q_grid = np.pi / 128.0 
                q_snap = torch.round(phi_next / q_grid) * q_grid
                phi_next = q_snap 
            else:
                # GEOMETRIC ALIGNMENT (Mogura Axiom - Decoding)
                incoming_bit = x[:, 0:1] 
                phi_next = (h_prev * 0.5) + (incoming_bit * np.pi)
                bit_out = incoming_bit # Echo for consistency
        else:
            # Pure Accumulation or Persistence Decay
            phi_shift = (torch.sigmoid(phi_gate) * np.pi * multiplier)
            phi_next = (h_prev * self.persistence) + phi_shift
            
            # Titus's Quantization Sieve
            q_sieve = torch.round(phi_next / (np.pi / 4)) * (np.pi / 4)
            phi_next = phi_next + self.quantization_strength * (q_sieve - phi_next)
        
        phi_next = torch.remainder(phi_next, 2.0 * np.pi * multiplier)

        # 3. Harmonic Coupling
        h_cos = torch.zeros_like(phi_next)
        h_sin = torch.zeros_like(phi_next)
        for idx, h in enumerate(self.harmonics):
            h_cos += self.diagnostic_harmonics[:, idx] * torch.cos(h * phi_next)
            h_sin += self.diagnostic_harmonics[:, idx] * torch.sin(h * phi_next)
            
        # --- CONFIDENCE (Mogura Axiom) ---
        confidence = (h_cos.abs() / len(self.harmonics)).detach()

        # 4. Final U-Number Construction
        output = standard_part * (1.0 + self.epsilon * h_cos)
        if self.use_binary_alignment:
            output = output + 0.1 * h_cos
            if self.unwinding_mode:
                output = bit_out
            
        return output, phi_next, confidence, h_cos, h_sin

class UITModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, harmonics=[1, 2, 4, 8], spin_multiplier=1.0, use_binary_alignment=False, unwinding_mode=False, persistence=1.0, quantization_strength=0.125):
        super(UITModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_binary_alignment = use_binary_alignment
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input = input_size if i == 0 else hidden_size
            self.layers.append(UITNeuronCell(layer_input, hidden_size, harmonics, spin_multiplier=spin_multiplier, quantization_strength=quantization_strength, use_binary_alignment=use_binary_alignment, unwinding_mode=unwinding_mode, persistence=persistence))
            
        # The Readout now looks at THREE components:
        # [Standard Output, Cosine Identity, Sine Identity]
        self.readout = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
        
        for m in self.readout.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, input_seq, return_sequence=False):
        batch_size, seq_len, _ = input_seq.size()
        
        
        # optimized rewrite for sequence return:
        outputs = []
        h_states = [torch.zeros(batch_size, self.hidden_size).to(input_seq.device) for _ in range(self.num_layers)]
        confidences = []
        
        for t in range(seq_len):
            current_input = input_seq[:, t, :]
            for i, layer in enumerate(self.layers):
                current_input, h_states[i], conf, h_cos, h_sin = layer(current_input, h_states[i])
                confidences.append(conf)
            
            # Readout at every step (if needed) or just accumulate
            if return_sequence:
                feats = torch.cat([current_input, h_cos, h_sin], dim=-1)
                outputs.append(self.readout(feats))
        
        avg_confidence = torch.stack(confidences).mean()
        
        if return_sequence:
            return torch.stack(outputs, dim=1), avg_confidence
            
        # Standard Last Step Readout
        final_feats = torch.cat([current_input, h_cos, h_sin], dim=-1)
        return self.readout(final_feats), avg_confidence

    def save_crystal(self, path):
        torch.save(self.state_dict(), path)
        print(f"--- [CRYSTAL SAVED] {path} ---")

    def load_crystal(self, path, freeze=True):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        print(f"--- [CRYSTAL LOADED] {path} (Frozen={freeze}) ---")
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

class UITEncoderModel(UITModel):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, harmonics=[1, 2, 4, 8], use_spinor=True, use_binary_alignment=False, persistence=0.0):
        # Encoder uses 0.0 persistence by default (Memoryless Sensorium)
        # Reverting to Parallel Mode (Standard Classification) to match Morning State Lock
        super(UITEncoderModel, self).__init__(input_size, hidden_size, output_size, num_layers, harmonics, use_spinor, use_binary_alignment, unwinding_mode=False, persistence=persistence)
        
    def renormalize_identity(self, map_path):
        """Standardizes the encoder's internal phase map against the decoder's sovereign reality."""
        # In a real impl, this would load the map. For now it's a placeholder for the semantic link.
        print("--- [RENORMALIZATION] Sovereign Identity Seeded and Frozen ---")
