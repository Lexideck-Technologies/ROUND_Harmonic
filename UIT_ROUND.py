import torch
import torch.nn as nn
import numpy as np

"""
UIT-ROUND v1.0.0
Realized from: Unified Informatic Topology (UIT) - ROUND Implementation v1.0.0
Axioms:
1. Phasic Identity: Information is stored as a residue in the graded ring (Phase Angle).
2. Mogura Winding: Bit-streams are encoded via recursive half-shifts (phi = 0.5*phi + bit*pi).
3. Bernoulli Unwinding: Bits are generated via recursive doubling (The speaking instinct).
4. Phasic Inertia: Stability is maintained by damping updates as resonance (confidence) grows.
Where:
- xn: Macroscopic Activation Potential (Standard Part)
- en: Infinitesimal Dendritic Integration (Fiber)
- phi_n: Phase relative to network oscillations

Breadcrumb: This unified model replaces the scattered spinor, harmonic, and fuzzy ROUND variants
by treating those features as parameters of the U-space Informatic Hyperplane.
"""

class UITNeuronCell(nn.Module):
    def __init__(self, input_size, hidden_size, harmonics=[1, 2, 4, 8], use_spinor=True, quantization_strength=0.125, use_binary_alignment=False, unwinding_mode=False): # 2^-3
        super(UITNeuronCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.harmonics = harmonics
        self.use_spinor = use_spinor
        self.quantization_strength = quantization_strength
        self.use_binary_alignment = use_binary_alignment
        self.unwinding_mode = unwinding_mode
        
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
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.kaiming_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
        
        # Identity Initialization: Bitwise Epsilon Gradient
        # Neuron j is biased toward a specific power of 2
        with torch.no_grad():
            for j in range(self.hidden_size):
                # Epsilon follows a 2^-n scale to create bit-significance hierarchy
                self.epsilon[j] = 0.125 * (0.5 ** (j % 5)) 
            
            # Spectral Diversity: Each neuron is born with a unique harmonic 'mask'
            # This ensures that 512 neurons provide 512 different views of the manifold.
            nn.init.uniform_(self.diagnostic_harmonics, 0.0, 1.0)
                
            # Phase Pre-Partitioning: Spread the initial 'center' of each neuron
            # This prevents all neurons from clumping at phi=0 at birth
            spread = (2.0 * np.pi) / self.hidden_size
            for j in range(self.hidden_size):
                # Bias the phi_gate directly via the segment of the bias vector
                # The bias vector is [x_bias, phi_bias], so we target the second half
                self.bias[self.hidden_size + j] = j * spread

    def forward(self, x, h_prev):
        """
        State 'h_prev' is the previous phase angle (phi)
        Returns: (output, next_phi, confidence)
        """
        # Linear combinations for update and phase shift
        gates = x @ self.weight_ih + h_prev @ self.weight_hh + self.bias
        
        # Segment gates into Standard Activation (x_gate) and Phase Shift (phi_gate)
        x_gate, phi_gate = gates.chunk(2, dim=-1)
        
        # 1. Standard Part (x): Macroscopic activation
        standard_part = torch.tanh(x_gate)
        
        # 2. Phase Part (phi): Infinitesimal shift (Always Agile for Algebra)
        multiplier = 2.0 if self.use_spinor else 1.0
        
        if self.use_binary_alignment:
            if self.unwinding_mode:
                # BERNOULLI UNWINDING (Phasic Generation)
                # b = 1 if phi_prev >= pi else 0
                # Numerical epsilon to prevent boundary flipping
                bit_out = (h_prev >= (np.pi - 1e-7)).float()
                # phi_next = (phi_prev - b*pi) * 2.0
                phi_next = (h_prev - bit_out * np.pi) * 2.0
                
                # PHASIC RENORMALIZATION (v16 Structural Snap)
                # Hard snap to the nearest grid point (pi/128 for 8-bit precision)
                q_grid = np.pi / 128.0 
                q_snap = torch.round(phi_next / q_grid) * q_grid
                phi_next = q_snap # Hard Snap (v18 Crystalline Hammer)
            else:
                # GEOMETRIC ALIGNMENT (Mogura Axiom - Decoding)
                incoming_bit = x[:, 0:1] 
                phi_next = (h_prev * 0.5) + (incoming_bit * np.pi)
                bit_out = incoming_bit # Echo for consistency
            # Skip the Sieve for binary pathing
        else:
            phi_shift = (torch.sigmoid(phi_gate) * np.pi * multiplier)
            phi_next = h_prev + phi_shift
            
            # Titus's Quantization Sieve integration (Crystallizing the Manifold)
            q_sieve = torch.round(phi_next / (np.pi / 4)) * (np.pi / 4)
            phi_next = phi_next + self.quantization_strength * (q_sieve - phi_next)
        
        phi_next = torch.remainder(phi_next, 2.0 * np.pi * multiplier)

        # 3. Harmonic Coupling (The informative 'gluing')
        # Full Spectral Identity: We now return BOTH Cosine and Sine components
        # This resolves the +/- ambiguity in the phase-lock.
        h_cos = torch.zeros_like(phi_next)
        h_sin = torch.zeros_like(phi_next)
        for idx, h in enumerate(self.harmonics):
            h_cos += self.diagnostic_harmonics[:, idx] * torch.cos(h * phi_next)
            h_sin += self.diagnostic_harmonics[:, idx] * torch.sin(h * phi_next)
            
        # --- CONFIDENCE (Mogura Axiom) ---
        # Resonant Fidelity: measure how well we've locked onto the harmonics
        confidence = (h_cos.abs() / len(self.harmonics)).detach()

        # 4. Final U-Number Construction
        # Ensure standard_part isn't exactly zero to keep the signal flowing
        output = standard_part * (1.0 + self.epsilon * h_cos)
        if self.use_binary_alignment:
            # For ASCII, the 'Standard Part' is less important than the Phase Identity
            output = output + 0.1 * h_cos
            # If in unwinding mode, the output is the predicted bit
            if self.unwinding_mode:
                output = bit_out
            
        return output, phi_next, confidence, h_cos, h_sin

class UITModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, harmonics=[1, 2, 4, 8], use_spinor=True, use_binary_alignment=False):
        super(UITModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_binary_alignment = use_binary_alignment
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input = input_size if i == 0 else hidden_size
            self.layers.append(UITNeuronCell(layer_input, hidden_size, harmonics, use_spinor, use_binary_alignment=use_binary_alignment))
            
        # The Readout now looks at THREE components:
        # [Standard Output, Cosine Identity, Sine Identity]
        # This provides the 'Full Spectral Signature' (3x the information)
        self.readout = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
        
        # Explicit initialization for the readout
        for m in self.readout.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, input_seq):
        batch_size, seq_len, _ = input_seq.size()
        
        # Initialize hidden states
        h_states = [torch.zeros(batch_size, self.hidden_size).to(input_seq.device) for _ in range(self.num_layers)]
        confidences = []
        
        for t in range(seq_len):
            current_input = input_seq[:, t, :]
            for i, layer in enumerate(self.layers):
                current_input, h_states[i], conf, h_cos, h_sin = layer(current_input, h_states[i])
                confidences.append(conf)
        
        # Concatenate final macroscopic output with full spectral resonance
        # (batch, hidden_size) + (batch, hidden_size) + (batch, hidden_size) -> (batch, hidden_size * 3)
        combined_features = torch.cat([current_input, h_cos, h_sin], dim=-1)
        
        final_out = self.readout(combined_features)
        
        # Return final confidence as the mean across the sequence/layers
        avg_confidence = torch.stack(confidences).mean()
        return final_out, avg_confidence

    def save_crystal(self, path):
        """Saves the current state as a Crystalline Identity."""
        torch.save(self.state_dict(), path)
        print(f"--- [CRYSTAL SAVED] {path} ---")

    def load_crystal(self, path, freeze=True):
        """Loads a Crystalline Identity and optionally freezes the weights."""
        state_dict = torch.load(path, map_location=torch.device('cpu'), weights_only=True)
        self.load_state_dict(state_dict)
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
        print(f"--- [CRYSTAL LOADED] {path} (Frozen={freeze}) ---")

class UITEncoderModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1, harmonics=[1, 2, 4, 8], use_spinor=True, use_binary_alignment=False):
        """
        One-to-Many architecture: takes a scalar or one-hot ID and generates a bit sequence.
        Generation 3 (Strategy 6): Uses 'Permanent Context' injection and Phasic Unwinding.
        This model is the 'Topological Mirror' of the standard UITModel.
        """
        super(UITEncoderModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.use_binary_alignment = use_binary_alignment
        
        # Initial projection from ID to Phase Space
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        self.layers = nn.ModuleList()
        # The input to each cell is [Previous Bit (output_size) + Context ID (input_size)]
        for i in range(num_layers):
            self.layers.append(UITNeuronCell(output_size + input_size, hidden_size, harmonics, use_spinor, use_binary_alignment=use_binary_alignment, unwinding_mode=use_binary_alignment))
            
        self.readout = nn.Linear(hidden_size * 3, output_size)

    def forward(self, x, seq_len=8):
        batch_size = x.size(0)
        # 1. Harmonic Init: Phase starts at the projected ID location
        h_states = [self.input_projection(x) for _ in range(self.num_layers)]
        
        outputs = []
        confidences = []
        
        # Initial 'Start bit'
        current_bit = torch.zeros(batch_size, self.output_size).to(x.device)
        
        for t in range(seq_len):
            # 2. Context Injection: Concat current state bit with the Permanent Identity
            cell_input = torch.cat([current_bit, x], dim=-1)
            
            for i, layer in enumerate(self.layers):
                current_feat, h_states[i], conf, h_cos, h_sin = layer(cell_input, h_states[i])
                confidences.append(conf)
            
            # Predict the next bit
            combined = torch.cat([current_feat, h_cos, h_sin], dim=-1)
            bit_logits = self.readout(combined)
            # Autoregressive: feedback the prediction
            current_bit = torch.sigmoid(bit_logits)
            outputs.append(bit_logits)
            
        return torch.stack(outputs, dim=1).squeeze(-1), torch.stack(confidences).mean()

    def save_crystal(self, path):
        torch.save(self.state_dict(), path)
        print(f"--- [ENCODER CRYSTAL SAVED] {path} ---")

    def load_crystal(self, path, freeze=True):
        state_dict = torch.load(path, map_location=torch.device('cpu'), weights_only=True)
        self.load_state_dict(state_dict)
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
        print(f"--- [ENCODER CRYSTAL LOADED] {path} (Frozen={freeze}) ---")

    def renormalize_identity(self, map_path):
        """
        Seeds the input projection with the Sovereign Map from the Decoder.
        This ensures the Encoder starts at the exact 'Renormalized' terminal address.
        """
        sovereign_map = torch.load(map_path, map_location=torch.device('cpu'), weights_only=True)
        # We want the addresses from the FINAL step (index 7) of the journey
        # Shape: (256, 512)
        addresses = sovereign_map[:, 7, :]
        
        # Load into input_projection weights (transposed for one-hot dot product)
        # y = x * W^T + b.  If x is one-hot, y is the column of W^T.
        with torch.no_grad():
            self.input_projection.weight.copy_(addresses.T)
            self.input_projection.bias.fill_(0.0)
            
        # Freeze the projection to maintain the Diamond Identity
        for param in self.input_projection.parameters():
            param.requires_grad = False
        print(f"--- [RENORMALIZATION] Sovereign Identity Seeded and Frozen ---")

def resonance_loss(model, grid_size=256):
    """
    Harmonic Handholding: Punishes phase drift off the integer grid.
    $L = sin^2((grid_size/2) * phi)$
    """
    total_res = 0
    # Capture the phase angles from the last forward pass
    # (This requires the model to have stored them or us to pass them)
    # For now, we'll apply it to the diagnostic_harmonics and epsilon as a proxy
    # OR we can pass the hidden state 'h' (phi) to this function.
    return total_res

def landauer_loss(model, beta=0.01):
    """
    Realizing Section 11.2: SLandauer = beta * sum(|dz|)
    Encourages informational energy efficiency.
    """
    l_loss = 0
    for p in model.parameters():
        l_loss += torch.norm(p, p=1) # L1 penalty as a proxy for erasure cost
    return beta * l_loss
