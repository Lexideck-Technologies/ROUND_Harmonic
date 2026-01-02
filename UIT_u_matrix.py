import torch
import torch.nn as nn
from UIT_ROUND import UITModel

# UIT-ROUND v0.8.0 - Project MONAD (U-Matrix)
# The Universal Matrix: A unified architecture for Physics, Logic, and Agency.

class UMatrix(nn.Module):
    def __init__(self, 
                 sensor_dim=12,   # 12-Tone HSL / Spectral Input
                 logic_dim=12,    # Corresponding State Dimension
                 hidden_dim=32,   # Internal "Monad" complexity
                 persistence_sensor=0.5,
                 persistence_logic=0.8,
                 quantization_strength=0.125,
                 harmonics=[1, 2, 4, 8],
                 spin_multiplier=1.0): 
        super().__init__()
        
        self.sensor_dim = sensor_dim
        self.logic_dim = logic_dim
        self.hidden_dim = hidden_dim # Store for init
        
        # --- LAYER 1: THE SENSORIUM (Prism Stack) ---
        self.sensorium = nn.ModuleDict({
            'sun': UITModel(sensor_dim, hidden_dim, logic_dim, num_layers=1, persistence=persistence_sensor, quantization_strength=quantization_strength, harmonics=harmonics, spin_multiplier=spin_multiplier),
            'canvas': UITModel(sensor_dim, hidden_dim, logic_dim, num_layers=1, persistence=persistence_sensor, quantization_strength=quantization_strength, harmonics=harmonics, spin_multiplier=spin_multiplier)
        })
        
        # --- LAYER 2: THE LOGIC CORE (Crystal) ---
        self.logic_core = UITModel(
            input_size=logic_dim * 2, # Takes fused Sun+Canvas input (Readouts)
            hidden_size=hidden_dim, 
            output_size=logic_dim, 
            num_layers=2,             # Deeper reasoning
            persistence=persistence_logic,
            quantization_strength=quantization_strength,
            harmonics=harmonics,
            spin_multiplier=spin_multiplier
        )
        
        # --- LAYER 3: THE CONTROL LOOP (Agency) ---
        dim_fused = hidden_dim * 3 # (h, cos, sin) from last reasoning layer
        self.controller = nn.Sequential(
            nn.Linear(dim_fused, 128),
            nn.ReLU(),
            nn.Linear(128, logic_dim) # Output Action/Prediction
        )
        
    def forward(self, input_signal, context_signal=None, h_states=None):
        """
        The U-Matrix Cycle (Manual Layer Stepping for Locked Engine Compatibility)
        """
        batch_size = input_signal.size(0)
        device = input_signal.device
        
        # Initialize states if None
        if h_states is None:
            h_states = self.init_states(batch_size, device)
            
        # Unpack States
        h_sun = h_states['sun']
        h_canvas = h_states['canvas']
        h_logic = h_states['logic'] # List of states for logic layers
        
        # --- STEP 1: SENSORIUM (Physics) ---
        # SUN
        sun_model = self.sensorium['sun']
        x_sun = input_signal[:, 0, :] # [Batch, Feature] (Seq len 1 assumed for step)
        out_sun, h_sun_new, _, sun_cos, sun_sin = sun_model.layers[0](x_sun, h_sun)
        
        # Sun Readout
        sun_feats = torch.cat([out_sun, sun_cos, sun_sin], dim=-1) # UITModel uses output+cos+sin
        y_sun = sun_model.readout(sun_feats) # [Batch, LogicDim]

        # CANVAS
        canvas_model = self.sensorium['canvas']
        if context_signal is None:
            x_canvas = torch.zeros_like(x_sun)
        else:
            x_canvas = context_signal[:, 0, :]
            
        out_canvas, h_canvas_new, _, can_cos, can_sin = canvas_model.layers[0](x_canvas, h_canvas)
        
        # Canvas Readout
        can_feats = torch.cat([out_canvas, can_cos, can_sin], dim=-1)
        y_canvas = canvas_model.readout(can_feats) # [Batch, LogicDim]
            
        # FUSE SENSORIUM Output (The "Observed Reality")
        logic_input = torch.cat([y_sun, y_canvas], dim=-1) # [Batch, LogicDim*2]
        
        # --- STEP 2: LOGIC CORE (Reasoning) ---
        # Logic Core takes fused readouts as input
        logic_model = self.logic_core
        
        if h_logic is None:
            h_logic = [torch.zeros(batch_size, logic_model.hidden_size).to(device) for _ in range(logic_model.num_layers)]
            
        current_input = logic_input
        new_logic_hs = []
        
        # Propagate through Logic Layers
        for i, layer in enumerate(logic_model.layers):
            current_input, h_next, _, h_cos_l, h_sin_l = layer(current_input, h_logic[i])
            new_logic_hs.append(h_next)
        
        # Logic Readout (High Level State)
        # Using `current_input` from last layer loop which is `standard_part * ...`?
        # Actually UITModel.forward uses `current_input` (which is updated to `output` of layer).
        # Wait, UITNeuronCell returns `output`.
        # `current_input` in loop becomes `output` of prev layer.
        
        # Readout from Last Layer
        logic_feats = torch.cat([current_input, h_cos_l, h_sin_l], dim=-1)
        y_logic = logic_model.readout(logic_feats) # [Batch, LogicDim]
        
        # --- STEP 3: CONTROL (Action) ---
        # Not used in benchmark? Benchmark calls `agent(x)`. Returns `y_p`.
        # Benchmark minimizes loss between `y_p` and `x`.
        # `x` is 1 dim. `y_logic` is 12 dim?
        # Benchmark `train_and_capture`: `criterion = nn.MSELoss()`. `loss(preds, x)`.
        # `x` has shape `(..., 1)`. 
        # `UMatrix` init `sensor_dim=1`. `logic_dim=1` in benchmark line 63.
        # So `y_logic` is size 1. Matches.
        
        # Pack States for next step
        new_states = {
            'sun': h_sun_new,
            'canvas': h_canvas_new,
            'logic': new_logic_hs
        }
        
        return y_logic, new_states

    def init_states(self, batch_size, device):
        # Helper to init all
        return {
            'sun': torch.zeros(batch_size, self.hidden_dim).to(device),
            'canvas': torch.zeros(batch_size, self.hidden_dim).to(device),
            'logic': None # Logic specific init handled in loop
        }
