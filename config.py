
import math

# ROUND Configuration
# Centralized constants for "Geometric Resonance" tuning.

def get_lock_strength(epoch, total_epochs, peak_strength=0.0625, floor_strength=0.005):
    # Gaussian Annealing: "Training Wheels Protocol"
    # 1. Learn (Low) -> 2. Form (High) -> 3. Master (High/Maintenance)
    
    # Bell Curve centered at epoch/2
    mu = total_epochs / 2.0
    sigma = total_epochs / 6.0
    
    factor = math.exp(-0.5 * ((epoch - mu) / sigma) ** 2)
    
    # Floor: Ensure we never drop below maintenance gravity
    # ONLY apply in the second half to allow initial exploration.
    floor = floor_strength if epoch > (total_epochs / 2) else 0.0
    return max(peak_strength * factor, floor)

# ==============================================================================
# DEFAULT / LEGACY CONFIGURATION (Fallback)
# ==============================================================================
HIDDEN_SIZE = 32
PEAK_LOCKING_STRENGTH = 0.0625  # Standard (1/16) - The Golden Setting
HARMONICS = [1, 2, 4, 8]        # Standard Spectrum
LR = 0.001953125 # 2^-9
EPOCHS_SHORT = 300
EPOCHS_LONG = 300
SPIN_FACTOR = 0.5
WOBBLE_GRAVITY = 0.5
WOBBLE_HARMONICS = [1]
# To support legacy code
LOCKING_STRENGTH = PEAK_LOCKING_STRENGTH

# ==============================================================================
# PER-TASK CONFIGURATIONS
# ==============================================================================

# 1. TOPOLOGY (Flattened Graph Cycle Detection)
TOPOLOGY_CONFIG = {
    'HIDDEN_SIZE': 32,
    'PEAK_LOCKING_STRENGTH': 0.0625, # 1/16
    'HARMONICS': [1, 2, 4, 8],       # Full Spectrum
    'EPOCHS': 300,
    'FLOOR': 0.005,
    'LR': 0.001953125,
    'WOBBLE': False,
    'SORT_INPUTS': True
}

# 2. PARITY (16-bit XOR)
PARITY_CONFIG = {
    'HIDDEN_SIZE': 32,
    'PEAK_LOCKING_STRENGTH': 0.0625, # 1/16 (Harmonic) - Replaces 0.1
    'HARMONICS': [1, 2, 4, 8],       # Full Spectrum
    'EPOCHS': 300,
    'FLOOR': 0.01,
    'LR': 0.01,
    'DELAYED_LOCKING': 0.4
}

# 3. BRACKETS (Dyck-2)
BRACKETS_CONFIG = {
    'HIDDEN_SIZE': 32,
    'PEAK_LOCKING_STRENGTH': 0.0625, # 1/16 (Harmonic) - Replaces 0.5
    'HARMONICS': [1, 2, 4, 8],       # Full Spectrum
    'EPOCHS': 300,
    'LR': 0.001953125,
    'TERMINAL_ONLY': True
}

# 4. COLORS (Semantic Algebra)
COLORS_CONFIG = {
    'HIDDEN_SIZE': 32,
    'PEAK_LOCKING_STRENGTH': 0.0625,
    'HARMONICS': [1, 2, 4, 8],
    'EPOCHS': 300,
    'LR': 0.001953125,
    'TERMINAL_ONLY': False
}

# 5. ASCII (Generative Creativity)
ASCII_CONFIG = {
    'HIDDEN_SIZE': 32,
    'PEAK_LOCKING_STRENGTH': 0.0625,
    'HARMONICS': [1, 2, 4, 8],
    'EPOCHS': 300,
    'LR': 0.001953125,
    'TERMINAL_ONLY': False
}

# 6. ORACLE (QA Consistency)
ORACLE_CONFIG = {
    'HIDDEN_SIZE': 32,
    'PEAK_LOCKING_STRENGTH': 0.0625,
    'HARMONICS': [1, 2, 4, 8],
    'EPOCHS': 300,
    'LR': 0.001953125,
    'TERMINAL_ONLY': True
}
