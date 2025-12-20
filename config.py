# version 0.7.3 - "The Hyper-Resolution Basin" (Config)
import math

# ROUND Configuration
# Centralized constants for "Geometric Resonance" tuning.

def get_lock_strength(epoch, total_epochs, peak_strength=0.125, floor_strength=0.03125):
    # Gaussian Annealing: "Training Wheels Protocol"
    mu = total_epochs / 2.0
    sigma = total_epochs / 6.0
    factor = math.exp(-0.5 * ((epoch - mu) / sigma) ** 2)
    
    # Neural Shield Engagement (50% mark)
    floor = floor_strength if epoch > (total_epochs * 0.5) else 0.0
    return max(peak_strength * factor, floor)

# ==============================================================================
# DEFAULT / LEGACY CONFIGURATION (Fallback)
# ==============================================================================
HIDDEN_SIZE = 32
PEAK_LOCKING_STRENGTH = 0.125  # Titanium Standard
HARMONICS = [1, 2, 4, 8]
LR = 0.001953125 # 2^-9
EPOCHS_SHORT = 400
EPOCHS_LONG = 400
SPIN_FACTOR = 0.5
WOBBLE_GRAVITY = 0.1
WOBBLE_HARMONICS = [1]
WOBBLE_COUPLING = -1.0 # Long-Term Transplant Engine

# ==============================================================================
# DENSITY DUEL CONFIGURATION
# ==============================================================================
HIDDEN_SIZE_R = 64
HIDDEN_SIZE_G = 256

def get_fair_hidden(round_size):
    return round_size * 4

# ==============================================================================
# PER-TASK CONFIGURATIONS
# ==============================================================================

# 1. TOPOLOGY (Logic/Counting - High Sensitivity)
TOPOLOGY_CONFIG = {
    'HIDDEN_R': 32,
    'HIDDEN_G': 128,
    'PEAK_LOCKING_STRENGTH': 0.0625, # Back to Golden (Counting is fragile)
    'HARMONICS': [1],
    'EPOCHS': 400,
    'FLOOR': 0.015625,    # 2^-6
    'LR': 0.001953125,  # 2^-9
    'WOBBLE': False,     # Disable drift for static logic
    'SORT_INPUTS': True,
    'DELAYED_LOCKING': 0.5
}

# 2. PARITY (Modulo-2 Logic)
PARITY_CONFIG = {
    'HIDDEN_R': 1,
    'HIDDEN_G': 128,
    'PEAK_LOCKING_STRENGTH': 0.125, # Titanium
    'HARMONICS': [1],
    'EPOCHS': 400,
    'FLOOR': 0.03125,
    'LR': 0.01,
    'DELAYED_LOCKING': 0.5
}

# 3. BRACKETS (State Tracking)
BRACKETS_CONFIG = {
    'HIDDEN_R': 32,
    'HIDDEN_G': 128,
    'PEAK_LOCKING_STRENGTH': 0.125,
    'HARMONICS': [1],
    'EPOCHS': 400,
    'FLOOR': 0.03125,
    'LR': 0.001953125,
    'TERMINAL_ONLY': True
}

# 4. COLORS (Semantic Algebra)
COLORS_CONFIG = {
    'HIDDEN_R': 128,  # Hyper-Resolution (Plasma Grade)
    'HIDDEN_G': 512,  # 4x Density Duel
    'PEAK_LOCKING_STRENGTH': 0.125,
    'HARMONICS': [1, 2, 4, 8],
    'EPOCHS': 800,
    'FLOOR': 0.015625,    # 2^-6 (More Wiggle Room)
    'LR': 0.0009765625,  # 2^-10 (Lower Heat)
    'WOBBLE': True,
    'WOBBLE_GRAVITY': 0.0, # Pure Drift
    'WOBBLE_COUPLING': -1.0,
    'TERMINAL_ONLY': False
}

# 5. ASCII (Generative Creativity - Obsidian Standard)
ASCII_CONFIG = {
    'HIDDEN_R': 32,
    'HIDDEN_G': 128,
    'PEAK_LOCKING_STRENGTH': 0.5, # Obsidian Level (Proven for ASCII)
    'HARMONICS': [1, 2, 4, 8],
    'EPOCHS': 400,
    'FLOOR': 0.125,
    'LR': 0.001953125, # 2^-9
    'WOBBLE_GRAVITY': 0.1,
    'WOBBLE_COUPLING': -1.0,
    'TERMINAL_ONLY': False
}

# 6. ORACLE (QA Consistency)
ORACLE_CONFIG = {
    'HIDDEN_R': 32,
    'HIDDEN_G': 128,
    'PEAK_LOCKING_STRENGTH': 0.125,
    'HARMONICS': [1, 2, 4, 8],
    'EPOCHS': 400,
    'FLOOR': 0.03125,
    'LR': 0.0009765625,
    'TERMINAL_ONLY': True
}

# 7. PERMUTATIONS
PERMS_CONFIG = {
    'HIDDEN_R': 64,
    'HIDDEN_G': 256,
    'PEAK_LOCKING_STRENGTH': 0.125,
    'HARMONICS': [1],
    'EPOCHS': 1500,
    'LR': 0.0009765625,
    'FLOOR': 0.03125,
    'RUNS': 3
}

# 8. LONG TERM (Obsidian Shielding)
LONG_TERM_CONFIG = {
    'HIDDEN_R': 64,
    'HIDDEN_G': 256,
    'PEAK_LOCKING_STRENGTH': 0.5,
    'FLOOR': 0.125,
    'EPOCHS': 10000,
    'LR': 0.000244140625
}
