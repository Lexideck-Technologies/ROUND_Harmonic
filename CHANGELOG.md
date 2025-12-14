# Changelog: Harmonic ROUND Optimization (v0.2.0 Candidate)

## Overview
This changelog documents the iterative optimization and tuning process of the Harmonic ROUND architecture, specifically focusing on the `HarmonicROUNDLoss` function and its hyperparameters. The goal was to find a "Universal Configuration" that provides stability and high performance across four diverse benchmarks: Brackets (Dyck-1), Clock (Modulo-8), Parity (16-bit XOR), and Topology (2D Winding).

## Key Improvements

### 1. Minimalist Harmonic Rail (`[1, 2]`)
*   **Change:** Reduced the harmonic series from `[1, 2, 4, 8, 16, 32]` down to just `[1, 2]`.
*   **Rationale:** We discovered that higher harmonics (`4, 8, 16, 32`) caused "Destructive Interference" in tasks sensitive to phase noise (specifically Parity).
*   **Observation:** The neuron does not need explicit higher harmonics to count or perform modulo arithmetic. By locking to the Fundamental (`1`) and the Octave (`2`), the neuron creates a solid "Ground State" (1-bit resolution). It then learns to "gear" its internal phase rotation to simulate higher frequencies as needed (e.g., solving Modulo-8 tasks with 62% accuracy purely via inference from a Modulo-2/4 rail).
*   **Result:** Eliminated "Ejection" events where the model would solve the task and then catastrophically diverge later in training.

### 2. The "Event Horizon" Floor Clamp (`0.032`)
*   **Change:** Implemented a `ReLU` clamp on the harmonic loss term: `loss = relu(sin^2(x) - 0.032)`.
*   **Rationale:** We observed that when the model achieved "Hyper-Precision" (loss < 0.02), it became unstable, likely due to floating-point precision issues or optimizer momentum in a near-zero gradient field.
*   **Observation:** By creating a "Dead Zone" (Event Horizon) at the bottom of the potential well (approx 10 degrees wide), we allow the neuron to settle without being harassed by microscopic locking gradients.
*   **Result:** 
    *   **Brackets:** 100% Stability across all runs. Loss floor drops to ~0.02.
    *   **Parity:** 100% Stability across all runs. No more "Quantum Dropouts".

### 3. Hyperparameter Tuning
*   **Locking Strength:** Tuned to `0.03125` (1/32). This value provides enough force to guide the neuron into the potential well but is weak enough ($\approx 3\%$) to allow the "Task Gradient" to dominate when necessary.
*   **Learning Rate:** Maintained at `0.002` (standard Adam default for RNNs), which balances well with the chosen locking strength.

### 4. Codebase Optimization & Minification
*   **Minified Architecture:** Compressed all benchmark scripts and the core `ROUND.py` library to remove boilerplate, enabling faster iteration and reducing context window usage.
*   **Data Management:** Introduced a centralized `data/` directory.
*   **Artifact Generation:** 
    *   Implemented `uuid4` batch IDs for all runs.
    *   Enhanced plotting code to include rich titles, labels, and correlation heatmaps even in the minified scripts.
    *   Automated logging to `data/log_{task}_{uid}.txt`.
    *   **Test Battery:** Created `run_battery.py` to automate the execution of all four benchmarks in sequence.


## Performance Summary (Final Configuration)

| Task | Configuration | Accuracy | Stability | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Brackets** | `[1,2]`, Str `1/32`, Floor `0.032` | **100%** | Perfect | Solves Dyck-1 effortlessly. Beats GRU. |
| **Parity** | `[1,2]`, Str `1/32`, Floor `0.032` | **100%** | Perfect | Solves 16-bit XOR. GRU fails (~93%). |
| **Clock** | `[1,2]`, Str `1/32`, Floor `0.032` | **~63%** | High | Beats GRU (~45%). Inferred Mod-8 from Mod-4 rail. |
| **Topology** | `[1,2]`, Str `1/32`, Floor `0.032` | **100%** | Perfect | Absolute lock. |

## Conclusion
The **Harmonic ROUND v0.2.0** (Candidate) demonstrates that a "Less is More" approach yields the most robust generalist neuron. By providing a broad, forgiving potential well (Rail [1, 2] + Floor Clamp), we allow the phase neuron to "jazz" around the solution space while preventing it from drifting into chaos. This is a significant advancement over the previous "Rigid Lock" approach.
