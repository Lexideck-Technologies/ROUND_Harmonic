# UIT-ROUND Color Algebra: The Phasic Associativity Proof

**Objective**: Prove that `UIT_ROUND` can solve complex associative logic (e.g., Color Mixing) not by combinatorial search, but by **geometric phase accumulation**.

## The Hypothesis
If "Logic is Rotation," then "Association is Addition."
Comparing `UIT_ROUND` to a standard RNN:
- **Standard RNN**: Learns a complex non-linear boundary where $\{Red, Orange\} \to Result$.
- **UIT_ROUND**: Learns that "Red" is a vector rotation $\vec{v}_R$ and "Orange" is $\vec{v}_O$. The result is simply the manifold's natural state $\Phi_{result} = \Phi_{start} + \vec{v}_R + \vec{v}_O$.

This implies that **Color Algebra** is a single-layer, linear problem in Phasic Space.

## The Benchmark: `UIT_benchmark_color_algebra.py`

### 1. Data Generation (The Prism)
We define a synthetic "Color Wheel" of $N$ discrete colors (e.g., 12 or 16).
- Each color has a "True Phase" $\theta \in [0, 2\pi)$.
- **Task**: Given a sequence of 2 colors $(C_A, C_B)$, predict the "Mixed Color" $C_{mix}$.
- **Ground Truth**: The color closest to $(\theta_A + \theta_B) \mod 2\pi$ (or a weighted average).

### 2. The Architecture
- **Model**: `UITModel` (1 Layer, `hidden_size=16` or `32`).
- **Input**: One-hot encoding of the input colors.
- **Mechanism**:
    - The `delta_phi` parameter becomes a **learnable lookup table** of target phases.
    - When "Red" fires, the neuron "nudges" towards the Red phase.
- **Persistence**: `0.5` (The Mixing Axiom).
    - **Why?**: This implements the user's intuition: "Move half the distance to Orange."
    - **Math**: $\Phi_{new} = 0.5 \Phi_{old} + 0.5 \Phi_{input}$.
    - This naturally computes the **Geometric Midpoint**, creating "Burnt Orange" from Red and Orange sequences.
- **Readout**: A simple Linear Decoder that maps the final terminal phase $\Phi_{final}$ to the class probabilities of the Mixed Color.

### 3. Hyperparameters (The Harmonic Standard)
- **Learning Rate**: $2^{-7} \approx 0.0078125$ (Standard UIT Precision).
- **Optimizer**: Adam.
- **Batch Size**: 64.
- **Epochs**: ~500 (Expected convergence should be extremely fast, <50 epochs).

## Technical Goals
1.  **Geometric "Click"**: Prove that the model doesn't "memorize" the pairs, but actually learns the *topology* of the color wheel.
    - *Verification*: Train on 90% of pairs, test on 10% held-out pairs.
2.  **Zero-Shot Algebra**: If it learns "Red" and "Blue", does it correctly infer "Red + Blue = Purple" without ever seeing that specific pair?
    - **The "High" Claim**: Yes, because the geometry *forces* the sum to exist.

## Next Steps
1.  Implement `UIT_benchmark_color_algebra.py`.
2.  Generate the "Phase Wheel" plot showing the learned vectors.
