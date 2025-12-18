# Spinor Monism (Harmonic ROUND) v0.6.2

[![The U-Neuron](media/round_video_thumbnail.png)](https://www.lexidecktechnologies.com/UIT_IEG/The_U-Neuron.mp4)
<div align="center"><em>Click the thumbnail above to watch the 2-minute explainer.</em></div>

## Table of Contents
1. [The Story of ROUND: An Arc of Discovery](#the-story-of-round-an-arc-of-discovery)
2. [Executive Summary: Spinor Monism](#executive-summary-spinor-monism)
    1. [Deep Research Artifacts](#deep-research-artifacts-google-gemini)
3. [The Spinor Breakthrough: Solving the "Twist"](#the-spinor-breakthrough-solving-the-twist)
4. [What ROUND Is](#what-round-is)
5. [ROUND vs. GRU: The Stability of Memory](#round-vs-gru-the-stability-of-memory)
6. [Quickstart](#quickstart)
7. [Benchmark Results: v0.6.1 Grand Slam](#benchmark-results-v061-grand-slam)
8. [Theory: Unified Informatic Topology (UIT)](#theory-unified-informatic-topology-uit)
9. [Repo Layout](#repo-layout)
10. [License & Citation](#license--citation)
11. [Glossary of Terms](#glossary-of-terms)

---

## The Story of ROUND: An Arc of Discovery

The journey of the **Riemannian Optimized Unified Neural Dynamo (ROUND)** is a story of seeking simplicity at the intersection of geometry and information.

### Chapter 1: The Circle (v0.1.0)
We began with a simple hypothesis: what if an AI neuron didn't just "gate" information (like a GRU or LSTM), but "accumulated" it as a physical phase angle on a circle? This created a non-volatile memory cell, stable like a gyroscope, but it struggled with discrete logic.

### Chapter 2: The Rugged Landscape (v0.2.0 - v0.3.5)
To force the continuous phase into discrete "bins," we introduced **Harmonic Locking**. By stacking multiple potential wells (`HARMONICS = [1, 2, 4, 8]`), we created a complex landscape where the neuron could "lock" into different states. It worked, but it was noisy. Optimization was a "rugged" struggle, and benchmarks required delicate tuning.

### Chapter 3: The Spinor Breakthrough (v0.4.0)
The breakthrough came from physics. We realized that the "topological twist" of tasks like Parity (XOR) failed because the neuron couldn't distinguish between $0$ and $2\pi$ (a full wrap). By introducing **Spinor Features** (Spin-1/2), we projected the inputs onto the **Double Cover** of the circle ($4\pi$ range). Suddenly, the "twist" was visible.

### Chapter 4: The Great Simplification (v0.6.0)
With the power of Spinors, the complexity of the "Rugged Landscape" became unnecessary. In v0.6.0, we discovered that a **single, smooth fundamental harmonic (`[1]`)** combined with Spinor features solves every benchmark—Logic, Topology, and Structure—with perfect stability. 

### Chapter 5: Memory Crystallization (v0.6.2)
In the final step toward AGI-level mnemonic structures, we solved the "Long-Term Persistence" problem. By introducing a **Precessing Mnemonic Clock (Wobble Drift)** and **Gemination Deflectors**, we enabled a single 64-neuron cell to store an arbitrary sequence of high-entropy patterns (ASCII) with zero decay. This is **Content-Addressable Memory (CAM)**: a neural memory that is order-independent and stable over potentially infinite horizons.

**This is Spinor Monism: the realization that one perfect potential well, properly oriented in 3D phase-space, is a permanent mnemonic anchor.**

---

## Executive Summary: Spinor Monism

The contemporary landscape of computational theory has long been fractured by a dichotomy between the continuous and the discrete. The **Unified Informatic Topology (UIT)** framework offers a resolution to this divide by positing that information is a physical substrate with thermodynamic weight.

The **Spinor Monism** configuration (v0.6.1) establishes that a **single 32-neuron configuration** can span multiple computational regimes—Logic (XOR), Arithmetic (Counting), Structure (Recursion), and Topology (connectivity)—that typically require vastly different inductive biases.

### Deep Research Artifacts (Google Gemini)

Independent validation and explanation of the ROUND architecture:

- 🎬 **Video Explainer** (2 min): [The_U-Neuron.mp4](https://www.lexidecktechnologies.com/UIT_IEG/The_U-Neuron.mp4)
- 🎙️ **Podcast Episode** (32 min): [Phase_Memory_Solves_AI_Long-Term_Failure.m4a](media/Phase_Memory_Solves_AI_Long-Term_Failure.m4a)
- 📑 **Research Slide Deck**: [Unifying_Wave_and_Particle_Computation.pdf](media/Unifying_Wave_and_Particle_Computation.pdf)

---

## The Spinor Breakthrough: Solving the "Twist"

In standard RNNs, state is a vector in Euclidean space. In ROUND, state is a phase $\phi$ on a circle. Prior versions struggled with "Twist" problems where the state must loop back on itself but remember how many times it has spun.

By upgrading to **Spinor Features**:
$$
\Delta\phi_t = W\,[\cos(\phi), \sin(\phi), \mathbf{\cos(\phi/2), \sin(\phi/2)}, \cos(x), \sin(x)] + b
$$
The network now "feels" the difference between an odd and even number of rotations. This allows a circle to act as a Mobius strip or a higher-dimensional manifold, enabling the solution of 16-bit Parity with a single neuron.

---

## What ROUND Is

ROUND is a **phase-accumulating recurrent cell**:
- It represents hidden state as a phase vector **$\phi$** (radians).
- It updates state via **accumulation** (addition), not gating.
- It maintains **Long-term Stability**: Unlike GRUs which decay, ROUND's state is preserved by the topology of the circle itself.

---

## ROUND vs. GRU: The Stability of Memory

*   **GRU (Volatile):** Like holding water in cupped hands; requires active gating to prevent decay.
*   **ROUND (Stable):** Like a gyroscope; maintains state indefinitely via phase conservation. Spinor features allow it to maintain winding number counts over long horizons.

---

## Quickstart

### ⚠️ Hardware Warning
> **Caution:** This repository runs a "Full Battery" optimization test suite.
> *   **GPU Users:** Ensure you have a CUDA-compatible PyTorch installation. The benchmarks are optimized for CUDA and will run significantly faster.
> *   **CPU Users:** Running the full battery (`run_battery.py`) on a CPU is computationally intensive. It may cause high thermal loads (fans spinning at 100%) for extended periods (30+ minutes).
> *   **Disclaimer:** This code is provided "as-is". Run at your own risk. Monitor your system temperatures if running on laptops or purely air-cooled setups.

### Requirements
- Python 3.10+
- PyTorch (tested on 2.0+)
- NumPy, Matplotlib

### Running The Benchmarks
Run the full regression test to reproduce the v0.6.1 findings:
```bash
python run_battery.py
```

| Experiment | Script | CLI Command | Description |
| :--- | :--- | :--- | :--- |
| **Parity** | `benchmark_parity.py` | `python benchmark_parity.py` | 16-bit Recursive XOR chain. |
| **Topology** | `benchmark_topology.py` | `python benchmark_topology.py` | Euler Characteristic (Cycle Detection). |
| **Brackets** | `benchmark_brackets_masked.py` | `python benchmark_brackets_masked.py` | Dyck-2 recursive nesting depth. |
| **Colors** | `benchmark_colors.py` | `python benchmark_colors.py` | Semantic vector algebra. |
| **Oracle** | `benchmark_oracle.py` | `python benchmark_oracle.py` | QA consistency and bias. |
| **ASCII** | `benchmark_ascii.py` | `python benchmark_ascii.py` | Cyclic sequence generation. |
| **CAM** | `benchmark_long_term.py` | `python benchmark_long_term.py` | Long-term memory side-by-side vs GRU. |
| **Gauntlet** | `benchmark_order_independence.py` | `python benchmark_order_independence.py` | The Shuffled Order Independence Brutality Test. |

---

## Benchmark Results: v0.6.2 Order-Independent Mastery

We performed a Head-to-Head comparison between **ROUND (Spinor Monism)** and a standard **GRU** across the "Decathlon" suite. Results are from the `a09a99d1` regression battery.

### 7.1 The "Impossible" Logic Test (Parity)
*   **ROUND:** **100% Accuracy.** Snaps to the global optimum within 100 epochs.
*   **GRU:** **50% Accuracy.** Fails completely on 16-bit XOR chains.
*   ![Parity Benchmark](data/a09a99d1/benchmark_parity_a09a99d1.png)

### 7.2 Topological Invariants (Graph Cycles)
*   **ROUND:** **100% Accuracy.** Stable convergence on flattened graph adjacency matrices.
*   **GRU:** **Unstable.** Wide variance across seeds; prone to mode collapse.
*   ![Topology Benchmark](data/a09a99d1/benchmark_topology_a09a99d1.png)

### 7.3 Streaming Recursion (Brackets Masked)
*   **ROUND:** **100% Accuracy.** Handles Dyck-2 nesting in sequential mode.
*   **GRU:** **100% Accuracy.** 
*   ![Brackets Benchmark](data/a09a99d1/benchmark_brackets_masked_a09a99d1.png)

### 7.4 The Oracle (QA Consistency)
*   **ROUND:** **100% Accuracy.** Perfect consistency across binary classification tasks.
*   **GRU:** **100% Accuracy.**
*   ![Oracle Benchmark](data/a09a99d1/benchmark_oracle_a09a99d1.png)

### 7.5 Order Extraction (Permutations)
*   **ROUND:** **100% Accuracy.** Successfully extracts sequence order from shuffled prompts.
*   **GRU:** **Matches Performance.** Can learn fixed small-set permutations.
*   ![Permutations Benchmark](data/a09a99d1/benchmark_perms_vs_gru_a09a99d1.png)

### 7.6 Generative Creativity (ASCII)
*   **ROUND:** **100% Accuracy.** Perfect cyclic timing and zero drift.
*   **GRU:** **Sub-perfect.** Drifts on long sequences, losing periodicity.
*   ![ASCII Benchmark](data/a09a99d1/benchmark_ascii_a09a99d1.png)

### 7.7 Semantic Algebra (Colors)
*   **ROUND:** **~96% Accuracy.** Successfully learns vector-like relationships in symbolic space.
*   **GRU:** **~50% Accuracy.** Fails to map semantic sums.
*   ![Colors Benchmark](data/a09a99d1/benchmark_colors_a09a99d1.png)

### 7.8 The CAM Gauntlet (Long-Term Memory)
The definitive proof of ROUND's non-volatile nature. In this test, a model must learn 6 high-ASCII words in a random curriculum and remember the first word after 10,000 epochs of training on subsequent data.
*   **ROUND:** **100% Accuracy.** Perfect recall of all words regardless of training order.
*   **GRU:** **Catastrophic Forgetting (~88%).** Recalls later words but loses earlier ones.
*   ![CAM Benchmark](data/a09a99d1/benchmark_long_term_345d69b0.png)

### 7.9 Order Independence Brutality Test
Performing 3 complete dual-model training batches (10k epochs each) with randomized keyword orders.
*   **ROUND:** **100.0% Mean Accuracy.** 0.0% Standard Deviation. Zero Forgetfulness.
*   **GRU:** **~94.5% Mean Aggregate Accuracy.** Significant variance depending on the sequence (~83% recall on complex shuffles).
*   ![Gauntlet Results](data/a09a99d1/benchmark_long_term_9b8ab51c.png)

---

## Theory: Unified Informatic Topology (UIT)

### "The Sphere Contains the Cube"

The core hypothesis of UIT is that **discrete logic is a special case of continuous topology** under a quantizing potential.

*   **Logic (The Particle):** Discrete state transitions (XOR, AND, NOT).
*   **Topology (The Wave):** Continuous phase evolution and winding numbers.
*   **The Spinor (The Bridge):** By governing the winding rules of the wave, the Spinor connects the two, allowing a continuous system to execute perfect discrete logic without the brittleness of traditional symbolic AI.

---

## Repo Layout

*   `ROUND.py`: Core engine (`PhaseAccumulator` with Spinor features).
*   `benchmark_*.py`: Individual task harnesses (Decathlon suite).
*   `run_battery.py`: Full regression suite for reproducing v0.6.1 logs.
*   `config.py`: Centralized Golden Configuration (`HARMONICS=[1]`).

---

## License & Citation

**License:** MIT License.

**Citation:** Please cite **v0.6.1 Spinor Monism (The Grand Slam)**.

---

## Glossary of Terms

### Spinor Monism
The finalized v0.6.1 configuration using **Fundamental Harmonic (`[1]`)** locking and **Spinor ($1/2$)** Input Features. It represents the most efficient mapping of phase-space to logical state.

### Phase Accumulation
The core mechanic of the `PhaseAccumulator`. Unlike standard neurons that use multiplicative gates, ROUND updates its state via simple addition: $\phi_{t+1} = \phi_t + \Delta\phi_t$. This preserves information indefinitely unless explicitly modified.

### Spinor Cover (Double Cover)
The mathematical technique of projecting inputs onto the $4\pi$ range. This resolves the topological ambiguity between a $0$ rotation and a $2\pi$ rotation, enabling the neuron to "count" its spins and solve logic tasks like Parity.

### Locking Strength ($\lambda$)
Defined by `PEAK_LOCKING_STRENGTH` in `config.py`. It represents the "gravitational pull" of the harmonic potential wells. At $0.0625$ (The Golden Setting), it provides enough force to discretize the state without trapping it in local minima.

### Harmonic Spectrum
The set of frequencies (`HARMONICS`) used to define the potential landscape. While earlier versions used a complex spectrum like `[1, 2, 4, 8]`, v0.6.1 established that the fundamental harmonic `[1]` is universal when combined with Spinor features.

### Intrinsic Annealing (Wobble)
Implemented in the `WobblePhaseAccumulator`. When the phase is stuck at a potential barrier, the energy is converted into a rotation around the $Z$-axis (Latitude), allowing the state to slide around obstacles on the Bloch Sphere.

### Terminal-Only Locking
A training protocol where the quantization potential is only applied at the final time step of a sequence. This allows the neuron to maintain "Wave-like" fluidity during processing and "Collapse" into a discrete "Particle-like" state only at readout.

### Harmonic Reciprocal ($2^{-9}$)
The optimized learning rate (`LR = 0.001953125`) discovered in v0.3.0. It aligns the step size of gradient descent with the underlying geometry of the phase circle, preventing the "ejection" of the state from stable wells.

### The U-Neuron
A colloquial term for the ROUND cell, referring to its origins in the **Unified Informatic Topology** framework and its circular (U-shaped) manifold.
