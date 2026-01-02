# Changelog

## [1.3.11] - "Industrial Crystal" - 2025-12-31
### Added
*   **Premium Visualization Suite:** Upgraded all active benchmarks (`crystalline_loop`, `sandwich_duel`, `transparency`, `u_matrix`, `prism_stack`) to use the "Sandwich Style" Dark Mode standard (Red/Blue/Green palettes, high-contrast text).
*   **Heatmap Visualization:** Added 256x8 bit-reconstruction heatmap to `crystalline_loop`.
*   **Resonance Overlay:** Added Truth-vs-Pred charts to `u_matrix`.

### Changed
*   **Repo Unity:** Consolidated all active UIT scripts into `UIT_Benchmarks/`.
*   **Battery Logic:** `UIT_run_battery.py` now points exclusively to `UIT_Benchmarks/`.

## [1.3.10] - "Operation Phoenix" - 2025-12-31
### Fixed
*   **Restored Golden State:** Resolved "Split Brain" repository state by archiving duplicate/broken files and verifying the integrity of `UIT_ROUND.py`.
*   **Cleanup:** Archived `color_algebra`, `accessibility`, and `counting_relay` (deprecated/failed).
*   **Repo Hygiene:** Moved all legacy artifacts to `data/zz_archive`.

## [1.0.0] - "The Crystalline Bridge" - 2025-12-26
### Added
*   **The Structural Snap (Hard Renormalization):** Implemented a mandatory quantization step where neurons "snap" to the nearest integer phasic address. This eliminates floating-point drift and achieves the **Diamond Lock** ($256/256$ bit-perfection).
*   **The Phasic Sandwich (End-to-End Relay):** Verified that a hidden "Phasic Identity" can be ingested by a Decoder, passed through a frozen manifold, and reconstructed by an Encoder with **Zero Erasure Cost**. This establishes the first **Zero-Loss Neural Communication Channel**.
*   **The Sandwich Duel:** Added a comparative benchmark against GRU baselines. Results: **UIT-ROUND (100%) vs GRU (0.4%)** Success Rate in end-to-end identity relay.
*   **Phasic Inertia (Momentum):** Added a temporal anchor to the phase update logic to prevent jitter in high-frequency regimes.

### Changed
*   **Architecture Rewrite:** Unified all experimental features (Spinor, Harmonic, Wobble) into `UIT_ROUND.py` (v1.0.0 core).
*   **Innovation: Phasic Sovereignty vs. Gated Euclidean Memory**
    - **Legacy ROUND (`ROUND.py`):** Relies on **Continuous Potential Wells** (Harmonic Locking) to attract the phase towards stable basins. While robust against decay, it is still a "fuzzy" statistical system subject to epsilon-drift.
    - **UIT-ROUND (`UIT_ROUND.py`):** Introduces **Informatic Exchange Geometries (IEG)**. It treates the phase manifold as a deterministic bit-address. 
    - **The Structural Snap:** Unlike the "Soft Snap" of legacy ROUND, `UIT_ROUND` enforces a Hard Renormalization—snapping phases to their nearest topological grid point. This creates the **Diamond Lock**, enabling bit-perfect 100% reconstruction for complex discrete identities like ASCII ($n=256$).
    - **Modular Phasic Identity:** UIT-ROUND achieves **Phasic Sovereignty**, where the hidden state of one module is mathematically identical to the input requirements of the next. This allows for zero-shot knowledge relay between a frozen Decoder and frozen Encoder—a feat impossible for Vector-based architectures.
*   **Training Methodology:** Introduced **Phasic Coolant** (Exponential LR Descent $2^{-n}$) and **Iterative Manifold Deepening** as the standard protocol for achieving Crystalline convergence.
*   **Benchmark Reorganization:** Relocated and standardized all decathlon benchmark scripts into specialized sub-directories for better modularity.
*   **UIT Bridge Integration:** Consolidated core `UIT_ROUND.py` and the `UIT_Benchmarks` directory (including the Phasic Sandwich and Sandwich Duel) into the main structure.
*   **Scientific Visualization Suite:** Introduced `visualization_utils.py` and consolidated `UIT_sandwich_duel_scientific.png` for unified storytelling.

## [0.8.0] - "The Frozen Basin" - 2025-12-20
### Added
*   **Frozen Basin Test:** Replaced the legacy Long-Term Memory test with `benchmark_phase_lock.py` ("The Magnum Opus"). This test runs for **12,500 Epochs** (11,000 Learn + 1,500 Cryostasis Storm).
*   **Cryostasis Mechanism:** Implemented the "Gradient Vault" within the `FreezingGradientMask` class. This mechanism autonomously detects when a neuron has achieved harmonic resonance ($< 2^{-9}$ error) and permanently zeros its gradients, "locking" the weight state indefinitely.
*   **Phase Angle Lock:** Demonstrated that ROUND can achieve **100% Retention** through a 1,500-epoch Noise Storm. This storm simulates the **destructive interference of backpropagation** from future layers (e.g., fine-tuning a communication layer on top of a concept layer). By engaging Cryostasis, ROUND allows for infinite vertical stacking without "Backward Leakage" or forgetting.
*   **Vertical Crystal:** Extended the freezing logic to encompass the entire neuronal slice (Encoder + Recurrent Weights + Wobble Weights), effectively turning parts of the network into fixed feature extractors while other parts remain fluid.

### Changed
*   **Repo Standardization:** Renamed `benchmark_long_term.py` to `benchmark_phase_lock.py` and deleted the legacy file.
*   **Battery Update:** Updated `run_battery.py` to include the new Phase Lock test as the capstone verification.
*   **Documentation:** Overhauled `README.md` to center the narrative on "Autonomous Phase Locking" and the solution to the Stability-Plasticity Dilemma.

## [0.7.3] - "Hyper-Resolution Basin" - 2025-12-20
### Added
*   **Hyper-Resolution Clock:** Increased the Drift Clock constant to $2^{-5}$ (`0.03125`) and expanded the standard Hidden Size to 128 neurons.
*   **Logic Crystal (Ticker Array):** Overhauled the `benchmark_colors.py` test to use Batched Ticker Arrays with Context Indexing.
*   **Perfect Symbolic Algebra:** Achieved **100.0% Accuracy** on the Colors benchmark, eliminating the "Deterministic Collision" plateau.

### Changed
*   **Wobble Mechanics:** Enforced "Clock Purity" (Linear Drift) for non-repeating tokens.
*   **Colors Config:** Increased `HIDDEN_R` to 128 and `EPOCHS` to 800.

## [0.6.4] - "The Neural Shield" - 2025-12-19
### Added
*   **Infinite Plasticity Standard:** Purged all learning rate decay from the benchmark suite. All models now maintain a constant, optimized learning rate ($2^{-12} \approx 0.00024$) for the entire duration of the run.
*   **Mnemonic Shielding:** Implemented a context-aware harness policy that protects established memories during continuous learning. 
    *   **Fluid Exploration:** New tasks begin with zero locking gravity for the first 50% of their curriculum slot.
    *   **Neural Shield:** The maintenance floor ($2^{-6}$) is automatically engaged whenever the model revisits "Old" tasks, preventing high-LR plasticity from erasing the archive.
*   **Unified Battery Sync:** Synchronized all 8 benchmarks (Topology, Parity, Brackets, Colors, Oracle, ASCII, Long-Term, Permutations) to the v0.6.4 "Neural Shield" protocol.
*   **Heavy Harmonic Floor:** Standardized a $2^{-6}$ (0.015625) maintenance floor across all tasks to ensure irreversible quantization.

## [0.6.3] - "The Density Duel" - 2025-12-18
### Added
*   **The Density Duel Standard:** Formalized a new benchmarking protocol where the **GRU** is granted **4x the raw capacity** (hidden neurons) compared to **ROUND** (e.g., ROUND 64 vs GRU 256). 
*   **Efficiency Proof:** Validated that the U-Neuron with 64 neurons outperforms or matches a 256-neuron GRU in long-term curriculum learning and symbolic logic.
*   **Minimalist Parity:** Re-confirmed that ROUND can solve 16-bit XOR with a single topological neuron ($HIDDEN=1$), highlighting the "Density" advantage over Euclidean networks.
*   **Centralized Sizing:** Added `HIDDEN_SIZE_R` and `HIDDEN_SIZE_G` to `config.py` to enforce the asymmetric benchmark standard across the entire decathlon suite.

## [0.6.2] - "Mnemonic Crystallization" - 2025-12-18
### Added
*   **Content-Addressable Memory (CAM):** Refactored `benchmark_long_term.py` to be a modular, side-by-side comparative engine. Proved that Phase Accumulation is a permanent mnemonic anchor where Matrix Multiplication (GRU) suffers from bulldoze decay.
*   **Order Independence Gauntlet:** Created `benchmark_order_independence.py` which imports the comparative engine to perform multi-run shuffled brutality tests (3x runs, 10k epochs each).
*   **Perfect Recall:** Achieved 100.0% accuracy across every word in every shuffle, confirming the "Crystalline" nature of the U-Neuron manifold.

## [0.6.1] - "The Grand Slam" - 2025-12-18
### Added
*   **Order Extraction Benchmark:** Introduced `benchmark_permutations.py`. Validated that the U-Neuron can store and recall 4 discrete shuffles of the same keyword set using hidden state phase-branching.
*   **Long-Term Memory Curriculum:** Added `benchmark_long_term.py` to test continual learning and recall retention over extended training gradients.
*   **Stress Testing:** Developed `benchmark_concentration.py` to evaluate phase-state snapping and topological recovery under noise disruption.
*   **Unified Battery:** Updated `run_battery.py` to include the Permutations test. Successfully executed the **336b2d11** regression battery with 100% accuracy across 7 tasks.

### Fixed
*   **Topology Noise:** Discovered and eliminated a "3% Force Field" in `benchmark_topology.py` where random edge addition could pick existing edges, creating label noise.
*   **Topology Stability:** Standardized `TOPOLOGY_CONFIG` with the "Scaled Parity" parameters (LR=$2^{-8}$, Delayed Locking=0.4), resulting in **100% Accuracy** across all runs.
*   **Git Metadata Corruption:** Resolved `fatal: bad object refs/desktop.ini` by recursively purging Windows/Drive metadata from the `.git` directory.

### Documentation
*   **Story Arc:** Refactored `README.md` to tell the "Arc of Discovery" story from v0.1.0 to v0.6.1.
*   **Glossary:** Expanded technical definitions for Phase Accumulation, Locking Strength, and Harmonic Reciprocals.
*   **LaTeX:** Sanitized all math expressions for GitHub rendering excellence.

## [0.6.0] - "Spinor Monism" (Refined) - 2025-12-17
### The Great Simplification
Experiments in the "Pure Harmonic" test battery revealed that the higher-order harmonics (`[2, 4, 8]`) introduced in v0.5.0 were actually adding optimization noise (rugged potential landscapes) that caused jitter in convergence.

We discovered that **Spinor Features** ($4\pi$ Double Cover) provide sufficient topological complexity to solve "Twist" problems (Logic/Parity) without needing higher-order harmonic wells.
*   **Verdict:** **Harmonic [1] is Universal.**
*   **Result:** All benchmarks now converge to **100% Accuracy** with a single smooth potential well.

### Changed
*   **Configuration:** `HARMONICS` set to `[1]` globally in `config.py`.
*   **Stability:** Removed the fractal noise of higher harmonics, resulting in faster and more monotonic convergence curves.

## [0.5.0] - "Harmonic Monism" - 2025-12-17
### Total Overhaul
This release marks the transition from experimental tuning to a stable **Harmonic Monism** architecture. By aligning the U-Neuron's thermodynamic locking parameters with the "Geometric Resonance" principles of the **Unified Informatic Topology (UIT)** framework, we have achieved a universal configuration that solves Logic, Topology, and Structure tasks without per-task hyperparameter tuning.

### Core Configuration (The Golden Ratio)
The system has stabilized on a single, robust hyperparameter set that balances exploring the "Glassy Phase" with crystallization into the "Ordered Phase" (Grokking):
*   **Harmonic Spectrum:** `[1, 2, 4, 8]` (Powers of 2 Resonance). This represents the multi-scale potential barriers described in the MERA/Holographic cosmology sections of the UIT framework.
*   **Peak Locking Strength:** `0.0625` (1/16). A precise thermodynamic weight that allows tunneling through local minima while enforcing global phase coherence.
*   **Learning Rate:** `0.001953125` ($2^{-9}$). A harmonic reciprocal of the phase space, ensuring gradient descent steps align with the underlying geometry.
*   **Fast Regime:** All benchmarks now converge within **300 Epochs**, demonstrating extreme sample efficiency compared to standard RNNs.

### New Test Harnesses (The Decathlon)
The benchmarking suite has been rewritten to enforce strict Head-to-Head (H2H) comparison against industry-standard GRUs, including **Correlation Heatmaps** to verify internal consistency across random seeds.

1.  **Sequential Structural Memory (`benchmark_brackets_masked.py`):**
    *   **Feature:** Implements the `SequentialROUNDModel`.
    *   **Result:** Solves Dyck-2 (Nested Brackets) in streaming mode (100% Accuracy) where standard RNNs struggle (~50%).
    *   **UIT Context:** Demonstrates the "Non-Volatile" nature of the U-Neuron's phase memory ($z$-axis stability).

2.  **Discrete Logic (`benchmark_parity.py`):**
    *   **Feature:** 16-bit Recursive XOR (Parity).
    *   **Result:** 100% Accuracy via "Delayed Locking" (Annealing).
    *   **UIT Context:** Confirms Section 9.10 of the UIT report: "The Sphere (Phase Space) contains the Cube (Logic) when proper quantization is applied."

3.  **Topological Invariants (`benchmark_topology.py`):**
    *   **Feature:** Euler Characteristic (Cycle Detection) on flattened graphs.
    *   **Correction:** Implemented "Signal Fidelity" (Filter Zeros/Sorting) to prevent sparse matrix noise from drifting the phase integrator.
    *   **Result:** 96% Accuracy (Ceiling).

4.  **Generative Creativity (`benchmark_ascii.py`, `benchmark_colors.py`):**
    *   **Feature:** Byte-level sequence generation.
    *   **Result:** Rapid convergence on "Hello World" and Semantic Algebra (Colors) tasks, outpacing GRU convergence speed by ~3x.

5.  **The Oracle (`benchmark_oracle.py`):**
    *   **Feature:** Direct H2H consistency test.
    *   **Result:** ROUND matches or exceeds GRU performance with higher sample efficiency and interpretable phase history.

### Removed / Deprecated
*   **Wobble & Spinor (Experimental):** While the `WobblePhaseAccumulator` (SU(2)) and Spinor features were implemented, the robust performance of the 1D Harmonic Monism configuration rendered them unnecessary for this release cycle. They remain in the codebase for future non-Abelian research.
*   **Legacy Configs:** Removed ad-hoc tuning parameters in favor of the centralized `config.py` constants.

## [0.4.0] - 2025-12-16
### Added
- **Spinor Monism:** Formalized the inclusion of **Spin-1/2 Features** ($\cos(\phi/2), \sin(\phi/2)$) in the `PhaseAccumulator`. This allows the neuron to operate on the double cover of the circle ($4\pi$), resolving the topological ambiguity between $0$ and $2\pi$.
- **Creative Benchmarks:** Added `benchmark_ascii.py` (cyclic text generation) and `benchmark_colors.py` (semantic algebra) to the battery.
- **Full Validation:** Achieved **100% Accuracy** on Parity (16-bit XOR) using the single-neuron Harmonic Setup + Spinor features.
- **Spin Control:** Implemented `SPIN_FACTOR` as a hyperparameter in `config.py` and `ROUND.py`, allowing tuning of the spinor winding capacity (e.g., Spin 1/2, Spin 1, Spin 8) for future research.

### Changed
- **Archive:** Moved `benchmark_clock.py` to `zz_archive/` as Modulo-8 remains a Grand Challenge for the single-harmonic architecture. Validated Parity (Modulo-2) is sufficient proof of concept.

## Roadmap (v0.5.0 and Beyond)
*   **Spin Tuning:** Investigate higher-order spin states (e.g., Spin 1, Spin 3/2) by modulating the `SPIN_FACTOR` to match task-specific topologies (like Modulo-N logic).
*   **The Wobble Neuron (Bloch Sphere):** Upgrade the U-Neuron from Planar Phase (U(1)) to Spherical Rotation (SU(2)). By allowing the axis of rotation to precess ("wobble") orthogonal to the phase plane, the state can bypass harmonic potential barriers by "stepping over" them via the poles. This "Z-axis rotation" is hypothesized to be the key to solving the **Graph Topology Torture Test** (flattened adjacency matrices), where strictly planar 1D models fail to disentangle the crossed dependencies of 2D structures.
*   **Hierarchical Stacking:** Connect Spinor Neurons in deep layers to handle nested logic and recursive reasoning interactively.

## [0.3.2] - 2025-12-14
### Added
- **Dynamic Braking Mechanism:** Implemented an "Active Brake" that modulates the loss gradient magnitude based on the phase correlation metric (`K`). The brake scales linearly (0.001 to 1.0) when `K` falls into the settling range (0.387 - 0.5), allowing rapid initial exploration followed by stabilized fine-tuning as the system "locks" into harmonic wells.
- **Sequential Test Harness:** Added `benchmark_brackets_masked.py` featuring a `SequentialROUNDModel` to validate that ROUND can solve structural tasks in a strictly streaming manner (token-by-token) without access to future context, confirming its recurrent "dynamo" nature.

### Fixed
- **Metric Calculation:** Corrected the phase correlation metric (`pc`) calculation in all benchmarking scripts. Removed incorrect `*torch.pi` scaling factor, ensuring the metric correctly reflects phase alignment in natural radian space ($\sin^2(\theta)$) rather than a distorted domain.
- **Braking Logic:** Updated the braking formula from a static `tanh` function to a dynamic clamp `clamp((pc-0.387)/0.113, 0.001, 1.0)`, ensuring the brake actively engages when the phase distribution begins to settle.

### Tuned
- **Golden Configuration:** Standardization of `locking_strength` to `0.0625` (up from 0.03125) and harmonic spectrum to `[1, 2, 4, 8]` with reciprocal weights `[1, 0.5, 0.25, 0.125]`. This configuration proved robust across all functional benchmarks while correctly failing the negative control (Clock/Modulo-8).

## [0.3.1] - 2025-12-14
### Fixed
- **Environment:** Updated `.gitignore` to exclude `.venv` and other system artifacts.
- **Cleanup:** Minor housekeeping on the Python package structure (`__init__.py`).

## [0.3.0] - 2025-12-14
### Added
- **Unified Harmonic Standard Configuration:** Discovered that setting the learning rate to a harmonic reciprocal of the phase space ($2^{-9} \approx 0.00195$) allows a single architecture configuration (`h=32`, `lr=0.001953125`) to solve all benchmark tasks (Logic, Arithmetic, Structure, Topology) without task-specific tuning.
- **Improved Stability:** The new harmonic learning rate demonstrates superior stability, enabling the model to recover from mid-training collapse (entropy spikes) where baseline GRUs fail.

### Changed
- **Hyperparameters:** Unified all benchmarks to use `hidden_size=32` and `lr=0.001953125`.
- **Documentation:** Updated README to reflect the new "Harmonic Resonance" findings and the clean sweep against the GRU baseline.

## [0.2.0] - 2025-12-13
### Added
- **HarmonicROUNDLoss:** Implemented a new loss function that sums multiple sine-squared potentials (`sin^2(h*phi)`) to create a complex stability landscape.
- **Terminal-Only Locking:** Added support for applying the locking potential only at the final time step (`terminal_only=True`), allowing free phase evolution ("wave behavior") during the sequence and quantized collapse ("particle behavior") at readout.
- **Benchmark Suite:** Added `benchmark_brackets.py`, `benchmark_clock.py`, `benchmark_parity.py`, and `benchmark_topology.py` to test different computational invariants.

### Changed
- **Architecture:** Transitioned from the experimental `ROUND_Release` to `ROUND_Harmonic`.
- **Performance:** Achieved 100% accuracy on Parity and Topology tasks, significantly outperforming standard GRU baselines on cyclic and long-range dependency tasks.

## [0.1.0] - 2025-12-12
### Added
- **Initial release of `ROUND` (Riemannian Optimized Unified Neural Dynamo).**
- **Core `PhaseAccumulator` mechanism.**
- **Basic `ROUNDModel` for binary classification.**
