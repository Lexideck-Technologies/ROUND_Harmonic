# Changelog

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
- Initial release of `ROUND` (Riemannian Optimized Unified Neural Dynamo).
- Core `PhaseAccumulator` mechanism.
- Basic `ROUNDModel` for binary classification.
