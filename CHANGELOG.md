# Changelog

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
