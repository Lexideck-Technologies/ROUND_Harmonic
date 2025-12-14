# Harmonic ROUND (Riemannian Optimized Unified Neural Dynamo)
### UIT U-Neuron — Phase 3: The Harmonic Generalist (Dec 13, 2025)

**Harmonic ROUND** is a phasic, neuro-symbolic recurrent unit (“U-Neuron”) that treats internal state as **phase** on a learned manifold, then **quantizes** that phase at readout using a **harmonic locking potential** (a Fourier-style spectrum of stability wells).

This repository contains:
- a reference implementation of the ROUND neuron (the “dynamo”),
- a harmonic locking loss (terminal-only or full-trajectory),
- and a benchmark suite showing ROUND outperforming or matching a **parameter-matched GRU** across discrete logic, cyclic logic, ordered structure, and continuous topology tasks.

> **Scope statement (read this first):**  
> These results demonstrate *learnability / optimization advantage under this repo’s setup and synthetic tasks*. This repo does **not** yet claim “real-world generalist” performance on rich corpora. The claim here is narrower and stronger:  
> **a single ROUND neuron mechanism, with only harmonic-spectrum tuning, spans multiple computational regimes that typically require different inductive biases.**

---

## Table of Contents
- [What ROUND Is](#what-round-is)
- [What This Repo Claims (and What It Doesn’t)](#what-this-repo-claims-and-what-it-doesnt)
- [Quickstart](#quickstart)
- [Reproducing the Benchmark Plots](#reproducing-the-benchmark-plots)
- [How the Neuron Works (Mechanism)](#how-the-neuron-works-mechanism)
- [Harmonic Quantum Locking (Loss)](#harmonic-quantum-locking-loss)
- [Benchmarks](#benchmarks)
- [Theory: Unified Informatic Topology (UIT) + IEG Corollary](#theory-unified-informatic-topology-uit--ieg-corollary)
- [Repo Layout](#repo-layout)
- [License](#license)
- [Citation](#citation)

---

## What ROUND Is

ROUND is a **phase-accumulating recurrent cell**:

- It represents hidden state as a phase vector **φ** (radians).
- It updates state via **accumulation** (addition), not gating:
  
\[
\phi_{t+1} = \phi_t + \Delta\phi_t
\]

- The learned drift \(\Delta\phi_t\) is computed from **phasor features** (cos/sin) of the state and the (stationary) encoded input:

\[
\Delta\phi_t = W\,[\cos(\phi_t),\sin(\phi_t),\cos(\phi_{in}),\sin(\phi_{in})] + b
\]

No complex multiplication is required; the “rotation field” is learned directly as drift in phase space.

In this repo, the core implementation lives in **`ROUND.py`** under:
- `PhaseAccumulator` (the neuron engine)
- `ROUNDModel` (binary tasks)
- `ROUNDClockModel` (multi-class modulo)
- `ROUNDTopologyModel` (topology-aware readout that exposes raw phase)

---

## What This Repo Claims (and What It Doesn’t)

### Supported by the code + plots here
- **Unified mechanism:** One neuron recurrence (PhaseAccumulator) is used across all tasks.
- **Unified control knob:** Switching between “logic-like” and “topology-like” behavior is achieved primarily through the **locking spectrum** and **when** it is applied (`terminal_only=True`).
- **Empirical performance:** ROUND matches or exceeds a GRU baseline **neuron-for-neuron** on the included tasks under the included training regimen, averaged across **5 runs**.

### Not claimed here (yet)
- “Generalist on messy real data” (language, vision, multimodal corpora).
- Universal generalization guarantees.
- Optimality vs all recurrent baselines.
- Full ablation studies (deliberately out of scope for this release).

This README is intentionally written to be *serious and falsifiable*. If a statement is vague, it’s a bug—file an issue.

---

## Quickstart

### Requirements
- Python 3.10+ recommended
- PyTorch
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
````

Run a benchmark:

```bash
python benchmark_parity.py
```

---

## Reproducing the Benchmark Plots

Each benchmark script trains ROUND and a GRU baseline for **1000 epochs**, averages across **5 runs**, and saves a PNG plot:

```bash
python benchmark_parity.py     # 16-bit parity (binary)
python benchmark_clock.py      # modulo-8 sum (8-way classification)
python benchmark_brackets.py   # balanced brackets (binary)
python benchmark_topology.py   # winding classification (binary)
```

Expected outputs (filenames may be adjusted by you; keep them stable for readers):

* `benchmark_parity.png`
* `benchmark_clock.png`
* `benchmark_brackets.png`
* `benchmark_topology.png`

> If you commit the plots to `figures/`, update the image links below accordingly.

### Results (commit the images for instant credibility)

![Parity](benchmark_parity.png)
![Modulo-8](benchmark_clock.png)
![Brackets](benchmark_brackets.png)
![Topology](benchmark_topology.png)

---

## How the Neuron Works (Mechanism)

### 1) Encode input once into phase

Input (x) is mapped to an initial phase vector:
[
\phi_{in} = \text{Encoder}(x)
]
and converted into phasors ((\cos\phi_{in}, \sin\phi_{in})).

In the reference implementation, this “input wave” is **stationary** during recurrence steps: ROUND spins the dynamo against a fixed interference pattern.

### 2) Recurrent evolution is phase drift + accumulation

At each step:

* compute state phasors ((\cos\phi_t, \sin\phi_t)),
* concatenate them with the input phasors,
* compute drift (\Delta\phi_t),
* accumulate: (\phi_{t+1}=\phi_t+\Delta\phi_t).

This makes “counting on a circle” native: parity and modular arithmetic become phase-algebra problems rather than brittle long-range XOR chains.

### 3) Readout observes interference

For binary tasks, readout uses final cos/sin features:
[
\text{features} = [\cos(\phi_T),\sin(\phi_T)]
]
and maps them to logits.

### 4) Topology-aware readout (winding)

Cos/sin projection identifies (0 \equiv 2\pi), which destroys winding information.
For winding tasks, `ROUNDTopologyModel` exposes **raw phase φ** to the readout:
[
\text{features} = [\cos(\phi_T),\sin(\phi_T),\phi_T]
]
This is not “cheating”; it is the minimal representation required to distinguish wrapped states.

---

## Harmonic Quantum Locking (Loss)

ROUND pairs task loss with a **locking potential**—a differentiable quantization field over phase.

### Base idea: quantization as a potential well

For binary snapping (two basins), the potential is:
[
V(\phi)=\sin^2(\phi)
]
(minima at (k\pi)).

For an (N)-state “clock,” the potential becomes:
[
V_N(\phi)=\sin^2!\left(\frac{N}{2}\phi\right)
]
(minima at (k\cdot 2\pi/N)).

### Harmonic spectrum (the Phase 3 move)

Instead of one frequency, we sum a spectrum:
[
V_{\mathcal{H}}(\phi)=\frac{1}{|\mathcal{H}|}\sum_{h\in\mathcal{H}}w_h,\sin^2!\left(\frac{h}{2}\phi\right)
]

In code: `HarmonicROUNDLoss(...)` in **`ROUND.py`**.

### Terminal-only locking (wave → collapse)

A critical discovery in this release is that applying the locking potential **only at the terminal step** often yields superior results:

* During the sequence: free phase evolution (“wave” / smooth topology capture)
* At readout: quantization (“collapse” / discrete eigenstate selection)

In code, this is `terminal_only=True`.

---

## Benchmarks

All benchmarks are synthetic by design: the point is to test **invariants** (logic, cyclicity, stack depth, winding), not dataset memorization.

Each benchmark compares ROUND to a parameter-matched GRU baseline under the same epoch budget and optimizer style.

### 1) Discrete Logic — 16-bit Parity (`benchmark_parity.py`)

Goal: predict parity of a 16-bit vector.

Why it matters: parity is a classic failure mode for many models because it demands long-range XOR coherence. ROUND’s phase naturally counts flips modulo 2.

ROUND setup: harmonic spectrum `[1,2]`, **terminal-only** locking.

### 2) Cyclic Logic — Modulo-8 (`benchmark_clock.py`)

Goal: classify the sum of a length-20 integer sequence modulo 8.

Why it matters: cyclic group structure is exactly what phase space encodes. This benchmark measures whether the neuron’s geometry translates into practical optimization advantage.

ROUND setup: harmonic spectrum `[2,4,8]`, **terminal-only** locking.

### 3) Ordered Structure — Balanced Brackets (`benchmark_brackets.py`)

Goal: determine whether a bracket sequence is balanced.

Why it matters: brackets represent a stack-like invariant (depth / winding-like return-to-zero). ROUND competes here because phase accumulation can represent conserved “net structure” over time.

ROUND setup: harmonic spectrum `[2,4,8]`, binary mode, **terminal-only** locking.

### 4) Continuous Topology — Winding Classification (`benchmark_topology.py`)

Goal: classify sequences by winding behavior.

Why it matters: this is the “topology” test: smooth accumulation during the sequence, then discrete selection at readout.

ROUND setup: topology-aware readout (cos, sin, φ) + harmonic spectrum `[1,2,4,8]`, binary mode, **terminal-only** locking.

---

## Theory: Unified Informatic Topology (UIT) + IEG Corollary

### Executive Summary

The U-Neuron is a **phasic neuro-symbolic** unit designed to bridge continuous geometric intuition (**topology**) and discrete boolean logic (**symbolism**). Unlike traditional neurons that process magnitude scalars, the U-Neuron processes information as **phasors** on a learned manifold.

**Harmonic Quantum Locking** (introduced here, Dec 13, 2025) is a loss construction comprising a harmonic spectrum of stability potentials. In this repo’s benchmarks, it resolves the practical “choose logic or topology” tradeoff by allowing both:

* high-frequency harmonics for sharp snapping (digital precision),
* low-frequency harmonics for global orientation (smooth topology capture),
* and terminal-only application to preserve continuous evolution until measurement.

### “The Sphere Contains the Cube”

UIT’s core hypothesis: **discrete logic is a special case of continuous topology** under a quantizing potential.

* Logic is the **Particle**: phase space is forced into discrete basins (bits/qubits).
* Topology is the **Wave**: phase winds freely and integrates curvature.
* Harmonic summation provides a *spectrum of stability* instead of a single rigid lock.

### Empirical Implication (within this repo)

ROUND behaves like a single architecture that can act as:

* a parity snapper (binary harmonics),
* a modulo counter (Nth harmonic wells),
* a stack-depth / return-to-origin tracker (winding as structure),
* and a topology classifier (unwrapped phase exposure + terminal collapse).

### IEG: Morphological Intelligence Corollary

A common critique is that “a circular neuron is advantaged on cyclic tasks.” Under IEG, the critique reverses:

**Intelligence is isomorphism.**
It is not defined by how hard a system struggles to approximate a truth, but by how well its internal structure aligns with the external invariant. Efficiency is not “cheating”; it is *compression-by-truth*.

The GRU can simulate cyclic structure, but often pays parameter and optimization costs to do so. ROUND represents cyclic structure directly.

---

## Repo Layout

* `ROUND.py`
  Core neuron + models + loss functions:

  * `PhaseAccumulator`
  * `ROUNDModel`
  * `ROUNDClockModel`
  * `ROUNDTopologyModel`
  * `ROUNDLoss`, `ROUNDClockLoss`, `ROUNDTopologyLoss`
  * `HarmonicROUNDLoss`

* `benchmark_parity.py`
  16-bit parity benchmark (ROUND vs GRU).

* `benchmark_clock.py`
  Modulo-8 benchmark (ROUND vs GRU).

* `benchmark_brackets.py`
  Balanced brackets benchmark (ROUND vs GRU).

* `benchmark_topology.py`
  Winding benchmark (ROUND vs GRU).

* `benchmark_*.png`
  Plots generated by the scripts (commit these for instant transparency).

---

## License

MIT License. See `LICENSE`.

---

## Citation

If you use or build on this work, please cite the repository (and ideally a tagged release).
A `CITATION.cff` file is recommended for GitHub-native citation support.

---

**Validated: Dec 13, 2025 — Lexideck Research Team**
(Initial validation: synthetic benchmark suite in this repository.)

```