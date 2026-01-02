# Concept: The Wobble Neuron (Spin-Axis Precession)
**Date:** 2025-12-16
**Status:** Theoretical (v0.5.0 Proposal)

## The Problem: Topological Locking
In v0.4.0 (Spinor Monism), the neuron operates phase $\phi$ in a fixed 2D plane (Unit Circle $S^1$ or Double Cover).
The "Harmonic Potential" creates barriers in this 1D phase space (like hills in a ring).
To move from one "Well" (State A) to another "Well" (State B), the system must "tunnel" through the potential barrier (the hill).
*   **Result:** In Modulo-8, the barrier between 0 and 1 is too high relative to the driving force, leading to **Mode Collapse** (getting stuck in the 0 basin).

## The Hypothesis: Orthogonal Wobble (Swivel Axis)
What if the neuron was not confined to the plane? What if the axis of rotation could "swivel" or "precess"?
This introduces an extra degree of freedom (Latitude $\theta$ on the Sphere $S^2$).

### Mechanism
1.  **State:** The neuron state is a point on the **Bloch Sphere** (or Quaternion Hypersphere $S^3$).
    *   $\psi = [ \cos(\theta/2), e^{i\phi}\sin(\theta/2) ]^T$ (Spinor)
    *   Or simply a 3-Vector $\vec{v} = (x, y, z)$.
2.  **Harmonic Potential:** The Locking Potential applies primarily to the **Equator** (or specific poles).
    *   $L \propto (1 - \vec{v} \cdot \vec{Target})^2$?
    *   Or $L \propto \sin^2(k \phi)$ (Phase Lock).
3.  **The Wobble:**
    *   To cross a barrier on the Equator ($\phi$), the state can "tilt" up towards the North Pole ($z > 0$).
    *   Near the pole, the longitudinal phase barrier $\sin^2(k\phi)$ vanishes (or the radius of the circle shrinks to 0).
    *   The state can "wobble" over the barrier and settle back down into the next well.
    *   **Analogy:** Stepping *over* a fence instead of walking through it.

## Implementation: 3D Rotation (SO(3))
Instead of `PhaseAccumulator` (Scalar addition), we use `RotationAccumulator` (Matrix Multiplication or Quaternion Multiplication).
*   **Input:** Generates a Torque/Rotation Vector $\vec{u}$.
*   **Update:** $\vec{v}_{t+1} = R(\vec{u}) \vec{v}_t$.
*   **Locking:** A "Gravity" term pulls the vector back to the Equator (Z=0) and to integer phases.

## Potential Impact
*   **Modulo-8 Solved:** The swarm could simply "tilt" out of the 0-basin, rotate, and drop into the 1-basin, bypassing the harmonic barrier entirely.
*   **Resilience:** "Wobble" allows the system to maintain "Momentum" in the orthogonal dimension.

## Next Steps
This requires a fundamental rewrite of the Core Neuron from `U(1)` (Phase) to `SU(2)` (Spin) mechanics. This is the definition of **v0.5.0**.
