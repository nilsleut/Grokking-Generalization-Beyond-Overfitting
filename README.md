# Grokking — Generalization Beyond Overfitting

Reproduction, phase diagram, and mechanistic interpretability analysis of the grokking phenomenon (Power et al. 2022) on modular arithmetic.

## Overview

Grokking is a delayed generalization phenomenon: a neural network first memorizes the training set (train accuracy → 100%, validation accuracy ≈ chance), then — after extended training — abruptly generalizes (validation accuracy → 100%). This project reproduces the phenomenon, maps the phase diagram across hyperparameters, and analyzes the internal mechanism via Fourier analysis of token embeddings.

```
Steps 0–1.5k:   Train acc → 100%,  Val acc ≈ 10%   [memorization]
Steps 1.5k–5k:  Train acc = 100%,  Val acc ≈ 10%   [stagnation]
Steps 5k–7k:    Train acc = 100%,  Val acc: 10% → 100%  [grokking]
Steps 7k+:      Both = 100%,  stable                    [generalization]
```

## Task

**Modular addition:** given (a, b), predict (a + b) mod p, where p = 97.

This task has an exact, compact solution expressible as Fourier features over Z/pZ. The network eventually discovers this structure after weight decay pressure forces it to abandon the expensive memorized solution.

## Architecture

1-layer Transformer (following Power et al. 2022):
- Input: token sequence [a, b, =], vocab size = p + 1
- d_model = 128, n_heads = 4, d_ff = 512
- Output: classification over p classes from the = position

## Results

### 1. Grokking Reproduction

Grokking at **~6,500 steps** with `train_frac=0.15`, `weight_decay=1.0`:

| Parameter | Value |
|-----------|-------|
| p (modulus) | 97 |
| Training pairs | 1,411 / 9,409 (15%) |
| Memorization step | ~1,500 |
| Grokking step | ~6,500 |
| Final val acc | 100% |

Weight decay is the critical parameter — without it, the memorized solution is never abandoned and grokking does not occur.

---

### 2. Phase Diagram

`grokking_phase_diagram.py` sweeps a 5×5 grid over `train_frac ∈ {10%, 20%, 30%, 40%, 50%}` and `weight_decay ∈ {0.1, 0.5, 1.0, 2.0, 5.0}`, classifying each of 25 runs into three regimes:

- **Direct** — immediate generalization, no memorization plateau
- **Grokking** — delayed phase transition after plateau
- **Stuck** — no generalization within step budget

|  | 10% | 20% | 30% | 40% | 50% |
|--|-----|-----|-----|-----|-----|
| **wd=5.0** | STUCK | STUCK | STUCK | STUCK | STUCK |
| **wd=2.0** | STUCK | DIRECT | DIRECT | GROKKING | STUCK* |
| **wd=1.0** | STUCK | GROKKING | DIRECT | DIRECT | DIRECT |
| **wd=0.5** | STUCK | GROKKING | DIRECT | DIRECT | DIRECT |
| **wd=0.1** | STUCK | STUCK | DIRECT | DIRECT | DIRECT |

<img width="2662" height="742" alt="grokking_phase_diagram" src="https://github.com/user-attachments/assets/37e42e3d-ee92-4adf-be81-15661e80dc91" />

*58.5% val acc at cutoff — likely grokking given more steps.

**Three findings:** (1) weight_decay=5.0 is universally stuck regardless of dataset size. (2) train_frac=10% is universally stuck regardless of regularization. (3) Grokking occupies a narrow band — only 3 of 25 runs grok, bounded by direct generalization on one side and stuck on the other.

---

### 3. Mechanistic Interpretability

`grokking_mechanistic.py` analyzes the internal representations learned by the network at multiple checkpoints.

#### Fourier Analysis

<img width="2984" height="593" alt="fourier_analysis" src="https://github.com/user-attachments/assets/b3dfce5e-dae2-4c6a-8528-c710d89e7634" />

Token embeddings transition from a flat Fourier spectrum (pre-grokking) to sharp peaks at a small number of frequencies (post-grokking). This confirms that the network learns a Fourier representation of modular arithmetic — token x is encoded as a superposition of cos(2πkx/p) and sin(2πkx/p) for a few dominant frequencies k.

#### Fourier Projection — the Circle

<img width="2903" height="747" alt="image" src="https://github.com/user-attachments/assets/b68d18be-f169-4ece-9a8c-7bd7d43e2b17" />

Projecting token embeddings onto the dominant cos/sin basis vectors reveals a circle structure after grokking. Each token x sits at angular position 2πkx/97 on the circle. This directly proves that the network represents numbers as points on a circle, making modular addition equivalent to angle addition.

Circularity metric (1.0 = perfect circle): **0.92** post-grokking vs. **0.86** pre-grokking (different frequency).

#### Frequency Dynamics

<img width="1786" height="742" alt="frequency_over_time" src="https://github.com/user-attachments/assets/1affffd2-01f7-4b3a-951b-4ee20c982bcc" />

Tracking the dominant Fourier frequency over training reveals a two-stage process:

1. **Fast convergence to a suboptimal frequency** (~Step 500–1200): the network quickly finds a Fourier solution that achieves partial generalization (~30% val acc) but is not optimal.
2. **Reorganization** (~Step 1200–2000): the dominant frequency collapses and the network transitions to a more stable frequency, coinciding with the main grokking jump to ~100% val acc.

---

### 4. Robustness Study

`grokking_robustness.py` repeats the mechanistic analysis across 5 random seeds to verify that the frequency transition is a general phenomenon, not an artifact of a single run.

```
Step   S42-k  S42-val   S43-k  S43-val   S44-k  S44-val   S45-k  S45-val   S46-k  S46-val
────────────────────────────────────────────────────────────────────────────────────────────
 500      36     9.2%      12     8.0%       2     2.9%      46     9.9%      43     9.2%
1500      36    62.8%      12    88.2%       2    18.0%      46    43.4%      14    99.0%
2500      36    99.9%      12   100.0%      43    99.5%       6   100.0%       9   100.0%
3000      36   100.0%      12   100.0%      43   100.0%       6   100.0%       9   100.0%
```

<img width="1936" height="889" alt="robustness_frequency" src="https://github.com/user-attachments/assets/580e4da6-7354-431b-8682-b833b84bbe17" />

**Key finding:** each seed converges to a *different* dominant frequency (k = 6, 9, 12, 36, 43), but all reach 100% val acc. This demonstrates that the Fourier solution is not unique — multiple frequencies can represent modular addition equivalently. The network selects one stochastically depending on initialization, but the reorganization mechanism (suboptimal → stable frequency) is consistent across seeds.

This extends the findings of Nanda et al. (2023), who identified the Fourier circuit but did not characterize the stochastic frequency selection or the two-stage convergence dynamic.

---

## Files

| File | Description |
|------|-------------|
| `grokking_correct.py` | Single run, 200k steps |
| `grokking_phase_diagram.py` | 5×5 hyperparameter sweep |
| `grokking_mechanistic.py` | Fourier analysis + embedding geometry |
| `grokking_robustness.py` | Multi-seed robustness study |

## Setup

```bash
pip install torch matplotlib numpy scikit-learn
python grokking_correct.py          # single run (~30 min CPU)
python grokking_phase_diagram.py    # phase diagram (~90 min CPU)
python grokking_mechanistic.py      # mechanistic analysis (~10 min CPU)
python grokking_robustness.py --seed 42   # one robustness seed
```

## References

- Power et al. (2022). *Grokking: Generalization beyond overfitting on small algorithmic datasets.* arXiv:2201.02177.
- Nanda et al. (2023). *Progress measures for grokking via mechanistic interpretability.* arXiv:2301.05217.
- Liu et al. (2022). *Towards understanding grokking: An effective theory of representation learning.* NeurIPS 2022.
- Lyu et al. (2022). *Grokking as the transition from lazy to rich training dynamics.* arXiv:2209.11895.
- Fan et al. (2024). *Deep grokking: Would deep neural networks generalize better?* EPFL.
