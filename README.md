# Grokking — Generalization Beyond Overfitting

Reproduction and phase diagram analysis of the grokking phenomenon (Power et al. 2022) on modular arithmetic.

## Overview

Grokking is a delayed generalization phenomenon: a neural network first memorizes the training set (train accuracy → 100%, validation accuracy ≈ chance), then — after extended training — abruptly generalizes (validation accuracy → 100%). This project reproduces the phenomenon and maps the phase diagram across dataset size and regularization strength.

## What is Grokking?

Standard intuition says: once a model overfits, it will not generalize without architectural changes or more data. Grokking disproves this. Given enough training steps and sufficient weight decay, an overfitted model undergoes a phase transition to a compact, generalizing solution.

```
Steps 0–1.5k:   Train acc → 100%,  Val acc ≈ 10%   [memorization]
Steps 1.5k–5k:  Train acc = 100%,  Val acc ≈ 10%   [stagnation]
Steps 5k–7k:    Train acc = 100%,  Val acc: 10% → 100%  [grokking]
Steps 7k+:      Both = 100%,  stable              [generalization]
```

## Task

**Modular addition:** given (a, b), predict (a + b) mod p, where p = 97.

This task has an exact, compact solution (Fourier features over Z/pZ), which the network eventually discovers after discarding the memorized solution under weight decay pressure.

## Architecture

1-layer Transformer (following Power et al. 2022):
- Input: token sequence [a, b, =], vocab size = p + 1
- d_model = 128, n_heads = 4, d_ff = 512
- Output: classification over p classes from the = position

## Key Result

Grokking at **~6,500 steps** with `train_frac=0.15`, `weight_decay=1.0`:

| Parameter | Value |
|-----------|-------|
| p (modulus) | 97 |
| Training pairs | 1,411 / 9,409 (15%) |
| Memorization step | ~1,500 |
| Grokking step | ~6,500 |
| Final val acc | 100% |

Weight decay is the critical parameter — it creates pressure toward the compact Fourier solution and away from the expensive memorized one. Without it, grokking does not occur.

## Phase Diagram

`grokking_phase_diagram.py` sweeps a 5×5 grid over `train_frac ∈ {10%, 20%, 30%, 40%, 50%}` and `weight_decay ∈ {0.1, 0.5, 1.0, 2.0, 5.0}`, classifying each of the 25 runs into three regimes:

- **Direct** — model generalizes immediately without a memorization plateau
- **Grokking** — delayed phase transition after memorization plateau
- **Stuck** — no generalization within the step budget

<img width="2662" height="742" alt="grokking_phase_diagram" src="https://github.com/user-attachments/assets/e3121d95-77f4-4cdc-9e71-2d3945e24f88" />

### Results (15,000 steps per run)

|  | 10% | 20% | 30% | 40% | 50% |
|--|-----|-----|-----|-----|-----|
| **wd=5.0** | STUCK | STUCK | STUCK | STUCK | STUCK |
| **wd=2.0** | STUCK | DIRECT | DIRECT | GROKKING | STUCK* |
| **wd=1.0** | STUCK | GROKKING | DIRECT | DIRECT | DIRECT |
| **wd=0.5** | STUCK | GROKKING | DIRECT | DIRECT | DIRECT |
| **wd=0.1** | STUCK | STUCK | DIRECT | DIRECT | DIRECT |

*Run 24 (50%, wd=2.0): 58.5% val acc — likely grokking given more steps.

### Three findings

**1. Weight decay = 5.0 is universally stuck.** Regularization is too strong to learn anything — weights collapse toward zero before any solution forms, regardless of dataset size.

**2. Train fraction = 10% is universally stuck.** With only 941 training pairs, the network lacks sufficient signal to discover the underlying Fourier structure, regardless of regularization.

**3. Grokking occupies a narrow band.** Only 3 of 25 runs grok. The regime is bounded by direct generalization (large dataset + moderate weight decay) on one side and stuck (small dataset or extreme weight decay) on the other. Grokking requires the memorized solution to be stable enough to form, but costly enough that weight decay eventually forces a transition to the compact solution.

## Setup

```bash
pip install torch matplotlib numpy
python grokking_correct.py        # single run (~30 min CPU)
python grokking_phase_diagram.py  # 5×5 phase diagram (~90 min CPU)
```

## References

- Power et al. (2022). *Grokking: Generalization beyond overfitting on small algorithmic datasets.* arXiv:2201.02177.
- Nanda et al. (2023). *Progress measures for grokking via mechanistic interpretability.* arXiv:2301.05217.
- Liu et al. (2022). *Towards understanding grokking: An effective theory of representation learning.* NeurIPS 2022.
- Lyu et al. (2022). *Grokking as the transition from lazy to rich training dynamics.* arXiv:2209.11895.
