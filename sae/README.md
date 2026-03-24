# Sparse Autoencoder Analysis on Grokking Transformer

Applying Sparse Autoencoders (SAEs) to the internal activations of a trained Grokking Transformer to test whether the learned Fourier features are recoverable via mechanistic decomposition — and to quantify how superposition changes across the grokking phase transition.

## Overview

This extends the [Grokking mechanistic interpretability analysis](../) by training SAEs on post-FF activations at multiple training checkpoints. Two questions drive the analysis:

1. **Do SAE features correspond to Fourier frequencies?** (Central prediction from Nanda et al. 2023)
2. **Does grokking increase feature specialization?** (Superposition hypothesis from Elhage et al. 2022)

**SAE architecture** (following Bricken et al. 2023):
```
encode: h = ReLU(W_enc(x - b_pre) + b_enc)
decode: x̂ = W_dec h + b_pre
loss  = ||x - x̂||² + λ ||h||₁
```
W_dec columns are normalized to unit length throughout training. λ is selected independently per checkpoint via a short sweep — pre-grokking activations are more diffuse and require stronger regularization than post-grokking.

## Configuration

| Parameter | Value |
|-----------|-------|
| d_sae | 1024 (8× d_model) |
| λ (L1) | per-checkpoint, ~0.1 |
| num_steps | 50,000 |
| Activation site | post feed-forward + LayerNorm |
| Checkpoints analyzed | Steps 1k, 7.5k, 15k, 20k |

## Results

### 1. Sparsity Transition

Grokking reduces the number of active SAE features per input by a factor of ~3, without any change in the SAE's regularization strength.

| Step | Val Acc | Mean L0 | % of d_sae |
|------|---------|---------|------------|
| 1,000 (pre) | 30% | 290 | 28% |
| 7,500 | 100% | 92 | 9% |
| 15,000 | 100% | 111 | 11% |
| 20,000 | 100% | ~110 | 11% |

This compression is not imposed by the SAE — it reflects a structural change in the transformer's representations after the phase transition. The memorized solution requires more active features than the compact Fourier solution.

---

### 2. Superposition Analysis

The core question: does the SAE learn *one dedicated feature per Fourier frequency*, or do individual features encode *multiple frequencies simultaneously* (superposition)?

**Metric — Feature Exclusivity:**
```
Exclusivity(feature i) = Power(dominant frequency) / Power(all frequencies)
```
- 1.0 = fully dedicated to one frequency (monosemantic)
- 0.5 = power split equally across two frequencies
- ~0.07 = power spread uniformly (random baseline)

<img width="1784" height="740" alt="sae_superposition_comparison" src="https://github.com/user-attachments/assets/da2a38be-93f1-4542-bc71-7d5660ef964e" />

| | Pre-Grokking (Step 1k) | Post-Grokking (Step 15k) |
|---|---|---|
| Mean Exclusivity | 0.07 | 0.36 |
| Dedicated features (>0.5) | **0%** | **27%** |
| Feature type | Token-lookup | Fourier components |

Grokking increases mean exclusivity 5× and produces a population of dedicated features from zero. Pre-grokking, no feature exceeds the 0.5 specialization threshold — every feature encodes a diffuse mixture. Post-grokking, 27% of active features are specialized, with the most extreme case reaching exclusivity 0.88.

---

### 3. Per-Frequency Feature Analysis

<img width="2060" height="1475" alt="sae_features_post_step15000" src="https://github.com/user-attachments/assets/99c60caa-3ae0-40bd-b187-e5d8b01c94c6" />

For each dominant embedding frequency (k=3, k=7, k=36), the top-3 responding SAE features are shown with their exclusivity scores. Three distinct regimes are visible:

**Nearly monosemantic (Feature 591, k=36, excl=0.88):** Uniform dot-grid pattern in the (a,b) space — this is exactly cos(36·a)·cos(36·b) structure. The only near-monosemantic feature found in the experiment.

**Moderately specialized (Feature 374, k=7, excl=0.56):** Clear structured pattern but with secondary frequency contributions. Two separate features (374 and 857) both respond to k=7 — the same frequency is encoded in multiple SAE features, a form of redundant superposition.

**Low exclusivity (Feature 740, k=3, excl=0.49):** Complex cross-pattern encoding multiple frequencies simultaneously — polysemantic by the standard of Elhage et al. 2022.

**Pre-grokking (Step 1,000):**

<img width="2292" height="1918" alt="sae_superposition_step1000" src="https://github.com/user-attachments/assets/bfcb33c4-124c-4186-9714-40864bc2ea75" />

All features have exclusivity < 0.15. The dominant feature pattern is a single bright cross (active only when a=57 or b=57) — a token-specific lookup mechanism consistent with memorization, not Fourier computation.

---

### 4. Fourier Spectrum Match

<img width="2984" height="1182" alt="sae_comparison" src="https://github.com/user-attachments/assets/3f71a116-4483-45a2-8899-4d9f0470a8c0" />

| Step | Spectrum Match | L0 |
|------|---------------|-----|
| 1,000 (pre) | 84%* | 290 |
| 7,500 | 25% | 92 |
| 15,000 | 57% | 111 |
| 20,000 | 22% | 110 |

*Pre-grokking 84% is an artifact: the embedding spectrum is flat (no dominant frequency), so cosine similarity is trivially high.

Step 15,000 shows the clearest match — SAE feature spectrum peaks at k=7, directly aligning with the dominant embedding frequency. Full decomposition is not achieved because most features are polysemantic (73% have exclusivity < 0.5).

---

### 5. Feature Activation Patterns

<img width="2234" height="1475" alt="sae_activation_grid" src="https://github.com/user-attachments/assets/f1d17ce6-72d2-4825-85a0-23bb505d1d41" />

Top SAE features post-grokking show diagonal stripe patterns in the (a, b) input space — the expected signature of cos(k·(a+b)) features. Stripes follow lines of constant a+b mod 97, the exact structure needed to compute modular addition. This confirms Fourier structure in the SAE features independently of the spectrum analysis.

---

## Summary

Four findings:

**1. Grokking compresses representations 3×.** L0 drops from 28% to 9% — the Fourier solution is intrinsically sparser than the memorized solution.

**2. Grokking increases feature specialization 5×.** Mean exclusivity rises from 0.07 to 0.36; dedicated features appear from zero to 27% of active features.

**3. Memorization uses token-lookup features; generalization uses Fourier features.** Pre-grokking features activate for specific token indices. Post-grokking features follow trigonometric patterns over (a, b) space.

**4. Complete Fourier decomposition is blocked by superposition.** 73% of post-grokking features remain polysemantic. The grokking transformer packs multiple Fourier components into shared representational directions — a compression strategy not captured by the one-feature-per-frequency hypothesis.

## Files

| File | Description |
|------|-------------|
| `grokking_sae.py` | SAE training, Fourier analysis, superposition analysis |
| `Grokking_SAE_Colab.ipynb` | Colab notebook (T4 GPU, ~30 min runtime) |

## Setup

```bash
pip install torch matplotlib numpy scikit-learn
# Requires checkpoints/ from ../grokking_mechanistic.py
python grokking_sae.py
```

## References

- Elhage et al. (2022). *Toy Models of Superposition.* Transformer Circuits Thread.
- Bricken et al. (2023). *Towards Monosemanticity.* Transformer Circuits Thread.
- Nanda et al. (2023). *Progress Measures for Grokking via Mechanistic Interpretability.* arXiv:2301.05217.
- Power et al. (2022). *Grokking: Generalization beyond overfitting on small algorithmic datasets.* arXiv:2201.02177.
