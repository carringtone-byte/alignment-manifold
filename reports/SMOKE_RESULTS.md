# OLMo 2 1B SFT→DPO Manifold Smoke Test

**Status:** Exploratory smoke test completed 2026-07-11  
**Hardware:** NVIDIA GeForce RTX 2060, 6 GB VRAM  
**Classification:** `causally_concentrated_subspace_partial_delta`

## Question

Do matched final-prompt-token residual changes from OLMo 2 1B SFT to DPO form
a compact nonlinear manifold, a global linear subspace, or neither? Are compressed
changes causally sufficient when inserted into SFT and necessary when removed from
DPO?

## Immutable checkpoints

- SFT: `allenai/OLMo-2-0425-1B-SFT`
  (`0d85a3d037876ce6ac7d4311d994400fc66ac27f`)
- DPO: `allenai/OLMo-2-0425-1B-DPO`
  (`c4b0485961ab24c2433b090f3b922f0913a9290f`)

The full token streams and example order were bit-identical across checkpoint
extractions. The semantic tokenizer fingerprint also matched after excluding
repository/cache provenance from the fingerprint.

## Design

- 200 deterministic prompts in 50 semantic clusters.
- Four strata: helpfulness, instruction following, honesty, and safety.
- Cluster-disjoint split: 120 train, 40 validation, 40 sealed test.
- Post-block residual stream, all 16 layers, final prompt token.
- FP16 checkpoint inference; float32/float64 analysis.
- Ranks: 1, 2, 4, 8, 16, 32 of residual width 2048.
- Geometry alternatives: global PCA, mixture/local PCA, nonlinear bottleneck
  autoencoder, and random matched-rank subspaces.
- Causal layer selected using exact full-delta interventions on 12 validation
  prompts in both directions.
- Layer/rank/strength selection used validation data only.
- Test effect: normalized recovery of the donor–parent next-token KL gap.

## Key finding 1: observational concentration is not causal localization

Pure reconstruction selected layer 0 and rank 16:

- centered test reconstruction R²: 0.969;
- random-subspace R²: approximately 0.008;
- mean shift alone: 96.9% of raw delta energy.

But even the exact full layer-0 delta closed less than 1% of the output gap. This
is a concrete demonstration that spectral/reconstruction order is not causal
order.

The validation-only bidirectional causal screen instead selected layer 14
(zero-indexed, the penultimate transformer block). Exact delta interventions at
that layer recovered:

- 87.7% for SFT→DPO addition;
- 92.1% for DPO→SFT removal.

## Key finding 2: a small linear subspace carries much of the causal effect

All values below are on the 40 sealed test prompts. Confidence intervals are
paired-prompt bootstrap intervals over the mean of addition and removal recovery.

| PCA rank | Width fraction | Centered R² | Raw delta energy | Bidirectional causal recovery | 95% CI |
|---:|---:|---:|---:|---:|---:|
| 0 (mean) | 0.00% | 0.000 | 0.257 | 0.175 | [0.066, 0.313] |
| 1 | 0.05% | 0.104 | 0.334 | 0.292 | [0.171, 0.390] |
| 2 | 0.10% | 0.176 | 0.387 | 0.388 | [0.289, 0.468] |
| 4 | 0.20% | 0.310 | 0.487 | 0.506 | [0.392, 0.602] |
| 8 | 0.39% | 0.441 | 0.585 | 0.599 | [0.499, 0.677] |
| 16 | 0.78% | 0.542 | 0.660 | 0.697 | [0.620, 0.759] |
| 32 | 1.56% | 0.563 | 0.676 | 0.715 | [0.643, 0.769] |

Three rank/norm-matched rotated controls had mean bidirectional recoveries of
-0.205, -0.029, and -0.096. Rank-32 bootstrap projection similarity was 0.734
(95% interval [0.675, 0.798]).

## Key finding 3: no nonlinear-manifold advantage

At causal layer 14 and latent dimension 32:

- local PCA improved centered reconstruction by only 0.008;
- the nonlinear autoencoder reconstructed less well than global PCA;
- the best nonlinear causal recovery was 0.019 below global PCA.

Accordingly, this experiment does **not** support calling the displacement set a
nonlinear manifold. It supports a narrower claim:

> A causally important portion of the tested SFT→DPO change is concentrated in a
> small global linear actuation subspace, even though that subspace does not
> reconstruct the complete activation displacement.

## Limits

This is not a confirmatory alignment result:

1. Prompts are a small deterministic smoke set, not a natural held-out benchmark.
2. There are only 40 test prompts and one checkpoint pair/training run.
3. The endpoint is next-token distribution recovery, not generated-behaviour or
   human preference recovery.
4. Reconstructions use the true donor delta as an oracle; no gate predicts
   coefficients from SFT alone.
5. Only the final prompt token and one-layer interventions are tested.
6. Only three causal random rotations were run.
7. The experiment does not distinguish propagated state differences from
   per-block transition-function changes.

## Go/no-go

**Go** for a larger single-family study, because the low-rank causal signal is
large, bidirectional, dose-sensitive, and unlike the controls. The next study
should expand prompts, add natural preference/behaviour evaluations, increase
random controls, test early-generation tokens, and use released RLVR revisions to
separate endpoint geometry from training-trajectory geometry.

