# Reproducibility Guide

## Levels of Reproduction

This repository supports three different verification levels.

### 1. Unit Tests

```powershell
python -m pip install -e ".[dev]"
pytest -q
```

These tests do not download model weights.

### 2. Smoke Experiment

Use `configs/smoke.yaml` to verify prompt generation, activation extraction,
geometry analysis, and causal metrics on the smallest configured experiment.
Hugging Face credentials are read only from `HF_TOKEN` and are never serialized
by the application.

### 3. Full Trajectory Experiments

The primary 7B configuration is `configs/trajectory_7b.yaml`. It pins the exact
model repositories and revisions used in the report. Extended-rank,
stratified-split, and precision-calibration configurations are also included.

Full reproduction requires downloading multiple 7B checkpoints. The pipeline
loads checkpoints sequentially, records resolved revisions and software
versions, and stores only selected-token activations. Model caches and generated
activation arrays are ignored by Git.

## Result Verification Without Rerunning Models

Compact machine-readable outputs are included under `results/`:

- `trajectory_7b_rank32.json`
- `trajectory_7b_extended_rank.json`
- `trajectory_7b_robustness.json`
- `trajectory_7b_causal.json`
- `provenance/*.manifest.json`

These files preserve the exact metrics and checkpoint provenance needed to
audit the prose report without redistributing the large activation arrays.

## Security

- Never commit Hugging Face tokens or `.env` files.
- Do not commit model caches, weights, checkpoints, or activation arrays.
- Review the external model cards and terms before downloading checkpoints.
- Repeat the repository credential scan before every public release.

