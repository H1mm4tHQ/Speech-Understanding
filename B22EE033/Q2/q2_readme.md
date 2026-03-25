# Q2 Modify

This folder contains a LibriSpeech-based reduced reproduction of the paper **Disentangled Representation Learning for Environment-agnostic Speaker Recognition**.

## What is included

- `train.py`: trains the baseline speaker encoder, then the paper-style disentangler and the critique-driven improvement.
- `eval.py`: reloads checkpoints and regenerates result tables and plots.
- `q2_core.py`: shared data pipeline, models, losses, metrics, plots, and PDF review code.
- `configs/`: experiment configs.
- `results/`: checkpoints, histories, tables, and plots.

## Reproduction commands

```powershell
python Q2_modify/train.py --config Q2_modify/configs/smoke.json --download
python Q2_modify/eval.py --config Q2_modify/configs/smoke.json --results-root Q2_modify
```

Larger run:

```powershell
python Q2_modify/train.py --config Q2_modify/configs/reproduction.json --download
python Q2_modify/eval.py --config Q2_modify/configs/reproduction.json --results-root Q2_modify
```

Review only:

```powershell
python Q2_modify/generate_review.py --output Q2_modify/review.pdf
```

## Methods

- `baseline`: TDNN speaker encoder with AAM-Softmax.
- `paper_reproduction`: frozen baseline encoder plus disentangler with speaker classification, environment classification, adversarial environment confusion, reconstruction, Pearson decorrelation, and cross-environment invariance.
- `improved`: `paper_reproduction` plus supervised contrastive regularization on the speaker code.

## Dataset choice

- Default dataset: `LibriSpeech/train-clean-100`.
- Evaluation uses held-out utterances from the selected speakers.

## Expected outputs

- `review.pdf`
- `results/summary.json`
- `results/summary.csv`
- `results/eer_by_condition.png`
- `results/top1_by_condition.png`
- per-method histories and `best.pt` checkpoints inside `results/baseline/`, `results/paper_reproduction/`, and `results/improved/`

## Executed in this workspace

The run completed in this workspace used:

```powershell
python Q2_modify/train.py --config Q2_modify/configs/smoke.json
python Q2_modify/eval.py --config Q2_modify/configs/smoke.json --results-root Q2_modify
```

Those results correspond to:

- `results/baseline/best.pt`
- `results/paper_reproduction/best.pt`
- `results/improved/best.pt`

Quick summary from `results/summary.csv`:

- `baseline`: `clean_noise_eer=0.4583`, `clean_reverb_eer=0.5000`, `clean_bandlimit_eer=0.5397`
- `paper_reproduction`: `clean_noise_eer=0.5437`, `clean_reverb_eer=0.5794`, `clean_bandlimit_eer=0.5000`
- `improved`: `clean_noise_eer=0.5030`, `clean_reverb_eer=0.5203`, `clean_bandlimit_eer=0.4816`

The larger `configs/reproduction.json` experiment is prepared but was not executed in this session because the full `train-clean-100` run on CPU would take substantially longer.
