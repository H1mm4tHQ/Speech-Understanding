# Q2 README

This folder contains the Q2 reduced reproduction for environment-agnostic speaker recognition using a baseline encoder and disentanglement-based variants on Speech Commands.

## What is implemented

- `train.py`: trains the selected methods from a JSON config, saves checkpoints, metric histories, and a summary file.
- `eval.py`: scans saved method folders, rebuilds summary tables, and writes the comparison plot.
- `generate_review.py`: produces the written review/report PDF for Q2.
- `configs/`: experiment configurations such as `base.json` and `improved.json`.
- `results/`: method subfolders, summary tables, and plots.

## Methods covered

- `baseline`: speaker encoder with speaker classification objective.
- `disentangle`-style variants from config: speaker/environment split with reconstruction, environment prediction, adversarial environment confusion, speaker consistency, and orthogonality regularization.

## Dataset setup

- The code uses `torchaudio.datasets.SPEECHCOMMANDS`.
- If the dataset is not already extracted under `data/SpeechCommands/speech_commands_v0.02`, `train.py` will try to download it automatically.
- The script selects speakers with enough utterances, builds train/test splits, and evaluates robustness under clean and mismatched environments.

## Requirements

Install dependencies with:

```powershell
pip install -r Assignment_1/Q2/requirements.txt
```

## How to run

Typical workflow:

```powershell
python Assignment_1/Q2/train.py --config Assignment_1/Q2/configs/base.json
python Assignment_1/Q2/train.py --config Assignment_1/Q2/configs/improved.json
python Assignment_1/Q2/eval.py --results-root Assignment_1/Q2/results
python Assignment_1/Q2/generate_review.py
```

## Main outputs

- `results/summary.json`
- `results/summary.csv`
- `results/mismatch_eer_plot.png`
- `results/comparison_barplot.png`
- method checkpoints inside `results/<method>/`
- `review.pdf`

## Metrics

- `clean_eer` and `clean_min_dcf`: verification performance without environment mismatch.
- `mismatch_eer` and `mismatch_min_dcf`: verification performance when clean and noisy conditions are mixed.

## Notes

- This is a reduced reproduction designed for local compute limits rather than a full large-scale paper reproduction.
- The exact methods run depend on the `methods` list inside the chosen config file.
