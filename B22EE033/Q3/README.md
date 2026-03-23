# Q3 README

This folder contains the Q3 work on ethical auditing, privacy-preserving voice transformation, and fairness-aware speech training using a local Common Voice sample.

## What is implemented

- `Audit.py`: audits documentation debt and representation bias from `cv-valid-test.csv`, then saves summary text and distribution plots.
- `privacymodule.py`: defines `PrivacyVoiceTransformer`, which applies pitch shifting and speed perturbation.
- `pp_demo.py`: creates a before/after demo pair in `examples/`.
- `evaluation_scripts.py`: computes lightweight proxy metrics for transformed audio quality and distance.
- `train_fair.py`: trains a small local CTC ASR model with an added fairness penalty over male/female batch losses.


## Data and folders

- `cv-valid-test.csv`: metadata for the local Common Voice subset.
- `cv-valid-test/`: local audio files referenced by the CSV.
- `.hf_cache/`: local Hugging Face cache used by nearby experiments.
- `examples/`: privacy transformation demo audio.

## Requirements

Install dependencies with:

```powershell
pip install -r Assignment_1/Q3/requirements.txt
```

## How to run

Recommended order:

```powershell
python Assignment_1/Q3/Audit.py
python Assignment_1/Q3/pp_demo.py
python Assignment_1/Q3/evaluation_scripts.py
python Assignment_1/Q3/train_fair.py > Assignment_1/Q3/train_fair_results.txt
```

## Main outputs

- `audit_summary.txt`
- `gender_distribution.png`
- `age_distribution.png`
- `audit_plots.pdf`
- `examples/before_1.wav`
- `examples/after_1.wav`
- `train_fair_results.txt`
- `q3_report.pdf`

## Notes

- The privacy evaluation script reports proxy values, not full DNSMOS or full Fréchet Audio Distance.
- The fairness training is a lightweight local CTC setup, not a full large pretrained ASR fine-tune.
- The demographic analysis depends on the metadata present in `cv-valid-test.csv`, which contains missing fields.
