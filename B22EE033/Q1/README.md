# Q1 README

This folder contains the Q1 speech analysis pipeline for manual cepstral feature extraction, leakage analysis, voiced/unvoiced boundary detection, and phonetic mapping with transformer-assisted alignment.

## What is implemented

- `prepare_data.py`: prepares the local Q1 manifest/audio layout.
- `mfcc_manual.py`: manual pre-emphasis, framing, windowing, FFT, mel filterbank, log compression, DCT, and cepstrum export.
- `leakage_snr.py`: compares rectangular, Hamming, and Hanning windows using spectral leakage ratio and SNR, and generates the leakage plot.
- `voiced_unvoiced.py`: computes cepstral voicing cues and frame-level voiced/unvoiced boundaries.
- `phonetic_mapping.py`: uses a pretrained local Wav2Vec2 checkpoint for transcript-constrained CTC forced alignment, then maps aligned word spans to phones using a small lexicon.
- `generate_report.py`: builds `q1_report.pdf` from the generated summaries and plots.

## Main data and outputs

- `data/manifest.csv`: transcript list and audio filenames used in Q1.
- `results/mfcc.npy`, `results/cepstrum.npy`: representative cepstral features.
- `results/mfcc_overview.png`: waveform, log-mel, and MFCC overview.
- `results/leakage_snr_plot.png`: windowed spectra plus leakage/SNR comparison.
- `results/voiced_unvoiced_plot.png`: voiced/unvoiced cues and detected boundaries.
- `results/phonetic_mapping_plot.png`: cropped phone map with manual boundaries.
- `results/rmse_table.csv`: RMSE between manual boundaries and model-timed phone boundaries.
- `q1_report.pdf`: final Q1 report.

## Requirements

Install dependencies with:

```powershell
pip install -r Assignment_1/Q1/requirements.txt
```

## How to run

Run the full Q1 pipeline in this order:

```powershell
python Assignment_1/Q1/prepare_data.py
python Assignment_1/Q1/mfcc_manual.py
python Assignment_1/Q1/leakage_snr.py
python Assignment_1/Q1/voiced_unvoiced.py
python Assignment_1/Q1/phonetic_mapping.py
python Assignment_1/Q1/generate_report.py
```

## Notes

- The phonetic alignment timing is driven by local Wav2Vec2 CTC forced alignment.
- Phone labels are expanded from a small lexicon inside each aligned word span, so this is not a phone-native forced aligner.
- RMSE values are measured in seconds.
