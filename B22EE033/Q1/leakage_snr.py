from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchaudio


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_audio(path):
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).numpy().astype(np.float32), sr


def load_manifest():
    with open(BASE_DIR / "data" / "manifest.csv", "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def window_fn(name, n):
    if name == "rectangular":
        return np.ones(n)
    if name == "hamming":
        return np.hamming(n)
    return np.hanning(n)


def analyze_segment(segment, window_name, n_fft=1024, keep_bins=5):
    win = window_fn(window_name, len(segment))
    windowed = segment * win
    spec = np.fft.rfft(windowed, n=n_fft)
    mag2 = np.abs(spec) ** 2
    total = np.sum(mag2)
    peak_idx = int(np.argmax(mag2))
    lo = max(0, peak_idx - keep_bins)
    hi = min(len(mag2), peak_idx + keep_bins + 1)
    main_energy = np.sum(mag2[lo:hi])
    leakage_energy = max(total - main_energy, 1e-12)
    leakage_ratio = leakage_energy / max(total, 1e-12)
    snr_db = 10 * np.log10(max(main_energy, 1e-12) / leakage_energy)
    return {
        "window": window_name,
        "peak_bin": peak_idx,
        "main_energy": float(main_energy),
        "leakage_energy": float(leakage_energy),
        "spectral_leakage_ratio": float(leakage_ratio),
        "snr_db": float(snr_db),
        "spectrum": mag2,
        "window_curve": win,
        "windowed_segment": windowed,
    }


def select_speech_segment(audio, sr, segment_ms=40):
    seg_len = max(1, int(segment_ms * sr / 1000))
    if len(audio) <= seg_len:
        return audio
    hop = max(1, seg_len // 2)
    best_start = 0
    best_energy = -np.inf
    for start in range(0, len(audio) - seg_len + 1, hop):
        segment = audio[start : start + seg_len]
        energy = float(np.mean(segment ** 2))
        if energy > best_energy:
            best_energy = energy
            best_start = start
    return audio[best_start : best_start + seg_len]


def main():
    rows = load_manifest()
    analyses = []
    representative = None
    representative_segment = None
    for idx, row in enumerate(rows):
        audio_path = BASE_DIR / "data" / row["audio_file"]
        audio, sr = load_audio(audio_path)
        segment = select_speech_segment(audio, sr)
        for window_name in ["rectangular", "hamming", "hanning"]:
            analysis = analyze_segment(segment, window_name)
            analysis["audio_file"] = row["audio_file"]
            analyses.append(analysis)
        if idx == 0:
            representative = row["audio_file"]
            representative_segment = segment

    table = pd.DataFrame(
        [
            {
                "audio_file": a["audio_file"],
                "window": a["window"],
                "spectral_leakage_ratio": a["spectral_leakage_ratio"],
                "snr_db": a["snr_db"],
            }
            for a in analyses
        ]
    )
    table.to_csv(RESULTS_DIR / "leakage_snr_table.csv", index=False)

    avg_table = (
        table.groupby("window", as_index=False)[["spectral_leakage_ratio", "snr_db"]]
        .mean()
        .sort_values("window")
    )
    representative_analyses = [x for x in analyses if x["audio_file"] == representative]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    for a in representative_analyses:
        axes[0].plot(10 * np.log10(a["spectrum"] + 1e-10), label=a["window"])
    axes[0].set_title("Windowed Spectra")
    axes[0].set_xlabel("FFT bin")
    axes[0].set_ylabel("Power (dB)")
    axes[0].legend()

    x = np.arange(len(avg_table))
    axes[1].bar(x - 0.15, avg_table["spectral_leakage_ratio"], width=0.3, label="Leakage ratio")
    axes[1].bar(x + 0.15, avg_table["snr_db"], width=0.3, label="SNR (dB)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(avg_table["window"])
    axes[1].set_title("Spectral Leakage and SNR by Window")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "leakage_snr_plot.png")
    plt.close(fig)

    with open(RESULTS_DIR / "leakage_snr_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_audio_files": len(rows),
                "representative_audio_file": representative,
                "per_file": table.to_dict(orient="records"),
                "average_by_window": avg_table.to_dict(orient="records"),
            },
            f,
            indent=2,
        )

    print("Saved leakage/SNR outputs to", RESULTS_DIR)


if __name__ == "__main__":
    main()
