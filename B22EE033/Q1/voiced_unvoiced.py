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


def frame_signal(signal, sr, frame_ms=25, hop_ms=10):
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    pad = (frame_len - len(signal) % hop_len) % hop_len
    signal = np.pad(signal, (0, pad))
    frames = []
    starts = []
    for start in range(0, len(signal) - frame_len + 1, hop_len):
        frames.append(signal[start : start + frame_len])
        starts.append(start)
    return np.stack(frames), np.array(starts), frame_len, hop_len


def cepstrum(frame, n_fft=512):
    spectrum = np.fft.rfft(frame, n=n_fft)
    log_mag = np.log(np.maximum(np.abs(spectrum), 1e-10))
    return np.fft.irfft(log_mag)


def detect_voiced_unvoiced(audio, sr):
    frames, starts, frame_len, hop_len = frame_signal(audio, sr)
    window = np.hamming(frame_len)
    energies, low_env, high_peak, voiced = [], [], [], []
    q_low = int(0.001 * sr)
    q_high_lo = int(0.002 * sr)
    q_high_hi = int(0.012 * sr)
    for frame in frames:
        f = frame * window
        c = cepstrum(f)
        energy = np.mean(f ** 2)
        low = np.sum(np.abs(c[:q_low]))
        high = np.max(np.abs(c[q_high_lo:q_high_hi]))
        energies.append(energy)
        low_env.append(low)
        high_peak.append(high)
    energies = np.array(energies)
    low_env = np.array(low_env)
    high_peak = np.array(high_peak)

    energy_th = np.median(energies)
    pitch_th = np.median(high_peak)
    mask = (energies > energy_th * 0.5) & (high_peak > pitch_th)
    times = starts / sr
    boundaries = []
    prev = mask[0]
    boundaries.append({"time_sec": float(times[0]), "label": "voiced" if prev else "unvoiced"})
    for idx in range(1, len(mask)):
        if mask[idx] != prev:
            boundaries.append({"time_sec": float(times[idx]), "label": "voiced" if mask[idx] else "unvoiced"})
            prev = mask[idx]
    rep_idx = int(np.argmax(energies))
    rep_frame = frames[rep_idx] * window
    rep_cepstrum = cepstrum(rep_frame)
    return {
        "times": times,
        "mask": mask,
        "energy": energies,
        "low_env": low_env,
        "high_peak": high_peak,
        "boundaries": boundaries,
        "rep_cepstrum": rep_cepstrum,
        "rep_frame_time_sec": float(times[rep_idx]),
        "q_low": q_low,
        "q_high_lo": q_high_lo,
        "q_high_hi": q_high_hi,
    }


def main():
    rows = load_manifest()
    all_outputs = []
    boundary_rows = []
    representative = None

    for idx, row in enumerate(rows):
        audio_path = BASE_DIR / "data" / row["audio_file"]
        audio, sr = load_audio(audio_path)
        out = detect_voiced_unvoiced(audio, sr)
        all_outputs.append({"audio_file": row["audio_file"], "boundaries": out["boundaries"]})
        boundary_rows.append(
            {
                "audio_file": row["audio_file"],
                "num_boundaries": len(out["boundaries"]),
                "voiced_ratio": float(np.mean(out["mask"])),
                "duration_sec": float(len(audio) / sr),
            }
        )
        if idx == 0:
            representative = (row["audio_file"], audio, sr, out)

    with open(RESULTS_DIR / "voiced_unvoiced_boundaries.json", "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, indent=2)
    pd.DataFrame(boundary_rows).to_csv(RESULTS_DIR / "voiced_unvoiced_summary.csv", index=False)

    audio_file, audio, sr, out = representative
    t = np.arange(len(audio)) / sr
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(t, audio)
    axes[0].set_title(f"Waveform ({audio_file})")
    axes[1].plot(out["times"], out["energy"], label="Frame energy")
    axes[1].plot(out["times"], out["high_peak"], label="High-quefrency peak")
    axes[1].legend()
    axes[1].set_title("Cepstral Voicing Cues")
    axes[2].step(out["times"], out["mask"].astype(int), where="post")
    axes[2].set_title("Voiced (1) / Unvoiced (0)")
    axes[2].set_xlabel("Time (s)")
    for b in out["boundaries"]:
        axes[0].axvline(b["time_sec"], color="r", alpha=0.3)
        axes[2].axvline(b["time_sec"], color="r", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "voiced_unvoiced_plot.png")
    plt.close(fig)

    q = np.arange(len(out["rep_cepstrum"])) / sr
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 7))
    axes2[0].plot(q * 1000, out["rep_cepstrum"], color="black", linewidth=1.2)
    axes2[0].axvspan(0, out["q_low"] / sr * 1000, color="tab:blue", alpha=0.25, label="Low quefrency")
    axes2[0].axvspan(
        out["q_high_lo"] / sr * 1000,
        out["q_high_hi"] / sr * 1000,
        color="tab:orange",
        alpha=0.25,
        label="High quefrency",
    )
    axes2[0].set_title(f"Representative Cepstrum Regions ({audio_file}, frame at {out['rep_frame_time_sec']:.3f}s)")
    axes2[0].set_xlabel("Quefrency (ms)")
    axes2[0].set_ylabel("Cepstrum amplitude")
    axes2[0].legend()

    axes2[1].plot(out["times"], out["low_env"], label="Low-quefrency envelope cue", color="tab:blue")
    axes2[1].plot(out["times"], out["high_peak"], label="High-quefrency pitch cue", color="tab:orange")
    axes2[1].set_title("Low vs High Quefrency Cues Over Time")
    axes2[1].set_xlabel("Time (s)")
    axes2[1].set_ylabel("Cue value")
    axes2[1].legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "quefrency_regions_plot.png")
    plt.close(fig2)

    print("Saved voiced/unvoiced outputs to", RESULTS_DIR)


if __name__ == "__main__":
    main()
