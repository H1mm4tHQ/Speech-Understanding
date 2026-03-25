from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import torchaudio


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MFCC_DIR = RESULTS_DIR / "mfcc"
MFCC_DIR.mkdir(parents=True, exist_ok=True)


def load_audio(path):
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).numpy().astype(np.float32), sr


def load_manifest():
    with open(BASE_DIR / "data" / "manifest.csv", "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pre_emphasis(signal, coeff=0.97):
    out = np.empty_like(signal)
    out[0] = signal[0]
    out[1:] = signal[1:] - coeff * signal[:-1]
    return out


def frame_signal(signal, sr, frame_ms=25, hop_ms=10):
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    pad = (frame_len - len(signal) % hop_len) % hop_len
    signal = np.pad(signal, (0, pad))
    frames = []
    for start in range(0, len(signal) - frame_len + 1, hop_len):
        frames.append(signal[start : start + frame_len])
    return np.stack(frames), frame_len, hop_len


def get_window(frame_len, kind="hamming"):
    if kind == "hamming":
        return np.hamming(frame_len)
    if kind == "hanning":
        return np.hanning(frame_len)
    return np.ones(frame_len)


def power_spectrum(frames, n_fft):
    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    return (1.0 / n_fft) * np.abs(spectrum) ** 2


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def mel_filterbank(sr, n_fft, n_mels=26, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = sr / 2
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for k in range(left, center):
            fb[i - 1, k] = (k - left) / max(1, center - left)
        for k in range(center, right):
            fb[i - 1, k] = (right - k) / max(1, right - center)
    return fb


def dct_type_ii(x, num_ceps):
    n = x.shape[1]
    k = np.arange(num_ceps)[:, None]
    m = np.arange(n)[None, :]
    basis = np.cos(np.pi / n * (m + 0.5) * k)
    return x @ basis.T


def compute_manual_mfcc(audio, sr, n_mels=26, num_ceps=13, n_fft=512, window="hamming"):
    emphasized = pre_emphasis(audio)
    frames, frame_len, hop_len = frame_signal(emphasized, sr)
    win = get_window(frame_len, window)
    windowed = frames * win[None, :]
    power = power_spectrum(windowed, n_fft)
    fb = mel_filterbank(sr, n_fft, n_mels=n_mels)
    mel_energies = np.maximum(power @ fb.T, 1e-10)
    log_mel = np.log(mel_energies)
    mfcc = dct_type_ii(log_mel, num_ceps)
    cepstrum = np.fft.irfft(np.log(np.maximum(np.abs(np.fft.rfft(windowed, n=n_fft, axis=1)), 1e-10)), axis=1)
    return {
        "emphasized": emphasized,
        "frames": frames,
        "windowed": windowed,
        "power": power,
        "filterbank": fb,
        "log_mel": log_mel,
        "mfcc": mfcc,
        "cepstrum": cepstrum,
        "frame_len": frame_len,
        "hop_len": hop_len,
    }


def main():
    rows = load_manifest()
    summaries = []
    representative = None

    for idx, row in enumerate(rows):
        audio_path = BASE_DIR / "data" / row["audio_file"]
        audio, sr = load_audio(audio_path)
        feats = compute_manual_mfcc(audio, sr)
        stem = Path(row["audio_file"]).stem

        np.save(MFCC_DIR / f"{stem}_mfcc.npy", feats["mfcc"])
        np.save(MFCC_DIR / f"{stem}_cepstrum.npy", feats["cepstrum"])

        summary = {
            "audio_file": row["audio_file"],
            "sample_rate": sr,
            "duration_sec": float(len(audio) / sr),
            "num_frames": int(feats["mfcc"].shape[0]),
            "num_ceps": int(feats["mfcc"].shape[1]),
            "frame_len": int(feats["frame_len"]),
            "hop_len": int(feats["hop_len"]),
        }
        summaries.append(summary)
        if idx == 0:
            representative = (audio, feats, row["audio_file"])

    np.save(RESULTS_DIR / "mfcc.npy", representative[1]["mfcc"])
    np.save(RESULTS_DIR / "cepstrum.npy", representative[1]["cepstrum"])
    with open(RESULTS_DIR / "mfcc_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_audio_files": len(summaries),
                "representative_audio_file": representative[2],
                "files": summaries,
            },
            f,
            indent=2,
        )

    audio, feats, rep_audio_file = representative
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    axes[0].plot(audio)
    axes[0].set_title(f"Waveform ({rep_audio_file})")
    axes[1].imshow(feats["log_mel"].T, aspect="auto", origin="lower")
    axes[1].set_title("Manual Log-Mel Filterbank Energies")
    axes[2].imshow(feats["mfcc"].T, aspect="auto", origin="lower")
    axes[2].set_title("Manual MFCCs")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mfcc_overview.png")
    plt.close(fig)

    print("Saved manual MFCC outputs to", RESULTS_DIR)


if __name__ == "__main__":
    main()
