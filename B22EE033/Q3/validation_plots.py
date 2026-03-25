from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


BASE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BASE_DIR / "examples"
BEFORE_FILE = EXAMPLES_DIR / "before_1.wav"
AFTER_FILE = EXAMPLES_DIR / "after_1.wav"
TRAIN_LOG = BASE_DIR / "train_fair_results.txt"


def load_audio(path: Path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), sr


def simple_quality_score(audio: np.ndarray) -> float:
    return float(np.mean(audio ** 2))


def compute_distance(a1: np.ndarray, a2: np.ndarray) -> float:
    min_len = min(len(a1), len(a2))
    return float(np.mean((a1[:min_len] - a2[:min_len]) ** 2))


def stft_db(audio: np.ndarray, n_fft: int = 512, hop: int = 160):
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    frames = []
    window = np.hanning(n_fft)
    for start in range(0, len(audio) - n_fft + 1, hop):
        frame = audio[start : start + n_fft] * window
        spec = np.fft.rfft(frame, n=n_fft)
        frames.append(20 * np.log10(np.maximum(np.abs(spec), 1e-6)))
    return np.stack(frames, axis=1)


def parse_training(path: Path):
    text = path.read_text(encoding="utf-8")
    rows = []
    epoch = None
    step = 0
    for line in text.splitlines():
        epoch_match = re.match(r"Epoch\s+(\d+)", line.strip())
        if epoch_match:
            epoch = int(epoch_match.group(1))
            continue
        loss_match = re.search(r"Base Loss:\s*([0-9.]+)\s*\|\s*Fair Loss:\s*([0-9.]+)", line)
        if loss_match and epoch is not None:
            rows.append(
                {
                    "global_step": step,
                    "epoch": epoch,
                    "base_loss": float(loss_match.group(1)),
                    "fair_loss": float(loss_match.group(2)),
                }
            )
            step += 1
    return rows


def main():
    before_audio, before_sr = load_audio(BEFORE_FILE)
    after_audio, after_sr = load_audio(AFTER_FILE)
    if before_sr != after_sr:
        raise ValueError("Before/after audio must have same sample rate for comparison.")

    before_t = np.arange(len(before_audio)) / before_sr
    after_t = np.arange(len(after_audio)) / after_sr

    dnsmos_proxy_before = simple_quality_score(before_audio)
    dnsmos_proxy_after = simple_quality_score(after_audio)
    fad_proxy_distance = compute_distance(before_audio, after_audio)

    before_spec = stft_db(before_audio)
    after_spec = stft_db(after_audio)
    training_rows = parse_training(TRAIN_LOG)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    axes[0, 0].plot(before_t, before_audio, color="tab:blue")
    axes[0, 0].set_title("Before Transformation Waveform")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")

    axes[0, 1].plot(after_t, after_audio, color="tab:orange")
    axes[0, 1].set_title("After Transformation Waveform")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Amplitude")

    im0 = axes[1, 0].imshow(before_spec, aspect="auto", origin="lower", cmap="magma")
    axes[1, 0].set_title("Before Transformation Spectrogram")
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].set_ylabel("Frequency bin")
    fig.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im1 = axes[1, 1].imshow(after_spec, aspect="auto", origin="lower", cmap="magma")
    axes[1, 1].set_title("After Transformation Spectrogram")
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].set_ylabel("Frequency bin")
    fig.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)

    metric_names = ["Before proxy\nenergy", "After proxy\nenergy", "Before/After\nproxy distance"]
    metric_values = [dnsmos_proxy_before, dnsmos_proxy_after, fad_proxy_distance]
    axes[2, 0].bar(metric_names, metric_values, color=["tab:blue", "tab:orange", "tab:green"])
    axes[2, 0].set_title("Validation Proxy Metrics")
    axes[2, 0].set_ylabel("Value")

    if training_rows:
        steps = [row["global_step"] for row in training_rows]
        base_loss = [row["base_loss"] for row in training_rows]
        fair_loss = [row["fair_loss"] for row in training_rows]
        axes[2, 1].plot(steps, base_loss, label="Base loss", color="tab:blue")
        axes[2, 1].plot(steps, fair_loss, label="Fairness loss", color="tab:red")
        axes[2, 1].set_title("Training Loss Trends")
        axes[2, 1].set_xlabel("Global step")
        axes[2, 1].set_ylabel("Loss")
        axes[2, 1].legend()
    else:
        axes[2, 1].text(0.5, 0.5, "No training log found", ha="center", va="center")
        axes[2, 1].set_title("Training Loss Trends")
        axes[2, 1].axis("off")

    plt.tight_layout()
    out_path = BASE_DIR / "validation_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved validation comparison to", out_path)


if __name__ == "__main__":
    main()
