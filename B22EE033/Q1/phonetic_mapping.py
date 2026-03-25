from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
from torchaudio.functional import forced_align, merge_tokens
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_MODEL = (
    BASE_DIR.parent / "Q3" / ".hf_cache" / "hub" / "models--facebook--wav2vec2-base-960h"
    / "snapshots" / "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
)

LEXICON = {
    "without": ["W", "IH", "TH", "AW", "T"],
    "the": ["DH", "AH"],
    "dataset": ["D", "EY", "T", "AH", "S", "EH", "T"],
    "article": ["AA", "R", "T", "IH", "K", "AH", "L"],
    "is": ["IH", "Z"],
    "useless": ["Y", "UW", "S", "L", "AH", "S"],
    "i've": ["AY", "V"],
    "got": ["G", "AA", "T"],
    "to": ["T", "UW"],
    "go": ["G", "OW"],
    "him": ["HH", "IH", "M"],
    "and": ["AH", "N", "D"],
    "you": ["Y", "UW"],
    "know": ["N", "OW"],
    "it": ["IH", "T"],
    "one": ["W", "AH", "N"],
    "you're": ["Y", "UH", "R"],
    "blocking": ["B", "L", "AA", "K", "IH", "NG"],
    "henderson": ["HH", "EH", "N", "D", "ER", "S", "AH", "N"],
    "stood": ["S", "T", "UH", "D"],
    "up": ["AH", "P"],
    "with": ["W", "IH", "TH"],
    "a": ["AH"],
    "spade": ["S", "P", "EY", "D"],
    "in": ["IH", "N"],
    "his": ["HH", "IH", "Z"],
    "hand": ["HH", "AE", "N", "D"],
    "sheep": ["SH", "IY", "P"],
    "had": ["HH", "AE", "D"],
    "taught": ["T", "AO", "T"],
    "that": ["DH", "AE", "T"],
}


def load_manifest():
    rows = []
    with open(BASE_DIR / "data" / "manifest.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_audio(path):
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).numpy().astype(np.float32), sr


def load_manual_boundaries():
    with open(RESULTS_DIR / "voiced_unvoiced_boundaries.json", "r", encoding="utf-8") as f:
        rows = json.load(f)
    return {row["audio_file"]: row["boundaries"] for row in rows}


def normalize_transcript(transcript):
    return transcript.upper().replace(",", "").replace(".", "").replace(" ", "|")


def align_chars(audio, sr, transcript, processor, model):
    if sr != 16000:
        audio = torchaudio.functional.resample(torch.tensor(audio).unsqueeze(0), sr, 16000).squeeze(0).numpy()
        sr = 16000

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    vocab = processor.tokenizer.get_vocab()
    aligned_text = normalize_transcript(transcript)
    target_ids = [vocab[ch] for ch in aligned_text if ch in vocab]
    targets = torch.tensor(target_ids, dtype=torch.int32).unsqueeze(0)
    input_lengths = torch.tensor([logits.shape[1]], dtype=torch.int32)
    target_lengths = torch.tensor([targets.shape[1]], dtype=torch.int32)
    log_probs = torch.log_softmax(logits, dim=-1)
    aligned_tokens, scores = forced_align(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=processor.tokenizer.pad_token_id,
    )
    token_spans = merge_tokens(aligned_tokens[0], scores[0], blank=processor.tokenizer.pad_token_id)

    frame_time = len(audio) / sr / max(1, logits.shape[1])
    char_segments = []
    word_segments = []
    current_word = []

    for ch, span in zip(aligned_text, token_spans):
        start = float(span.start * frame_time)
        end = float(span.end * frame_time)
        if ch == "|":
            if current_word:
                word_segments.append(
                    {
                        "word": "".join(item["char"] for item in current_word),
                        "start": current_word[0]["start"],
                        "end": current_word[-1]["end"],
                    }
                )
                current_word = []
            continue
        seg = {"char": ch.lower(), "start": start, "end": end}
        char_segments.append(seg)
        current_word.append(seg)

    if current_word:
        word_segments.append(
            {
                "word": "".join(item["char"] for item in current_word),
                "start": current_word[0]["start"],
                "end": current_word[-1]["end"],
            }
        )

    return char_segments, word_segments


def words_to_phone_segments(transcript, word_segments):
    words = transcript.lower().replace(",", "").replace(".", "").split()
    phones = []
    for word, word_seg in zip(words, word_segments):
        word_phones = LEXICON.get(word, ["SPN"])
        phone_dur = (word_seg["end"] - word_seg["start"]) / max(1, len(word_phones))
        cursor = word_seg["start"]
        for ph in word_phones:
            phones.append({"phone": ph, "start": cursor, "end": cursor + phone_dur})
            cursor += phone_dur
    return phones


def boundary_rmse(manual_boundaries, phone_segments):
    manual = np.array([b["time_sec"] for b in manual_boundaries[1:]], dtype=np.float32)
    forced = np.array([p["start"] for p in phone_segments[1:]], dtype=np.float32)
    if len(manual) == 0 or len(forced) == 0:
        return float("nan")
    k = min(len(manual), len(forced))
    return float(np.sqrt(np.mean((manual[:k] - forced[:k]) ** 2)))


def plot_rmse_bars(rmse_df):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(rmse_df))
    bars = ax.bar(x, rmse_df["rmse_sec"], color="tab:orange", alpha=0.85)
    ax.axhline(rmse_df["rmse_sec"].mean(), color="tab:red", linestyle="--", label="Mean RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(rmse_df["audio_file"], rotation=25, ha="right")
    ax.set_ylabel("RMSE (s)")
    ax.set_title("RMSE Between Manual Boundaries and Wav2Vec2 Forced Alignment")
    ax.legend()
    for bar, value in zip(bars, rmse_df["rmse_sec"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "rmse_plot.png")
    plt.close(fig)


def main():
    rows = load_manifest()
    processor = Wav2Vec2Processor.from_pretrained(str(LOCAL_MODEL))
    model = Wav2Vec2ForCTC.from_pretrained(str(LOCAL_MODEL))
    model.eval()
    manual_boundaries_by_file = load_manual_boundaries()

    rmse_rows = []
    phone_rows = []
    char_rows = []
    representative = None

    for idx, target in enumerate(rows):
        audio_path = BASE_DIR / "data" / target["audio_file"]
        transcript = target["transcript"]
        audio, sr = load_audio(audio_path)
        char_segments, word_segments = align_chars(audio, sr, transcript, processor, model)
        phone_segments = words_to_phone_segments(transcript, word_segments)
        manual_boundaries = manual_boundaries_by_file.get(target["audio_file"], [])
        rmse = boundary_rmse(manual_boundaries, phone_segments)

        rmse_rows.append({"audio_file": target["audio_file"], "rmse_sec": rmse})
        phone_rows.extend([{**seg, "audio_file": target["audio_file"]} for seg in phone_segments])
        char_rows.extend([{**seg, "audio_file": target["audio_file"]} for seg in char_segments])
        if idx == 0:
            representative = (target["audio_file"], phone_segments, manual_boundaries, rmse)

    pd.DataFrame(phone_rows).to_csv(RESULTS_DIR / "phone_segments.csv", index=False)
    pd.DataFrame(char_rows).to_csv(RESULTS_DIR / "char_segments.csv", index=False)
    rmse_df = pd.DataFrame(rmse_rows)
    rmse_df.to_csv(RESULTS_DIR / "rmse_table.csv", index=False)
    plot_rmse_bars(rmse_df)

    audio_file, phone_segments, manual_boundaries, rmse = representative
    display_start = phone_segments[0]["start"] if phone_segments else 0.0
    display_end = phone_segments[-1]["end"] if phone_segments else 0.0
    visible_boundaries = [
        b for b in manual_boundaries if display_start <= b["time_sec"] <= display_end
    ]

    fig, ax = plt.subplots(figsize=(10, 3))
    for seg in phone_segments:
        ax.axvspan(seg["start"] - display_start, seg["end"] - display_start, alpha=0.28, color="tab:blue")
        ax.text(
            (seg["start"] + seg["end"]) / 2 - display_start,
            0.7,
            seg["phone"],
            ha="center",
            va="center",
            fontsize=8,
        )
    for b in visible_boundaries:
        ax.axvline(b["time_sec"] - display_start, color="#ff6b6b", linestyle="--", alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(0.05, display_end - display_start))
    ax.set_title(f"Phone Mapping and Manual Boundaries (RMSE={rmse:.4f}s)")
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phonetic_mapping_plot.png")
    plt.close(fig)

    with open(RESULTS_DIR / "phonetic_mapping_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_audio_files": len(rows),
                "representative_audio_file": audio_file,
                "mean_rmse_sec": float(rmse_df["rmse_sec"].mean()),
                "per_file_rmse": rmse_rows,
                "model_path": str(LOCAL_MODEL),
                "alignment_method": "Wav2Vec2 CTC forced alignment on transcript characters, then lexicon-based phone mapping inside aligned word spans",
            },
            f,
            indent=2,
        )
    print("Saved phonetic mapping outputs to", RESULTS_DIR)


if __name__ == "__main__":
    main()
