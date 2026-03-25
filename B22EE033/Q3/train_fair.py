from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

BASE_DIR = Path(__file__).resolve().parent
LOCAL_CSV = BASE_DIR / "cv-valid-test.csv"
if not LOCAL_CSV.exists():
    raise FileNotFoundError(f"Missing local metadata file: {LOCAL_CSV}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load Local Dataset
# -------------------------------
df = pd.read_csv(LOCAL_CSV)
df = df.dropna(subset=["text", "gender"])
df = df[df["gender"].isin(["male", "female"])].copy()
df = df.head(32).copy()
df["audio_path"] = df["filename"].apply(lambda x: str(BASE_DIR / x))

vocab_chars = sorted(set(" ".join(df["text"].str.lower().tolist())))
char2id = {"<blank>": 0}
for idx, ch in enumerate(vocab_chars, start=1):
    char2id[ch] = idx
id2char = {idx: ch for ch, idx in char2id.items()}

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    hop_length=160,
    n_mels=64
)

def encode_text(text):
    text = text.lower()
    return torch.tensor([char2id[ch] for ch in text if ch in char2id], dtype=torch.long)

def load_audio(path):
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    features = mel_transform(waveform).squeeze(0).transpose(0, 1)
    features = torch.log(features + 1e-6)
    return features

class LocalSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        return {
            "input_features": load_audio(row["audio_path"]),
            "labels": encode_text(row["text"]),
            "gender": row["gender"],
        }

def collate_fn(batch):
    features = [b["input_features"] for b in batch]
    labels = [b["labels"] for b in batch]
    genders = [b["gender"] for b in batch]

    input_lengths = torch.tensor([x.size(0) for x in features], dtype=torch.long)
    label_lengths = torch.tensor([x.size(0) for x in labels], dtype=torch.long)

    padded_features = pad_sequence(features, batch_first=True)
    padded_labels = torch.cat(labels)

    return {
        "input_features": padded_features,
        "labels": padded_labels,
        "input_lengths": input_lengths,
        "label_lengths": label_lengths,
        "gender": genders,
    }

dataset = LocalSpeechDataset(df)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

class TinyCTCASR(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super().__init__()
        self.encoder = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, features):
        encoded, _ = self.encoder(features)
        return self.classifier(encoded)

model = TinyCTCASR(input_dim=64, hidden_dim=64, vocab_size=len(char2id)).to(device)
ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

# -------------------------------
# Fairness Loss
# -------------------------------
def fairness_loss(group_losses):
    if len(group_losses) < 2:
        return torch.tensor(0.0, device=device)

    values = torch.stack(list(group_losses.values()))
    return torch.var(values)

# -------------------------------
# Optimizer
# -------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

lambda_fair = 0.5

# -------------------------------
# Training Loop
# -------------------------------
for epoch in range(2):
    print(f"\nEpoch {epoch}")

    for batch in loader:
        model.train()
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        label_lengths = batch["label_lengths"].to(device)

        logits = model(input_features)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        base_loss = ctc_loss_fn(log_probs, labels, input_lengths, label_lengths)

        # -----------------------
        # Compute group losses
        # -----------------------
        group_losses = {}

        genders = batch["gender"]

        for g in ["male", "female"]:
            indices = [i for i, x in enumerate(genders) if x == g]

            if len(indices) > 0:
                idx = torch.tensor(indices, device=device)
                sub_inputs = input_features[idx]
                sub_input_lengths = input_lengths[idx]

                label_start = 0
                selected_labels = []
                selected_label_lengths = []
                for sample_idx, sample_len in enumerate(label_lengths.tolist()):
                    label_slice = labels[label_start:label_start + sample_len]
                    if sample_idx in indices:
                        selected_labels.append(label_slice)
                        selected_label_lengths.append(sample_len)
                    label_start += sample_len

                if selected_labels:
                    sub_labels = torch.cat(selected_labels).to(device)
                    sub_label_lengths = torch.tensor(selected_label_lengths, device=device)
                    sub_logits = model(sub_inputs)
                    sub_log_probs = F.log_softmax(sub_logits, dim=-1).transpose(0, 1)
                    group_losses[g] = ctc_loss_fn(
                        sub_log_probs,
                        sub_labels,
                        sub_input_lengths,
                        sub_label_lengths
                    )

        fair_loss = fairness_loss(group_losses)

        total_loss = base_loss + lambda_fair * fair_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(
            f"Base Loss: {base_loss.item():.4f} | "
            f"Fair Loss: {fair_loss.item():.4f}"
        )
