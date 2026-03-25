import json
import random
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader, Dataset


ENVIRONMENTS = ["clean", "noise", "reverb", "bandlimit"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, payload: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def ensure_librispeech(root: Path, subsets: List[str], download: bool) -> None:
    for subset in subsets:
        subset_dir = root / "LibriSpeech" / subset
        if subset_dir.exists():
            continue
        torchaudio.datasets.LIBRISPEECH(str(root), url=subset, download=download)


def index_librispeech(root: Path, subsets: List[str], download: bool) -> List[dict]:
    ensure_librispeech(root, subsets, download)
    items: List[dict] = []
    for subset in subsets:
        subset_root = root / "LibriSpeech" / subset
        for flac_path in subset_root.rglob("*.flac"):
            speaker_id = int(flac_path.parent.parent.name)
            chapter_id = int(flac_path.parent.name)
            utterance_id = int(flac_path.stem.split("-")[-1])
            items.append(
                {
                    "path": str(flac_path),
                    "subset": subset,
                    "speaker_id": speaker_id,
                    "chapter_id": chapter_id,
                    "utterance_id": utterance_id,
                    "transcript": "",
                    "sample_rate": 16000,
                }
            )
    return items


def build_speaker_splits(items: List[dict], cfg: dict) -> Dict[str, List[dict]]:
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for item in items:
        grouped[item["speaker_id"]].append(item)

    eligible = []
    for speaker_id, utts in grouped.items():
        if len(utts) >= cfg["min_utterances_per_speaker"]:
            eligible.append((speaker_id, sorted(utts, key=lambda x: x["path"])))

    eligible.sort(key=lambda x: (-len(x[1]), x[0]))
    selected = eligible[: cfg["num_speakers"]]

    train_items: List[dict] = []
    val_items: List[dict] = []
    test_items: List[dict] = []

    for speaker_id, utts in selected:
        train_end = min(cfg["train_utterances_per_speaker"], len(utts))
        val_end = min(train_end + cfg["val_utterances_per_speaker"], len(utts))
        test_end = min(val_end + cfg["test_utterances_per_speaker"], len(utts))
        train_items.extend(utts[:train_end])
        val_items.extend(utts[train_end:val_end])
        test_items.extend(utts[val_end:test_end])

    return {
        "train": train_items,
        "val": val_items,
        "test": test_items,
        "speaker_ids": [speaker_id for speaker_id, _ in selected],
    }


def apply_environment(waveform: torch.Tensor, env_id: int) -> torch.Tensor:
    squeezed = False
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
        squeezed = True
    if env_id == 0:
        return waveform.squeeze(0) if squeezed else waveform
    if env_id == 1:
        noise_scale = random.uniform(0.01, 0.04)
        augmented = torch.clamp(waveform + noise_scale * torch.randn_like(waveform), -1.0, 1.0)
        return augmented.squeeze(0) if squeezed else augmented
    if env_id == 2:
        kernel_len = random.choice([400, 800, 1200])
        decay = torch.exp(-torch.linspace(0, random.uniform(3.0, 6.0), kernel_len, device=waveform.device))
        rir = decay / decay.abs().sum().clamp_min(1e-6)
        reverbed = F.conv1d(
            waveform.reshape(-1, 1, waveform.shape[-1]),
            rir.view(1, 1, -1),
            padding=kernel_len - 1,
        )
        reverbed = reverbed.reshape(waveform.shape[0], waveform.shape[1], -1)[..., : waveform.shape[-1]]
        return reverbed.squeeze(0) if squeezed else reverbed
    lowpass = torchaudio.functional.lowpass_biquad(waveform, 16000, cutoff_freq=random.uniform(1800, 3200))
    bandlimited = torchaudio.functional.highpass_biquad(lowpass, 16000, cutoff_freq=random.uniform(120, 250))
    bandlimited = torch.clamp(bandlimited, -1.0, 1.0)
    return bandlimited.squeeze(0) if squeezed else bandlimited


def choose_different_env(env_id: int) -> int:
    candidates = [idx for idx in range(len(ENVIRONMENTS)) if idx != env_id]
    return random.choice(candidates)


class LogMelFrontend(nn.Module):
    def __init__(self, sample_rate: int, n_mels: int, frame_length: int, hop_length: int):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=frame_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.instancenorm = nn.InstanceNorm1d(n_mels)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3 and waveform.shape[1] == 1:
            waveform = waveform.squeeze(1)
        feat = self.mel(waveform).clamp_min(1e-6).log()
        feat = self.instancenorm(feat)
        return feat.transpose(1, 2)


class BaselineTrainDataset(Dataset):
    def __init__(self, items: List[dict], speaker_to_idx: Dict[int, int], cfg: dict):
        self.items = items
        self.speaker_to_idx = speaker_to_idx
        self.segment_samples = int(cfg["sample_rate"] * cfg["segment_seconds"])
        self.sample_rate = cfg["sample_rate"]
        self.train_with_augmentation = cfg.get("baseline_train_augmentation", True)

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[-1] < self.segment_samples:
            waveform = F.pad(waveform, (0, self.segment_samples - waveform.shape[-1]))
        elif waveform.shape[-1] > self.segment_samples:
            max_offset = waveform.shape[-1] - self.segment_samples
            offset = random.randint(0, max_offset)
            waveform = waveform[:, offset : offset + self.segment_samples]
        return waveform

    def __getitem__(self, idx: int):
        item = self.items[idx]
        waveform = self._load(item["path"])
        env_id = random.randrange(len(ENVIRONMENTS)) if self.train_with_augmentation else 0
        waveform = apply_environment(waveform, env_id)
        return waveform, self.speaker_to_idx[item["speaker_id"]], env_id


class DisentanglerTrainDataset(Dataset):
    def __init__(self, items: List[dict], speaker_to_idx: Dict[int, int], cfg: dict):
        self.items = items
        self.speaker_to_idx = speaker_to_idx
        self.segment_samples = int(cfg["sample_rate"] * cfg["segment_seconds"])
        self.sample_rate = cfg["sample_rate"]

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[-1] < self.segment_samples:
            waveform = F.pad(waveform, (0, self.segment_samples - waveform.shape[-1]))
        elif waveform.shape[-1] > self.segment_samples:
            max_offset = waveform.shape[-1] - self.segment_samples
            offset = random.randint(0, max_offset)
            waveform = waveform[:, offset : offset + self.segment_samples]
        return waveform

    def __getitem__(self, idx: int):
        item = self.items[idx]
        waveform = self._load(item["path"])
        env_a = random.randrange(len(ENVIRONMENTS))
        env_b = choose_different_env(env_a)
        return {
            "view_a": apply_environment(waveform.clone(), env_a),
            "view_b": apply_environment(waveform.clone(), env_b),
            "env_a": env_a,
            "env_b": env_b,
            "speaker_idx": self.speaker_to_idx[item["speaker_id"]],
            "speaker_id": item["speaker_id"],
            "path": item["path"],
        }


class EvaluationUtteranceDataset(Dataset):
    def __init__(self, items: List[dict], cfg: dict):
        self.items = items
        self.segment_samples = int(cfg["sample_rate"] * cfg["segment_seconds"])
        self.sample_rate = cfg["sample_rate"]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        waveform, sr = torchaudio.load(item["path"])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[-1] < self.segment_samples:
            waveform = F.pad(waveform, (0, self.segment_samples - waveform.shape[-1]))
        else:
            waveform = waveform[:, : self.segment_samples]
        return waveform, item["speaker_id"], item["path"]


def collate_waveforms(batch):
    return (
        torch.stack([x[0] for x in batch], dim=0),
        torch.tensor([x[1] for x in batch], dtype=torch.long),
        torch.tensor([x[2] for x in batch], dtype=torch.long),
    )


def collate_disentangler(batch):
    return (
        torch.stack([x["view_a"] for x in batch], dim=0),
        torch.stack([x["view_b"] for x in batch], dim=0),
        torch.tensor([x["env_a"] for x in batch], dtype=torch.long),
        torch.tensor([x["env_b"] for x in batch], dtype=torch.long),
        torch.tensor([x["speaker_idx"] for x in batch], dtype=torch.long),
        [x["speaker_id"] for x in batch],
        [x["path"] for x in batch],
    )


def collate_eval(batch):
    return (
        torch.stack([x[0] for x in batch], dim=0),
        [x[1] for x in batch],
        [x[2] for x in batch],
    )


class TDNNBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentiveStatsPool(nn.Module):
    def __init__(self, channels: int, bottleneck: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, bottleneck, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(bottleneck, channels, kernel_size=1),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention(x)
        mean = torch.sum(x * weights, dim=-1)
        var = torch.sum(((x - mean.unsqueeze(-1)) ** 2) * weights, dim=-1).clamp_min(1e-5)
        std = torch.sqrt(var)
        return torch.cat([mean, std], dim=-1)


class SpeakerEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        channels = cfg["encoder_channels"]
        self.frontend = LogMelFrontend(
            sample_rate=cfg["sample_rate"],
            n_mels=cfg["n_mels"],
            frame_length=cfg["frame_length"],
            hop_length=cfg["hop_length"],
        )
        self.tdnn = nn.Sequential(
            TDNNBlock(cfg["n_mels"], channels, kernel_size=5),
            TDNNBlock(channels, channels, kernel_size=3, dilation=2),
            TDNNBlock(channels, channels, kernel_size=3, dilation=3),
            TDNNBlock(channels, channels, kernel_size=1),
        )
        self.pool = AttentiveStatsPool(channels)
        self.proj = nn.Sequential(
            nn.Linear(channels * 2, cfg["embedding_dim"]),
            nn.BatchNorm1d(cfg["embedding_dim"]),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        feat = self.frontend(waveform)
        x = self.tdnn(feat.transpose(1, 2))
        pooled = self.pool(x)
        return self.proj(pooled)


class AAMSoftmax(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=-1)
        weights = F.normalize(self.weight, dim=-1)
        cosine = F.linear(embeddings, weights).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)
        target = torch.cos(theta + self.margin)
        one_hot = F.one_hot(labels, num_classes=weights.shape[0]).float()
        logits = cosine * (1.0 - one_hot) + target * one_hot
        return logits * self.scale


class BaselineSpeakerNet(nn.Module):
    def __init__(self, cfg: dict, num_speakers: int):
        super().__init__()
        self.encoder = SpeakerEncoder(cfg)
        self.head = AAMSoftmax(cfg["embedding_dim"], num_speakers, cfg["aam_scale"], cfg["aam_margin"])

    def forward(self, waveform: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(waveform)
        return self.head(emb, labels), emb

    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(waveform), dim=-1)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class DisentanglerNet(nn.Module):
    def __init__(self, cfg: dict, num_speakers: int):
        super().__init__()
        total_dim = cfg["speaker_code_dim"] + cfg["env_code_dim"]
        self.encoder = nn.Sequential(
            nn.Linear(cfg["embedding_dim"], cfg["dis_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(cfg["dis_hidden_dim"], total_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(total_dim, cfg["dis_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(cfg["dis_hidden_dim"], cfg["embedding_dim"]),
        )
        self.speaker_classifier = nn.Linear(cfg["speaker_code_dim"], num_speakers)
        self.env_classifier = nn.Linear(cfg["env_code_dim"], len(ENVIRONMENTS))
        self.adv_env_classifier = nn.Linear(cfg["speaker_code_dim"], len(ENVIRONMENTS))
        self.speaker_code_dim = cfg["speaker_code_dim"]

    def forward(self, embedding: torch.Tensor, grl_lambda: float = 1.0) -> dict:
        code = self.encoder(embedding)
        speaker_code = code[:, : self.speaker_code_dim]
        env_code = code[:, self.speaker_code_dim :]
        recon = self.decoder(code)
        return {
            "speaker_code": speaker_code,
            "env_code": env_code,
            "reconstruction": recon,
            "speaker_logits": self.speaker_classifier(speaker_code),
            "env_logits": self.env_classifier(env_code),
            "adv_env_logits": self.adv_env_classifier(grad_reverse(speaker_code, grl_lambda)),
        }

    def embed(self, embedding: torch.Tensor) -> torch.Tensor:
        out = self.forward(embedding, grl_lambda=0.0)
        return F.normalize(out["speaker_code"], dim=-1)


def pearson_correlation_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    a = a / a.std(dim=0, keepdim=True).clamp_min(1e-6)
    b = b / b.std(dim=0, keepdim=True).clamp_min(1e-6)
    corr = torch.matmul(a.transpose(0, 1), b) / max(1, a.shape[0] - 1)
    return corr.abs().mean()


def same_utterance_invariance_loss(code_a: torch.Tensor, code_b: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(code_a, code_b, dim=-1)).mean()


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    embeddings = F.normalize(embeddings, dim=-1)
    logits = embeddings @ embeddings.transpose(0, 1) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    mask = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(mask, float("-inf"))
    positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~mask
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_prob = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob))
    denom = positive_mask.sum(dim=1).clamp_min(1)
    loss = -(positive_log_prob.sum(dim=1) / denom)
    valid = positive_mask.sum(dim=1) > 0
    return loss[valid].mean() if valid.any() else torch.tensor(0.0, device=embeddings.device)


def compute_eer(same_scores: List[float], diff_scores: List[float]) -> float:
    scores = [(s, 1) for s in same_scores] + [(s, 0) for s in diff_scores]
    scores.sort(key=lambda x: x[0])
    positives = max(1, len(same_scores))
    negatives = max(1, len(diff_scores))
    fn = len(same_scores)
    fp = 0
    best_gap = 1.0
    best_eer = 1.0
    for _, label in scores:
        if label == 1:
            fn -= 1
        else:
            fp += 1
        frr = fn / positives
        far = fp / negatives
        gap = abs(frr - far)
        if gap < best_gap:
            best_gap = gap
            best_eer = 0.5 * (frr + far)
    return float(best_eer)


def compute_min_dcf(same_scores: List[float], diff_scores: List[float], p_target: float = 0.05) -> float:
    thresholds = sorted(set(same_scores + diff_scores))
    if not thresholds:
        return 1.0
    best = float("inf")
    for threshold in thresholds:
        miss = sum(score < threshold for score in same_scores) / max(1, len(same_scores))
        fa = sum(score >= threshold for score in diff_scores) / max(1, len(diff_scores))
        best = min(best, miss * p_target + fa * (1.0 - p_target))
    return float(best)


def build_eval_items(split_items: List[dict], max_eval_per_speaker: int) -> List[dict]:
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for item in split_items:
        grouped[item["speaker_id"]].append(item)
    eval_items = []
    for _, utts in grouped.items():
        eval_items.extend(sorted(utts, key=lambda x: x["path"])[:max_eval_per_speaker])
    return eval_items


@dataclass
class EvalEmbeddings:
    clean: torch.Tensor
    noise: torch.Tensor
    reverb: torch.Tensor
    bandlimit: torch.Tensor
    speakers: List[int]
    paths: List[str]


def evaluate_condition(enroll: torch.Tensor, probe: torch.Tensor, labels: List[int]) -> dict:
    same_scores: List[float] = []
    diff_scores: List[float] = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                continue
            score = F.cosine_similarity(enroll[i].unsqueeze(0), probe[j].unsqueeze(0)).item()
            if labels[i] == labels[j]:
                same_scores.append(score)
            else:
                diff_scores.append(score)
    return {
        "eer": compute_eer(same_scores, diff_scores),
        "min_dcf": compute_min_dcf(same_scores, diff_scores),
    }


def closed_set_accuracy(enroll: torch.Tensor, probe: torch.Tensor, labels: List[int]) -> float:
    sim = probe @ enroll.transpose(0, 1)
    pred = sim.argmax(dim=1).tolist()
    pred_labels = [labels[idx] for idx in pred]
    return sum(int(x == y) for x, y in zip(pred_labels, labels)) / max(1, len(labels))


def evaluate_embeddings(eval_embeddings: EvalEmbeddings) -> dict:
    labels = eval_embeddings.speakers
    conditions = {
        "clean_clean": eval_embeddings.clean,
        "clean_noise": eval_embeddings.noise,
        "clean_reverb": eval_embeddings.reverb,
        "clean_bandlimit": eval_embeddings.bandlimit,
    }
    output = {}
    for name, probe in conditions.items():
        metrics = evaluate_condition(eval_embeddings.clean, probe, labels)
        output[name] = {
            "eer": metrics["eer"],
            "min_dcf": metrics["min_dcf"],
            "top1": closed_set_accuracy(eval_embeddings.clean, probe, labels),
        }
    return output


class BaselineWrapper(nn.Module):
    def __init__(self, baseline_model: BaselineSpeakerNet):
        super().__init__()
        self.baseline = baseline_model

    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.baseline.embed(waveform)


class RefinedEmbeddingWrapper(nn.Module):
    def __init__(self, baseline_model: BaselineSpeakerNet, disentangler: DisentanglerNet):
        super().__init__()
        self.baseline = baseline_model
        self.disentangler = disentangler

    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        base = self.baseline.embed(waveform)
        return self.disentangler.embed(base)


def extract_embeddings(model: nn.Module, loader: DataLoader, device: torch.device) -> EvalEmbeddings:
    clean_list = []
    noise_list = []
    reverb_list = []
    band_list = []
    speakers_all: List[int] = []
    paths_all: List[str] = []
    model.eval()
    with torch.no_grad():
        for waveforms, speakers, paths in loader:
            waveforms = waveforms.to(device)
            views = [apply_environment(waveforms.clone(), env_id) for env_id in range(len(ENVIRONMENTS))]
            embeddings = [model.embed(view).cpu() for view in views]
            clean_list.append(embeddings[0])
            noise_list.append(embeddings[1])
            reverb_list.append(embeddings[2])
            band_list.append(embeddings[3])
            speakers_all.extend(speakers)
            paths_all.extend(paths)
    return EvalEmbeddings(
        clean=torch.cat(clean_list, dim=0),
        noise=torch.cat(noise_list, dim=0),
        reverb=torch.cat(reverb_list, dim=0),
        bandlimit=torch.cat(band_list, dim=0),
        speakers=speakers_all,
        paths=paths_all,
    )


def evaluate_model(wrapper: nn.Module, eval_loader: DataLoader, device: torch.device) -> dict:
    return evaluate_embeddings(extract_embeddings(wrapper, eval_loader, device))


def baseline_train_epoch(model, loader, optimizer, device) -> dict:
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for waveforms, speakers, _ in loader:
        waveforms = waveforms.to(device)
        speakers = speakers.to(device)
        optimizer.zero_grad()
        logits, _ = model(waveforms, speakers)
        loss = ce(logits, speakers)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * waveforms.shape[0]
        total_correct += (logits.argmax(dim=1) == speakers).sum().item()
        total += waveforms.shape[0]
    return {"loss": total_loss / max(1, total), "acc": total_correct / max(1, total)}


def disentangler_train_epoch(
    baseline_model: BaselineSpeakerNet,
    disentangler: DisentanglerNet,
    loader,
    optimizer,
    device,
    cfg: dict,
    method_name: str,
) -> dict:
    baseline_model.eval()
    disentangler.train()
    ce = nn.CrossEntropyLoss()
    running = defaultdict(float)
    total = 0
    use_supcon = method_name == "improved"
    for view_a, view_b, env_a, env_b, speakers, _, _ in loader:
        view_a = view_a.to(device)
        view_b = view_b.to(device)
        env_a = env_a.to(device)
        env_b = env_b.to(device)
        speakers = speakers.to(device)
        with torch.no_grad():
            emb_a = baseline_model.embed(view_a)
            emb_b = baseline_model.embed(view_b)
        optimizer.zero_grad()
        out_a = disentangler(emb_a, grl_lambda=cfg["grl_lambda"])
        out_b = disentangler(emb_b, grl_lambda=cfg["grl_lambda"])
        speaker_loss = 0.5 * (ce(out_a["speaker_logits"], speakers) + ce(out_b["speaker_logits"], speakers))
        recon_loss = 0.5 * (
            F.mse_loss(out_a["reconstruction"], emb_a) + F.mse_loss(out_b["reconstruction"], emb_b)
        )
        env_loss = 0.5 * (ce(out_a["env_logits"], env_a) + ce(out_b["env_logits"], env_b))
        adv_loss = 0.5 * (ce(out_a["adv_env_logits"], env_a) + ce(out_b["adv_env_logits"], env_b))
        corr_loss = 0.5 * (
            pearson_correlation_loss(out_a["speaker_code"], out_a["env_code"])
            + pearson_correlation_loss(out_b["speaker_code"], out_b["env_code"])
        )
        invariance_loss = same_utterance_invariance_loss(out_a["speaker_code"], out_b["speaker_code"])
        loss = (
            cfg["lambda_speaker"] * speaker_loss
            + cfg["lambda_recon"] * recon_loss
            + cfg["lambda_env"] * env_loss
            + cfg["lambda_adv"] * adv_loss
            + cfg["lambda_corr"] * corr_loss
            + cfg["lambda_invariance"] * invariance_loss
        )
        supcon_loss = torch.tensor(0.0, device=device)
        if use_supcon:
            all_codes = torch.cat([out_a["speaker_code"], out_b["speaker_code"]], dim=0)
            all_labels = torch.cat([speakers, speakers], dim=0)
            supcon_loss = supervised_contrastive_loss(all_codes, all_labels, cfg["supcon_temperature"])
            loss = loss + cfg["lambda_supcon"] * supcon_loss
        loss.backward()
        optimizer.step()
        batch = speakers.shape[0]
        total += batch
        for key, value in {
            "loss": loss,
            "speaker_loss": speaker_loss,
            "recon_loss": recon_loss,
            "env_loss": env_loss,
            "adv_loss": adv_loss,
            "corr_loss": corr_loss,
            "invariance_loss": invariance_loss,
            "supcon_loss": supcon_loss,
        }.items():
            running[key] += value.item() * batch
    return {key: value / max(1, total) for key, value in running.items()}


def build_dataloaders(cfg: dict, download: bool) -> Tuple[dict, Dict[int, int]]:
    data_root = Path(cfg["data_root"])
    items = index_librispeech(data_root, cfg["subsets"], download)
    splits = build_speaker_splits(items, cfg)
    speaker_to_idx = {speaker_id: idx for idx, speaker_id in enumerate(splits["speaker_ids"])}
    return {
        "baseline_train": DataLoader(
            BaselineTrainDataset(splits["train"], speaker_to_idx, cfg),
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            collate_fn=collate_waveforms,
        ),
        "disentangler_train": DataLoader(
            DisentanglerTrainDataset(splits["train"], speaker_to_idx, cfg),
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            collate_fn=collate_disentangler,
        ),
        "eval": DataLoader(
            EvaluationUtteranceDataset(build_eval_items(splits["test"], cfg["eval_utterances_per_speaker"]), cfg),
            batch_size=cfg["eval_batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            collate_fn=collate_eval,
        ),
        "splits": splits,
    }, speaker_to_idx


def flatten_metrics(metrics: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            out.update(flatten_metrics(value, f"{prefix}{key}_"))
        else:
            out[f"{prefix}{key}"] = value
    return out


def save_metrics_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            values = []
            for key in keys:
                value = row[key]
                values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
            f.write(",".join(values) + "\n")


def plot_condition_bars(summary_rows: List[dict], out_path: Path) -> None:
    methods = [row["method"] for row in summary_rows]
    conditions = ["clean_clean", "clean_noise", "clean_reverb", "clean_bandlimit"]
    x = np.arange(len(methods))
    width = 0.18
    plt.figure(figsize=(10, 4.5))
    for idx, condition in enumerate(conditions):
        values = [row[f"{condition}_eer"] for row in summary_rows]
        plt.bar(x + (idx - 1.5) * width, values, width=width, label=condition)
    plt.xticks(x, methods)
    plt.ylabel("EER")
    plt.title("Verification EER by Condition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_top1_bars(summary_rows: List[dict], out_path: Path) -> None:
    methods = [row["method"] for row in summary_rows]
    conditions = ["clean_clean", "clean_noise", "clean_reverb", "clean_bandlimit"]
    x = np.arange(len(methods))
    width = 0.18
    plt.figure(figsize=(10, 4.5))
    for idx, condition in enumerate(conditions):
        values = [row[f"{condition}_top1"] for row in summary_rows]
        plt.bar(x + (idx - 1.5) * width, values, width=width, label=condition)
    plt.xticks(x, methods)
    plt.ylabel("Top-1 Accuracy")
    plt.title("Identification Accuracy by Condition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def pca_project(embeddings: torch.Tensor, n_components: int = 2) -> torch.Tensor:
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(centered, q=min(centered.shape[0] - 1, centered.shape[1], 8))
    return centered @ v[:, :n_components]


def plot_embedding_projection(embeddings: EvalEmbeddings, out_path: Path, title: str, max_points: int = 100) -> None:
    count = min(max_points, embeddings.clean.shape[0])
    clean = embeddings.clean[:count]
    noise = embeddings.noise[:count]
    projected = pca_project(torch.cat([clean, noise], dim=0))
    speakers = embeddings.speakers[:count]
    plt.figure(figsize=(7, 6))
    for index, speaker in enumerate(sorted(set(speakers))[:12]):
        selected = [i for i, spk in enumerate(speakers) if spk == speaker]
        if not selected:
            continue
        plt.scatter(projected[selected, 0], projected[selected, 1], s=18, label=f"spk {speaker}" if index < 8 else None)
        plt.scatter(
            projected[[i + count for i in selected], 0],
            projected[[i + count for i in selected], 1],
            s=18,
            marker="x",
        )
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
