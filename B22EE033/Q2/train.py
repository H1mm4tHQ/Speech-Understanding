import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_speaker_id(path: str) -> str:
    stem = Path(path).stem
    return stem.split("_nohash_")[0]


def ensure_speechcommands_download(root: Path) -> Path:
    extracted = root / "SpeechCommands" / "speech_commands_v0.02"
    if extracted.exists():
        return extracted
    torchaudio.datasets.SPEECHCOMMANDS(str(root), download=True)
    return extracted


def index_speechcommands(root: Path):
    extracted = ensure_speechcommands_download(root)
    items = []
    for wav_path in extracted.rglob("*.wav"):
        if "_background_noise_" in wav_path.parts:
            continue
        label = wav_path.parent.name
        speaker_id = parse_speaker_id(wav_path.name)
        items.append(
            {
                "path": str(wav_path),
                "label": label,
                "speaker_id": speaker_id,
            }
        )
    return items


def select_speakers(items, num_speakers, max_train_per_spk, max_test_per_spk):
    counts = Counter(x["speaker_id"] for x in items)
    valid_speakers = [spk for spk, cnt in counts.items() if cnt >= (max_train_per_spk + max_test_per_spk)]
    selected = sorted(valid_speakers, key=lambda s: (-counts[s], s))[:num_speakers]
    return selected


def build_split(items, speakers, max_per_speaker, start_idx=0):
    grouped = defaultdict(list)
    for item in items:
        if item["speaker_id"] in speakers:
            grouped[item["speaker_id"]].append(item)
    out = []
    for spk in speakers:
        spk_items = sorted(grouped[spk], key=lambda x: x["path"])
        out.extend(spk_items[start_idx:start_idx + max_per_speaker])
    return out


def apply_environment(waveform, env_id):
    if env_id == 0:
        return waveform
    if env_id == 1:
        noise = torch.randn_like(waveform) * 0.02
        return torch.clamp(waveform + noise, -1.0, 1.0)
    kernel_len = 1600
    decay = torch.exp(-torch.linspace(0, 4, kernel_len))
    rir = decay.unsqueeze(0)
    out = F.conv1d(waveform.unsqueeze(0), rir.unsqueeze(1), padding=kernel_len - 1).squeeze(0)
    return out[:, : waveform.shape[1]]


class SpeakerEnvDataset(Dataset):
    def __init__(self, items, speaker_to_idx, sample_rate=16000, duration_sec=1.0, eval_mode=False):
        self.items = items
        self.speaker_to_idx = speaker_to_idx
        self.num_samples = int(sample_rate * duration_sec)
        self.eval_mode = eval_mode
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=40
        )

    def __len__(self):
        return len(self.items)

    def _prepare_waveform(self, waveform):
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.shape[1] < self.num_samples:
            waveform = F.pad(waveform, (0, self.num_samples - waveform.shape[1]))
        return waveform[:, : self.num_samples]

    def _to_feature(self, waveform):
        feat = self.melspec(waveform).squeeze(0).transpose(0, 1)
        return torch.log(feat + 1e-6)

    def __getitem__(self, idx):
        item = self.items[idx]
        waveform, sr = torchaudio.load(item["path"])
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        base = self._prepare_waveform(waveform)
        if self.eval_mode:
            clean = self._to_feature(apply_environment(base, 0))
            noisy = self._to_feature(apply_environment(base, 1))
            return {
                "clean": clean,
                "noisy": noisy,
                "speaker_idx": self.speaker_to_idx[item["speaker_id"]],
                "speaker_id": item["speaker_id"],
            }
        env_id = random.choice([0, 1, 2])
        aug = self._to_feature(apply_environment(base, env_id))
        return {
            "feature": aug,
            "speaker_idx": self.speaker_to_idx[item["speaker_id"]],
            "env_id": env_id,
        }


def collate_train(batch):
    feats = [x["feature"] for x in batch]
    speakers = torch.tensor([x["speaker_idx"] for x in batch], dtype=torch.long)
    envs = torch.tensor([x["env_id"] for x in batch], dtype=torch.long)
    lengths = torch.tensor([x.shape[0] for x in feats], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(feats, batch_first=True)
    return padded, lengths, speakers, envs


def collate_eval(batch):
    clean = nn.utils.rnn.pad_sequence([x["clean"] for x in batch], batch_first=True)
    noisy = nn.utils.rnn.pad_sequence([x["noisy"] for x in batch], batch_first=True)
    speakers = torch.tensor([x["speaker_idx"] for x in batch], dtype=torch.long)
    speaker_ids = [x["speaker_id"] for x in batch]
    return clean, noisy, speakers, speaker_ids


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class FeatureEncoder(nn.Module):
    def __init__(self, n_mels=40, embedding_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.proj = nn.Linear(64, embedding_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.mean(dim=-1)
        return self.proj(x)


class BaselineModel(nn.Module):
    def __init__(self, embedding_dim, num_speakers):
        super().__init__()
        self.encoder = FeatureEncoder(40, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_speakers)

    def forward(self, x):
        emb = self.encoder(x)
        return emb, self.classifier(emb)

    def embed(self, x):
        emb = self.encoder(x)
        return F.normalize(emb, dim=-1)


class DisentangleModel(nn.Module):
    def __init__(self, embedding_dim, latent_dim, speaker_dim, env_dim, num_speakers, num_envs=3):
        super().__init__()
        self.encoder = FeatureEncoder(40, embedding_dim)
        self.ae_encoder = nn.Sequential(
            nn.Linear(embedding_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.spk_head = nn.Linear(speaker_dim, num_speakers)
        self.env_head = nn.Linear(env_dim, num_envs)
        self.adv_env_head = nn.Linear(speaker_dim, num_envs)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, embedding_dim),
        )
        self.speaker_dim = speaker_dim
        self.env_dim = env_dim

    def forward(self, x, adv_lambda=1.0):
        emb = self.encoder(x)
        z = self.ae_encoder(emb)
        spk = z[:, : self.speaker_dim]
        env = z[:, self.speaker_dim : self.speaker_dim + self.env_dim]
        recon = self.decoder(torch.cat([spk, env], dim=-1))
        spk_logits = self.spk_head(spk)
        env_logits = self.env_head(env)
        adv_logits = self.adv_env_head(grad_reverse(spk, adv_lambda))
        return {
            "embedding": emb,
            "speaker_code": spk,
            "env_code": env,
            "recon": recon,
            "spk_logits": spk_logits,
            "env_logits": env_logits,
            "adv_logits": adv_logits,
        }

    def embed(self, x):
        out = self.forward(x, adv_lambda=0.0)
        return F.normalize(out["speaker_code"], dim=-1)


def consistency_loss(codes, speaker_labels):
    losses = []
    for spk in speaker_labels.unique():
        idx = (speaker_labels == spk).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() > 1:
            group = codes[idx]
            center = group.mean(dim=0, keepdim=True)
            losses.append(1 - F.cosine_similarity(group, center, dim=-1).mean())
    if not losses:
        return torch.tensor(0.0, device=codes.device)
    return torch.stack(losses).mean()


def orthogonality_loss(spk_code, env_code):
    spk_n = F.normalize(spk_code, dim=-1)
    env_n = F.normalize(env_code, dim=-1)
    cross = torch.matmul(spk_n.transpose(0, 1), env_n) / max(1, spk_n.shape[0])
    return cross.abs().mean()


def cosine_scores(embeddings, labels):
    same, diff = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            score = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
            if labels[i] == labels[j]:
                same.append(score)
            else:
                diff.append(score)
    return same, diff


def compute_eer(same_scores, diff_scores):
    scores = [(s, 1) for s in same_scores] + [(s, 0) for s in diff_scores]
    scores.sort(key=lambda x: x[0])
    positives = len(same_scores)
    negatives = len(diff_scores)
    fn = positives
    fp = 0
    best_gap = 1.0
    best_eer = 1.0
    for _, label in scores:
        if label == 1:
            fn -= 1
        else:
            fp += 1
        frr = fn / max(1, positives)
        far = fp / max(1, negatives)
        gap = abs(frr - far)
        if gap < best_gap:
            best_gap = gap
            best_eer = 0.5 * (frr + far)
    return best_eer


def compute_min_dcf(same_scores, diff_scores, p_target=0.05, c_miss=1.0, c_fa=1.0):
    thresholds = sorted(set(same_scores + diff_scores))
    best = float("inf")
    for th in thresholds:
        miss = sum(s < th for s in same_scores) / max(1, len(same_scores))
        fa = sum(s >= th for s in diff_scores) / max(1, len(diff_scores))
        cost = c_miss * miss * p_target + c_fa * fa * (1 - p_target)
        best = min(best, cost)
    return best


def evaluate_model(model, loader, device):
    model.eval()
    clean_embs, noisy_embs, labels = [], [], []
    with torch.no_grad():
        for clean, noisy, speakers, _ in loader:
            clean = clean.to(device)
            noisy = noisy.to(device)
            clean_embs.append(model.embed(clean).cpu())
            noisy_embs.append(model.embed(noisy).cpu())
            labels.extend(speakers.tolist())
    clean_embs = torch.cat(clean_embs, dim=0)
    noisy_embs = torch.cat(noisy_embs, dim=0)

    clean_same, clean_diff = cosine_scores(clean_embs, labels)
    mismatch_same, mismatch_diff = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            score = F.cosine_similarity(clean_embs[i].unsqueeze(0), noisy_embs[j].unsqueeze(0)).item()
            if labels[i] == labels[j]:
                mismatch_same.append(score)
            else:
                mismatch_diff.append(score)

    return {
        "clean_eer": compute_eer(clean_same, clean_diff),
        "clean_min_dcf": compute_min_dcf(clean_same, clean_diff),
        "mismatch_eer": compute_eer(mismatch_same, mismatch_diff),
        "mismatch_min_dcf": compute_min_dcf(mismatch_same, mismatch_diff),
    }


def train_one_method(method, cfg, train_loader, test_loader, num_speakers, device, out_dir):
    if method == "baseline":
        model = BaselineModel(cfg["embedding_dim"], num_speakers).to(device)
    else:
        model = DisentangleModel(
            cfg["embedding_dim"],
            cfg["latent_dim"],
            cfg["speaker_dim"],
            cfg["env_dim"],
            num_speakers,
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    ce_loss = nn.CrossEntropyLoss()
    train_history = []

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        count = 0
        for feats, _, speakers, envs in train_loader:
            feats = feats.to(device)
            speakers = speakers.to(device)
            envs = envs.to(device)
            optimizer.zero_grad()

            if method == "baseline":
                _, spk_logits = model(feats)
                loss = ce_loss(spk_logits, speakers)
            else:
                out = model(feats, adv_lambda=1.0)
                spk_loss = ce_loss(out["spk_logits"], speakers)
                recon_loss = F.l1_loss(out["recon"], out["embedding"].detach())
                env_loss = ce_loss(out["env_logits"], envs)
                adv_loss = ce_loss(out["adv_logits"], envs)
                cons_loss = consistency_loss(out["speaker_code"], speakers)
                ortho = orthogonality_loss(out["speaker_code"], out["env_code"])
                loss = (
                    spk_loss
                    + cfg["lambda_recon"] * recon_loss
                    + cfg["lambda_env"] * env_loss
                    + cfg["lambda_adv"] * adv_loss
                    + cfg["lambda_consistency"] * cons_loss
                    + cfg["lambda_ortho"] * ortho
                )

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            count += 1

        metrics = evaluate_model(model, test_loader, device)
        metrics["train_loss"] = epoch_loss / max(1, count)
        metrics["epoch"] = epoch
        train_history.append(metrics)
        print(
            f"[{method}] epoch={epoch} loss={metrics['train_loss']:.4f} "
            f"clean_eer={metrics['clean_eer']:.4f} mismatch_eer={metrics['mismatch_eer']:.4f}"
        )

    checkpoint_path = out_dir / f"{method}_checkpoint.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg, "method": method}, checkpoint_path)
    with open(out_dir / f"{method}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(train_history, f, indent=2)
    return train_history, checkpoint_path


def plot_results(histories, out_dir):
    plt.figure(figsize=(7, 4))
    for method, hist in histories.items():
        plt.plot([x["epoch"] for x in hist], [x["mismatch_eer"] for x in hist], marker="o", label=method)
    plt.xlabel("Epoch")
    plt.ylabel("Mismatch EER")
    plt.title("Speaker Verification Under Environment Mismatch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mismatch_eer_plot.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Assignment_1/Q2/configs/base.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    data_root = Path(cfg["data_root"])
    results_root = Path(cfg["results_root"])
    data_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    print("Indexing Speech Commands dataset...")
    all_items = index_speechcommands(data_root)
    speakers = select_speakers(
        all_items,
        cfg["num_speakers"],
        cfg["max_train_utts_per_speaker"],
        cfg["max_test_utts_per_speaker"],
    )
    speaker_to_idx = {spk: idx for idx, spk in enumerate(speakers)}
    train_split = build_split(all_items, speakers, cfg["max_train_utts_per_speaker"], start_idx=0)
    test_split = build_split(
        all_items,
        speakers,
        cfg["max_test_utts_per_speaker"],
        start_idx=cfg["max_train_utts_per_speaker"],
    )

    train_ds = SpeakerEnvDataset(train_split, speaker_to_idx, eval_mode=False)
    test_ds = SpeakerEnvDataset(test_split, speaker_to_idx, eval_mode=True)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_train)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_eval)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    histories = {}
    ckpts = {}

    for method in cfg["methods"]:
        out_dir = results_root / method
        out_dir.mkdir(parents=True, exist_ok=True)
        history, ckpt = train_one_method(method, cfg, train_loader, test_loader, len(speakers), device, out_dir)
        histories[method] = history
        ckpts[method] = str(ckpt)

    plot_results(histories, results_root)

    summary_rows = []
    for method, history in histories.items():
        best = min(history, key=lambda x: x["mismatch_eer"])
        summary_rows.append(
            {
                "method": method,
                "best_epoch": best["epoch"],
                "clean_eer": best["clean_eer"],
                "clean_min_dcf": best["clean_min_dcf"],
                "mismatch_eer": best["mismatch_eer"],
                "mismatch_min_dcf": best["mismatch_min_dcf"],
                "checkpoint": ckpts[method],
            }
        )

    with open(results_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    with open(results_root / "summary.csv", "w", encoding="utf-8") as f:
        f.write("method,best_epoch,clean_eer,clean_min_dcf,mismatch_eer,mismatch_min_dcf,checkpoint\n")
        for row in summary_rows:
            f.write(
                f"{row['method']},{row['best_epoch']},{row['clean_eer']:.4f},{row['clean_min_dcf']:.4f},"
                f"{row['mismatch_eer']:.4f},{row['mismatch_min_dcf']:.4f},{row['checkpoint']}\n"
            )

    print("Training complete. Results saved to", results_root)


if __name__ == "__main__":
    main()
