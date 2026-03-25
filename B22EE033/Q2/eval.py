import argparse
from pathlib import Path

import torch

from q2_core import (
    BaselineSpeakerNet,
    BaselineWrapper,
    DisentanglerNet,
    RefinedEmbeddingWrapper,
    build_dataloaders,
    ensure_dir,
    evaluate_model,
    extract_embeddings,
    flatten_metrics,
    load_json,
    plot_condition_bars,
    plot_embedding_projection,
    plot_top1_bars,
    save_json,
    save_metrics_csv,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root", default="Q2_modify")
    args = parser.parse_args()

    cfg = load_json(args.config)
    results_root = Path(args.results_root)
    loaders, speaker_to_idx = build_dataloaders(cfg, download=False)
    num_speakers = len(speaker_to_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline_ckpt = torch.load(results_root / "results" / "baseline" / "best.pt", map_location=device)
    baseline = BaselineSpeakerNet(cfg, num_speakers).to(device)
    baseline.load_state_dict(baseline_ckpt["model_state"])
    baseline.eval()
    baseline_wrapper = BaselineWrapper(baseline)

    rows = []
    for method_name in ["baseline", "paper_reproduction", "improved"]:
        if method_name == "baseline":
            wrapper = baseline_wrapper
        else:
            ckpt = torch.load(results_root / "results" / method_name / "best.pt", map_location=device)
            disentangler = DisentanglerNet(cfg, num_speakers).to(device)
            disentangler.load_state_dict(ckpt["model_state"])
            disentangler.eval()
            wrapper = RefinedEmbeddingWrapper(baseline, disentangler)

        metrics = evaluate_model(wrapper, loaders["eval"], device)
        embeddings = extract_embeddings(wrapper, loaders["eval"], device)
        out_dir = ensure_dir(results_root / "results" / method_name)
        plot_embedding_projection(
            embeddings,
            out_dir / "embedding_projection.png",
            f"{method_name} PCA (clean circles, noise crosses)",
        )
        rows.append(
            {
                "method": method_name,
                "checkpoint": str((out_dir / "best.pt").resolve()),
                **flatten_metrics(metrics),
            }
        )

    save_json(results_root / "results" / "summary.json", rows)
    save_metrics_csv(results_root / "results" / "summary.csv", rows)
    plot_condition_bars(rows, results_root / "results" / "eer_by_condition.png")
    plot_top1_bars(rows, results_root / "results" / "top1_by_condition.png")
    print(f"Evaluation summary written to {results_root / 'results'}")


if __name__ == "__main__":
    main()
