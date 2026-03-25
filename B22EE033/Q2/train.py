import argparse

import torch

from q2_core import (
    BaselineSpeakerNet,
    BaselineWrapper,
    DisentanglerNet,
    RefinedEmbeddingWrapper,
    baseline_train_epoch,
    build_dataloaders,
    disentangler_train_epoch,
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
    set_seed
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    cfg = load_json(args.config)
    set_seed(cfg["seed"])
    results_root = ensure_dir(cfg["results_root"])
    ensure_dir(results_root / "results")

    loaders, speaker_to_idx = build_dataloaders(cfg, download=args.download)
    num_speakers = len(speaker_to_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline = BaselineSpeakerNet(cfg, num_speakers).to(device)
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=cfg["baseline_lr"], weight_decay=1e-4)
    baseline_history = []
    best_baseline_state = None
    best_baseline_eer = float("inf")

    for epoch in range(cfg["baseline_epochs"]):
        train_stats = baseline_train_epoch(baseline, loaders["baseline_train"], baseline_optimizer, device)
        metrics = evaluate_model(BaselineWrapper(baseline), loaders["eval"], device)
        row = {"epoch": epoch, **train_stats, **flatten_metrics(metrics)}
        baseline_history.append(row)
        print(
            f"[baseline] epoch={epoch} loss={row['loss']:.4f} acc={row['acc']:.4f} "
            f"clean_noise_eer={row['clean_noise_eer']:.4f}"
        )
        if row["clean_noise_eer"] < best_baseline_eer:
            best_baseline_eer = row["clean_noise_eer"]
            best_baseline_state = {"model_state": baseline.state_dict(), "epoch": epoch, "config": cfg}

    baseline_dir = ensure_dir(results_root / "results" / "baseline")
    torch.save(best_baseline_state, baseline_dir / "best.pt")
    save_json(baseline_dir / "history.json", baseline_history)
    save_metrics_csv(baseline_dir / "history.csv", baseline_history)

    baseline.load_state_dict(best_baseline_state["model_state"])
    baseline.eval()
    for param in baseline.parameters():
        param.requires_grad = False

    summary_rows = []
    baseline_wrapper = BaselineWrapper(baseline)
    baseline_metrics = evaluate_model(baseline_wrapper, loaders["eval"], device)
    baseline_embeddings = extract_embeddings(baseline_wrapper, loaders["eval"], device)
    plot_embedding_projection(
        baseline_embeddings,
        baseline_dir / "embedding_projection.png",
        "Baseline PCA (clean circles, noise crosses)",
    )
    summary_rows.append(
        {
            "method": "baseline",
            "checkpoint": str((baseline_dir / "best.pt").resolve()),
            **flatten_metrics(baseline_metrics),
        }
    )

    for method_name in ["paper_reproduction", "improved"]:
        disentangler = DisentanglerNet(cfg, num_speakers).to(device)
        optimizer = torch.optim.AdamW(disentangler.parameters(), lr=cfg["disentangler_lr"], weight_decay=1e-4)
        history = []
        best_state = None
        best_eer = float("inf")

        for epoch in range(cfg["disentangler_epochs"]):
            train_stats = disentangler_train_epoch(
                baseline,
                disentangler,
                loaders["disentangler_train"],
                optimizer,
                device,
                cfg,
                method_name,
            )
            metrics = evaluate_model(RefinedEmbeddingWrapper(baseline, disentangler), loaders["eval"], device)
            row = {"epoch": epoch, **train_stats, **flatten_metrics(metrics)}
            history.append(row)
            print(
                f"[{method_name}] epoch={epoch} loss={row['loss']:.4f} "
                f"clean_noise_eer={row['clean_noise_eer']:.4f} clean_reverb_eer={row['clean_reverb_eer']:.4f}"
            )
            if row["clean_noise_eer"] < best_eer:
                best_eer = row["clean_noise_eer"]
                best_state = {
                    "model_state": disentangler.state_dict(),
                    "epoch": epoch,
                    "config": cfg,
                    "method": method_name,
                }

        method_dir = ensure_dir(results_root / "results" / method_name)
        torch.save(best_state, method_dir / "best.pt")
        save_json(method_dir / "history.json", history)
        save_metrics_csv(method_dir / "history.csv", history)

        disentangler.load_state_dict(best_state["model_state"])
        wrapper = RefinedEmbeddingWrapper(baseline, disentangler)
        metrics = evaluate_model(wrapper, loaders["eval"], device)
        embeddings = extract_embeddings(wrapper, loaders["eval"], device)
        plot_embedding_projection(
            embeddings,
            method_dir / "embedding_projection.png",
            f"{method_name} PCA (clean circles, noise crosses)",
        )
        summary_rows.append(
            {
                "method": method_name,
                "checkpoint": str((method_dir / "best.pt").resolve()),
                **flatten_metrics(metrics),
            }
        )

    save_json(results_root / "results" / "summary.json", summary_rows)
    save_metrics_csv(results_root / "results" / "summary.csv", summary_rows)
    plot_condition_bars(summary_rows, results_root / "results" / "eer_by_condition.png")
    plot_top1_bars(summary_rows, results_root / "results" / "top1_by_condition.png")
    print(f"Finished. Artifacts saved under {results_root}")


if __name__ == "__main__":
    main()
