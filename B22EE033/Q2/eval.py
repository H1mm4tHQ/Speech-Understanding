import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="Assignment_1/Q2/results")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    summary = []
    for method_dir in sorted([p for p in results_root.iterdir() if p.is_dir()]):
        metrics_path = method_dir / f"{method_dir.name}_metrics.json"
        ckpt_path = method_dir / f"{method_dir.name}_checkpoint.pt"
        if not metrics_path.exists():
            continue
        history = load_metrics(metrics_path)
        best = min(history, key=lambda x: x["mismatch_eer"])
        summary.append(
            {
                "method": method_dir.name,
                "best_epoch": best["epoch"],
                "clean_eer": best["clean_eer"],
                "clean_min_dcf": best["clean_min_dcf"],
                "mismatch_eer": best["mismatch_eer"],
                "mismatch_min_dcf": best["mismatch_min_dcf"],
                "checkpoint": str(ckpt_path),
            }
        )

    if not summary:
        raise FileNotFoundError("No method metric files found. Run train.py first.")

    with open(results_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(results_root / "summary.csv", "w", encoding="utf-8") as f:
        f.write("method,best_epoch,clean_eer,clean_min_dcf,mismatch_eer,mismatch_min_dcf,checkpoint\n")
        for row in summary:
            f.write(
                f"{row['method']},{row['best_epoch']},{row['clean_eer']:.4f},{row['clean_min_dcf']:.4f},"
                f"{row['mismatch_eer']:.4f},{row['mismatch_min_dcf']:.4f},{row['checkpoint']}\n"
            )

    methods = [row["method"] for row in summary]
    mismatch_eer = [row["mismatch_eer"] for row in summary]
    clean_eer = [row["clean_eer"] for row in summary]

    plt.figure(figsize=(7, 4))
    x = range(len(methods))
    plt.bar([i - 0.15 for i in x], clean_eer, width=0.3, label="Clean EER")
    plt.bar([i + 0.15 for i in x], mismatch_eer, width=0.3, label="Mismatch EER")
    plt.xticks(list(x), methods)
    plt.ylabel("EER")
    plt.title("Baseline vs Disentanglement Variants")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_root / "comparison_barplot.png")
    plt.close()

    clean_min_dcf = [row["clean_min_dcf"] for row in summary]
    mismatch_min_dcf = [row["mismatch_min_dcf"] for row in summary]

    plt.figure(figsize=(7, 4))
    plt.bar([i - 0.15 for i in x], clean_min_dcf, width=0.3, label="Clean minDCF")
    plt.bar([i + 0.15 for i in x], mismatch_min_dcf, width=0.3, label="Mismatch minDCF")
    plt.xticks(list(x), methods)
    plt.ylabel("minDCF")
    plt.title("Baseline vs Disentanglement Variants (minDCF)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_root / "min_dcf_barplot.png")
    plt.close()

    print("Evaluation artifacts written to", results_root)
    for row in summary:
        print(
            f"{row['method']}: clean_eer={row['clean_eer']:.4f}, "
            f"mismatch_eer={row['mismatch_eer']:.4f}, checkpoint={row['checkpoint']}"
        )


if __name__ == "__main__":
    main()
