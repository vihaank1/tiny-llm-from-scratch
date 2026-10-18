"""Plot real train/validation metrics from CSV or an existing console log."""
import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_metrics(path):
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as handle:
            rows = [{key: float(row[key]) for key in ("step", "train", "val")}
                    for row in csv.DictReader(handle)]
    else:
        pattern = r"Step\s+(\d+):\s*train=([\d.eE+-]+),\s*val=([\d.eE+-]+)"
        rows = [dict(zip(("step", "train", "val"), map(float, match)))
                for match in re.findall(pattern, path.read_text(encoding="utf-8"))]
    if len(rows) < 2:
        raise ValueError("Need at least two recorded evaluations; no values are interpolated.")
    if any(b["step"] <= a["step"] for a, b in zip(rows, rows[1:])):
        raise ValueError("Use a log containing one run with increasing steps.")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/loss_curve.png"))
    parser.add_argument("--title", default="Training and validation loss")
    args = parser.parse_args()
    rows = read_metrics(args.metrics)
    best = min(rows, key=lambda row: row["val"])
    fig, ax = plt.subplots(figsize=(8.5, 4.6), layout="constrained")
    for field, color, label in [("train", "#2563eb", "Training"), ("val", "#d97706", "Validation")]:
        ax.plot([r["step"] for r in rows], [r[field] for r in rows], marker="o",
                markersize=4, linewidth=2, color=color, label=label)
    ax.axvline(best["step"], color="#64748b", linestyle="--", linewidth=1)
    ax.annotate(f"Best validation: {best['val']:.4f}\nStep {int(best['step']):,}",
                xy=(best["step"], best["val"]), xytext=(12, 32),
                textcoords="offset points", fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "#64748b"})
    ax.set(title=args.title, xlabel="Optimizer updates (logged step)",
           ylabel="Cross-entropy (nats / character)")
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
