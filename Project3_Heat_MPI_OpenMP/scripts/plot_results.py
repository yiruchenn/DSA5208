#!/usr/bin/env python3
"""Plot benchmark CSV on any machine with matplotlib installed.

python3 -m pip install matplotlib
python3 scripts/plot_results.py results/strong_scaling.csv
"""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csvfile", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        parser.error("matplotlib is required for plots: python3 -m pip install matplotlib")

    with open(args.csvfile, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows: parser.error("Empty CSV")
    scaling = rows[0]["scaling"]
    if any(row["scaling"] != scaling for row in rows):
        parser.error("Do not mix strong and weak scaling in one CSV")
    groups = {}
    for row in rows:
        if row["kind"] != "parallel": continue
        label = f"{row['mode']}, {row['schedule']}, {row['threads']} threads/rank"
        groups.setdefault(label, []).append(row)
    if not groups: parser.error("No parallel measurements in CSV")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [("median_seconds", "Time per run (s)", "runtime")]
    if scaling == "strong":
        metrics.append(("speedup", "Speedup versus serial", "speedup"))
    metrics.append(("efficiency", "Parallel efficiency" if scaling == "strong"
                    else "Weak scaling efficiency", "efficiency"))
    for metric, ylabel, basename in metrics:
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        for label, data in sorted(groups.items()):
            points = sorted((int(row["total_cores"]), float(row[metric])) for row in data)
            ax.plot([x for x, _ in points], [y for _, y in points], "o-", label=label)
        if scaling == "strong" and metric == "speedup":
            total = sorted({int(r["total_cores"]) for values in groups.values() for r in values})
            ax.plot(total, total, "k--", linewidth=1, label="ideal linear")
        if metric == "efficiency": ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
        ax.set(xlabel="Allocated CPU cores (MPI ranks x OpenMP threads)", ylabel=ylabel,
               title=f"2D Jacobi heat solver: {scaling} scaling")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        target = args.output_dir / f"{scaling}_{basename}.png"
        fig.savefig(target, dpi=200)
        print(f"Saved {target}")
        plt.close(fig)


if __name__ == "__main__":
    main()
