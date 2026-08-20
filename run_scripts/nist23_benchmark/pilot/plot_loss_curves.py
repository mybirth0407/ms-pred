#!/usr/bin/env python
"""Plot per-epoch validation-loss curves for the NIST'23 pilot models.

Two modes:
  # replot from the committed data file (default)
  python plot_loss_curves.py

  # re-extract val_loss from the training stdout logs, refresh the TSV, then replot
  python plot_loss_curves.py --logs /path/to/logdir
      # expects <logdir>/{massformer,iceberg,glacier}_pilot.log containing lines like
      #   "... Epoch 7, step 53695-- val_loss : 0.2991..."

Data file `pilot_val_loss.tsv` (model, epoch, val_loss) is small and version-controlled
so the figure is reproducible without the multi-MB training logs. All three spectrum
losses are cosine-based (1-cos); GLACIER's is the joint (intensity + fragment) val loss,
so the curves are directionally comparable, not identical objectives. ICEBERG is the
stage-2 (inten) loss; its stage-1 (gen) BCE is a different scale and not shown.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
TSV = HERE / "pilot_val_loss.tsv"
PNG = HERE / "pilot_loss_curves.png"

# label -> (log basename, colour, marker)
MODELS = {
    "MassFormer (1-cos)":      ("massformer_pilot.log", "#d95f02", "o"),
    "ICEBERG inten (1-cos)":   ("iceberg_pilot.log",    "#1b9e77", "s"),
    "GLACIER joint (ongoing)": ("glacier_pilot.log",    "#7570b3", "^"),
}
# final held-out cos@100 (val fold, 4,971 specs); None = not finished yet
FINAL_COS = {"MassFormer (1-cos)": 0.4954, "ICEBERG inten (1-cos)": 0.6730,
             "GLACIER joint (ongoing)": None}

_PAT = re.compile(r"Epoch (\d+), step \d+-- val_loss : ([0-9.]+)")


def extract_from_logs(logdir: Path) -> dict:
    """label -> {epoch: val_loss} parsed from the training stdout logs."""
    series = {}
    for label, (basename, _c, _m) in MODELS.items():
        path = logdir / basename
        seen = {}
        if path.exists():
            for line in open(path, errors="ignore"):
                for ep, vl in _PAT.findall(line):
                    seen[int(ep)] = float(vl)  # last write per epoch wins
        series[label] = seen
    return series


def write_tsv(series: dict, tsv: Path):
    with open(tsv, "w") as fp:
        fp.write("model\tepoch\tval_loss\n")
        for label, seen in series.items():
            for ep in sorted(seen):
                fp.write(f"{label}\t{ep}\t{seen[ep]:.6f}\n")


def read_tsv(tsv: Path) -> dict:
    series = {label: {} for label in MODELS}
    with open(tsv) as fp:
        next(fp)
        for line in fp:
            label, ep, vl = line.rstrip("\n").split("\t")
            series.setdefault(label, {})[int(ep)] = float(vl)
    return series


def plot(series: dict, png: Path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, (_basename, color, marker) in MODELS.items():
        seen = series.get(label, {})
        if not seen:
            continue
        xs = sorted(seen)
        ys = [seen[e] for e in xs]
        ax.plot(xs, ys, marker=marker, color=color, label=label, linewidth=2, markersize=5)
        ax.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=color)

    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss  (cosine-based, lower = better)")
    ax.set_title("NIST'23 pilot — validation loss per epoch\n"
                 "MassFormer / ICEBERG(inten) = 1-cos on binned spectrum;  GLACIER = joint val loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ticks = ax.get_yticks()
    ax2.set_yticks(ticks)
    ax2.set_yticklabels([f"{1 - t:.2f}" for t in ticks])
    ax2.set_ylabel("≈ cosine (1 - loss)")

    parts = []
    for k, v in FINAL_COS.items():
        parts.append(f"{k.split()[0]} {v:.4f}" if v is not None else f"{k.split()[0]} pending")
    fig.text(0.5, -0.02, "final held-out cos@100 (val fold):  " + "   ".join(parts),
             ha="center", fontsize=8, style="italic")

    fig.tight_layout()
    fig.savefig(png, dpi=130, bbox_inches="tight")
    print("saved:", png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, default=None,
                    help="dir with {model}_pilot.log; re-extracts and refreshes the TSV")
    ap.add_argument("--tsv", type=Path, default=TSV)
    ap.add_argument("--out", type=Path, default=PNG)
    args = ap.parse_args()

    if args.logs is not None:
        series = extract_from_logs(args.logs)
        write_tsv(series, args.tsv)
        print("wrote", args.tsv)
    else:
        series = read_tsv(args.tsv)
    plot(series, args.out)


if __name__ == "__main__":
    main()
