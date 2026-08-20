#!/usr/bin/env python
"""Plot per-epoch train/validation loss curves for the NIST'23 pilot models.

Two modes:
  # replot from the committed data file (default)
  python plot_loss_curves.py

  # re-extract from the training stdout logs, refresh the TSV, then replot
  python plot_loss_curves.py --logs /path/to/logdir
      # expects <logdir>/{massformer,iceberg,glacier}_pilot.log

Losses are parsed from the training stdout:
  val   : "Epoch N, step M-- val_loss : X"                       (per epoch)
  train : "Epoch N, step M-- train_loss : X"                     (per step; averaged per epoch)
          or "train_loss_epoch : X"                              (per epoch; ICEBERG inten)

`pilot_val_loss.tsv` (model, epoch, split, loss) is small and version-controlled so the
figure reproduces without the multi-MB logs.

Caveats: MassFormer / ICEBERG(inten) losses are 1-cos on the binned spectrum, so the
right axis (1-loss) tracks the held-out cos@100. GLACIER's *val* loss is the hungarian
intensity cos loss, but its *train* loss is the full joint objective (intensity + 0.1*frag,
magma-blended) — a different quantity, so GLACIER train and val are not the same scale.
ICEBERG shows only the stage-2 (inten) losses.
"""
import argparse
import re
from collections import defaultdict
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
FINAL_COS = {"MassFormer (1-cos)": 0.4954, "ICEBERG inten (1-cos)": 0.6730,
             "GLACIER joint (ongoing)": None}

_VAL = re.compile(r"Epoch (\d+), step \d+-- val_loss : ([0-9.]+)")
_TRAIN_STEP = re.compile(r"Epoch (\d+), step \d+-- train_loss : ([0-9.]+)")
_TRAIN_EPOCH_PREF = re.compile(r"Epoch (\d+), step \d+-- train_loss_epoch : ([0-9.]+)")
_TRAIN_EPOCH_BARE = re.compile(r"train_loss_epoch : ([0-9.]+)")


def extract_from_logs(logdir: Path) -> dict:
    """label -> {'val': {ep: loss}, 'train': {ep: loss}}."""
    out = {}
    for label, (basename, _c, _m) in MODELS.items():
        path = logdir / basename
        val, train_step = {}, defaultdict(list)
        train_epoch_pref, train_epoch_bare = {}, []
        if path.exists():
            for line in open(path, errors="ignore"):
                if "val_loss :" in line:
                    for ep, x in _VAL.findall(line):
                        val[int(ep)] = float(x)
                if "train_loss :" in line:
                    for ep, x in _TRAIN_STEP.findall(line):
                        train_step[int(ep)].append(float(x))
                if "train_loss_epoch :" in line:
                    m = _TRAIN_EPOCH_PREF.findall(line)
                    if m:
                        for ep, x in m:
                            train_epoch_pref[int(ep)] = float(x)
                    else:
                        for x in _TRAIN_EPOCH_BARE.findall(line):
                            train_epoch_bare.append(float(x))
        # resolve one train value per epoch
        if train_epoch_pref:
            train = train_epoch_pref
        elif train_step:
            train = {ep: sum(v) / len(v) for ep, v in train_step.items()}
        elif train_epoch_bare and val:
            eps = sorted(val)
            n = min(len(eps), len(train_epoch_bare))
            train = {eps[i]: train_epoch_bare[i] for i in range(n)}
        else:
            train = {}
        out[label] = {"val": val, "train": train}
    return out


def write_tsv(series: dict, tsv: Path):
    with open(tsv, "w") as fp:
        fp.write("model\tepoch\tsplit\tloss\n")
        for label, d in series.items():
            for split in ("train", "val"):
                for ep in sorted(d.get(split, {})):
                    fp.write(f"{label}\t{ep}\t{split}\t{d[split][ep]:.6f}\n")


def read_tsv(tsv: Path) -> dict:
    series = {label: {"train": {}, "val": {}} for label in MODELS}
    with open(tsv) as fp:
        header = next(fp).rstrip("\n").split("\t")
        for line in fp:
            row = line.rstrip("\n").split("\t")
            if len(row) == 4:  # model, epoch, split, loss
                label, ep, split, loss = row
            else:              # legacy: model, epoch, val_loss
                label, ep, loss = row
                split = "val"
            series.setdefault(label, {"train": {}, "val": {}})[split][int(ep)] = float(loss)
    return series


def plot(series: dict, png: Path):
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for label, (_basename, color, marker) in MODELS.items():
        d = series.get(label, {})
        val = d.get("val", {})
        train = d.get("train", {})
        if val:
            xs = sorted(val)
            ys = [val[e] for e in xs]
            ax.plot(xs, ys, marker=marker, color=color, linewidth=2, markersize=5,
                    label=f"{label} — val")
            ax.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color=color)
        if train:
            xs = sorted(train)
            ys = [train[e] for e in xs]
            ax.plot(xs, ys, marker=marker, color=color, linewidth=1.4, markersize=3,
                    linestyle="--", alpha=0.75, label=f"{label} — train")

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss  (cosine-based, lower = better)")
    ax.set_title("NIST'23 pilot — train (dashed) vs validation (solid) loss per epoch\n"
                 "MassFormer / ICEBERG(inten) = 1-cos;  GLACIER val = hungarian inten cos,  "
                 "GLACIER train = full joint loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=1)

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
