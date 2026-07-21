"""Aggregate NIST'23 spectrum-prediction benchmark results into one table.

Each model's predict step writes ``preds/pred_eval.yaml`` (via
``analysis/spec_pred_eval.py``) containing ``avg_cos_sim``, ``avg_entropy_sim`` and
``avg_coverage`` for the test set. This script collects every such file across all
benchmarked models and both splits into a single TSV + Markdown table.

Usage:
    python analysis/nist23_benchmark_aggregate.py \\
        --results-dir results \\
        --out-tsv results/nist23_benchmark.tsv \\
        --out-md results/nist23_benchmark.md
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

# result-dir prefix  ->  human-readable model name
MODEL_DIRS = {
    "iceberg_nist23": "ICEBERG 2.1",
    "scarf_inten_nist23": "SCARF",
    "marason_nist23": "MARASON",
    "massformer_baseline_nist23": "MassFormer",
    "molnetms_baseline_nist23": "3DMolMS",
    "glacier_nist23": "GLACIER",
}

SPLIT_LABEL = {
    "split_1_rnd1": "random (split_1)",
    "scaffold_1_rnd1": "scaffold (scaffold_1)",
}

METRICS = ["avg_cos_sim", "avg_entropy_sim", "avg_coverage"]


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results", type=Path)
    p.add_argument("--out-tsv", default="results/nist23_benchmark.tsv", type=Path)
    p.add_argument("--out-md", default="results/nist23_benchmark.md", type=Path)
    return p.parse_args()


def main():
    args = get_args()
    rows = []
    for dir_prefix, model_name in MODEL_DIRS.items():
        model_root = args.results_dir / dir_prefix
        if not model_root.is_dir():
            continue
        for split_folder, split_label in SPLIT_LABEL.items():
            eval_file = model_root / split_folder / "preds" / "pred_eval.yaml"
            if not eval_file.exists():
                continue
            data = yaml.safe_load(eval_file.read_text())
            row = {
                "model": model_name,
                "split": split_label,
                "n_test": len(data.get("individuals", [])),
            }
            for m in METRICS:
                row[m] = data.get(m, float("nan"))
                sem_key = m.replace("avg_", "sem_")
                if sem_key in data:
                    row[sem_key] = data[sem_key]
            rows.append(row)

    if not rows:
        print(
            f"No pred_eval.yaml files found under {args.results_dir}/. "
            "Run the model predict steps first (see run_scripts/nist23_benchmark/)."
        )
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["split", "avg_cos_sim"], ascending=[True, False]).reset_index(drop=True)

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_tsv, sep="\t", index=False)

    # Compact Markdown view: cosine / entropy / coverage per model per split.
    disp = df.copy()
    for m in METRICS:
        disp[m] = disp[m].map(lambda v: f"{v:.4f}")
    disp = disp.rename(
        columns={
            "avg_cos_sim": "cosine",
            "avg_entropy_sim": "entropy",
            "avg_coverage": "coverage",
        }
    )
    keep = ["model", "split", "cosine", "entropy", "coverage", "n_test"]
    md = _to_markdown(disp[keep])
    args.out_md.write_text("# NIST'23 spectrum-prediction benchmark\n\n" + md + "\n")

    print(md)
    print(f"\nWrote {args.out_tsv} and {args.out_md}")


def _to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub Markdown table without the tabulate dependency."""
    cols = list(df.columns)
    cells = [[str(v) for v in row] for row in df.to_numpy()]
    widths = [
        max(len(cols[i]), *(len(r[i]) for r in cells)) if cells else len(cols[i])
        for i in range(len(cols))
    ]
    fmt = lambda vals: "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(vals)) + " |"
    lines = [fmt(cols), "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"]
    lines += [fmt(r) for r in cells]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
