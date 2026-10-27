#!/usr/bin/env python3
"""Aggregate declared feature-study variants on one explicit evaluation split.

Screening defaults to validation. Only use ``--split test`` for a frozen final
comparison. Manifest columns are run_id, variant, feature_groups, baseline,
split_id. split_id must be the same opaque split fingerprint for paired runs.
Run metrics must include per-family metrics for the selected split.
"""

from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METRICS = ("mae", "rmse", "r2")
REQUIRED = {"run_id", "variant", "feature_groups", "baseline", "split_id"}


def ci(values, seed=20261026, draws=20000):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    if not len(v):
        return np.nan, np.nan
    sims = v[rng.integers(0, len(v), size=(draws, len(v)))].mean(axis=1)
    return tuple(float(x) for x in np.quantile(sims, [0.025, 0.975]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--expected-families", type=int, default=9)
    p.add_argument("--bootstrap-draws", type=int, default=20000)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with a.manifest.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError(f"manifest requires columns: {sorted(REQUIRED)}")
        manifest = list(reader)
    if not manifest:
        raise ValueError("manifest contains no runs")
    ids = [r["run_id"].strip() for r in manifest]
    if any(not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("run_id values must be non-empty and unique")
    byid = {r["run_id"].strip(): r for r in manifest}
    loaded = {}
    rows = []
    for r in manifest:
        rid = r["run_id"].strip()
        split_id = r["split_id"].strip()
        if not split_id:
            raise ValueError(f"{rid}: split_id is required for paired comparisons")
        path = a.runs_root / rid / "metrics.json"
        m = json.loads(path.read_text())
        # Refuse fallback-to-root structures: they could silently select test metrics.
        selected = m.get(a.split)
        if not isinstance(selected, dict):
            raise ValueError(f"{rid}: metrics.json has no {a.split!r} metrics")
        fam = selected.get("per_family")
        if not isinstance(fam, dict) or not fam:
            raise ValueError(f"{rid}: {a.split} per_family metrics are required")
        if a.expected_families and len(fam) != a.expected_families:
            raise ValueError(
                f"{rid}: expected {a.expected_families} {a.split} families, found {len(fam)}"
            )
        loaded[rid] = (selected, fam, m)
        pooled = selected.get("pooled", {})
        macro = selected.get("macro_family", {})
        rows.append(
            {
                "run_id": rid,
                "variant": r["variant"].strip(),
                "feature_groups": r["feature_groups"].strip(),
                "baseline": r["baseline"].strip(),
                "split_id": split_id,
                "split": a.split,
                "pooled_mae": pooled.get("mae"),
                "pooled_rmse": pooled.get("rmse"),
                "pooled_r2": pooled.get("r2"),
                "macro_family_mae": macro.get("mae"),
                "macro_family_rmse": macro.get("rmse"),
                "macro_family_r2": macro.get("r2"),
                "family_ids": ";".join(sorted(fam)),
            }
        )
    df = pd.DataFrame(rows)
    effects = []
    for r in manifest:
        rid = r["run_id"].strip()
        baseid = r["baseline"].strip()
        if not baseid:
            continue
        if baseid not in byid:
            raise ValueError(f"{rid}: baseline {baseid!r} is absent from manifest")
        if byid[baseid]["split_id"].strip() != r["split_id"].strip():
            raise ValueError(
                f"{rid}: baseline and variant have different split_id values"
            )
        selected, fam, _ = loaded[rid]
        bselected, bfam, _ = loaded[baseid]
        if set(fam) != set(bfam):
            raise ValueError(
                f"{rid}: baseline {baseid} has different {a.split} family IDs"
            )
        item = {
            "run_id": rid,
            "baseline": baseid,
            "feature_groups": r["feature_groups"].strip(),
            "split": a.split,
            "family_count": len(fam),
        }
        for metric in METRICS:
            deltas = np.asarray(
                [float(fam[k][metric]) - float(bfam[k][metric]) for k in sorted(fam)]
            )
            lo, hi = ci(
                deltas,
                seed=20261026 + len(effects) * 11 + METRICS.index(metric),
                draws=a.bootstrap_draws,
            )
            item[f"macro_{metric}_delta"] = float(np.mean(deltas))
            item[f"macro_{metric}_ci95_low"] = lo
            item[f"macro_{metric}_ci95_high"] = hi
            pooled = selected.get("pooled", {}).get(metric)
            basep = bselected.get("pooled", {}).get(metric)
            item[f"pooled_{metric}_delta"] = (
                float(pooled - basep)
                if pooled is not None and basep is not None
                else np.nan
            )
        effects.append(item)
    edf = pd.DataFrame(effects)
    # Add declared-baseline deltas and bootstrap summaries to the compact primary table.
    if not edf.empty:
        df = df.merge(
            edf.drop(columns=["feature_groups", "split", "family_count"]),
            on=["run_id", "baseline"],
            how="left",
        )
    df.to_csv(a.out_dir / "run_metrics.csv", index=False)
    edf.to_csv(a.out_dir / "paired_effects.csv", index=False)
    # Interactions are residuals from additive macro-family RMSE changes, paired to one baseline.
    interactions = []
    for r in manifest:
        groups = [
            x.strip()
            for x in r["feature_groups"].replace(";", ",").split(",")
            if x.strip()
        ]
        if len(groups) < 2 or not r["baseline"].strip():
            continue
        base = r["baseline"].strip()
        idsingle = []
        for group in groups:
            candidates = [
                x
                for x in manifest
                if x["baseline"].strip() == base
                and [
                    z.strip()
                    for z in x["feature_groups"].replace(";", ",").split(",")
                    if z.strip()
                ]
                == [group]
            ]
            if not candidates:
                break
            idsingle.append(candidates[0]["run_id"].strip())
        if len(idsingle) != len(groups):
            continue
        r_fam = loaded[r["run_id"].strip()][1]
        b_fam = loaded[base][1]
        combo = np.asarray(
            [float(r_fam[k]["rmse"]) - float(b_fam[k]["rmse"]) for k in sorted(r_fam)]
        )
        add = np.zeros_like(combo)
        for one in idsingle:
            onefam = loaded[one][1]
            add += np.asarray(
                [
                    float(onefam[k]["rmse"]) - float(b_fam[k]["rmse"])
                    for k in sorted(r_fam)
                ]
            )
        resid = combo - add
        lo, hi = ci(resid, seed=20261026 + len(interactions), draws=a.bootstrap_draws)
        interactions.append(
            {
                "run_id": r["run_id"].strip(),
                "baseline": base,
                "feature_groups": ",".join(groups),
                "macro_rmse_interaction": float(resid.mean()),
                "ci95_low": lo,
                "ci95_high": hi,
                "family_count": len(resid),
            }
        )
    pd.DataFrame(interactions).to_csv(a.out_dir / "interactions.csv", index=False)
    # Figures identify the selected split in both title and axes, including frozen-test mode.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df.pooled_rmse, df.macro_family_rmse)
    for _, r in df.iterrows():
        ax.annotate(
            str(r["variant"]),
            (r.pooled_rmse, r.macro_family_rmse),
            fontsize=8,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.set(
        xlabel=f"{a.split.capitalize()} pooled RMSE",
        ylabel=f"{a.split.capitalize()} macro-family RMSE",
        title=f"Feature study: {a.split} performance",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(a.out_dir / "performance.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not edf.empty:
        z = edf.sort_values("macro_rmse_delta")
        labels = [f"{r.run_id} vs {r.baseline}" for r in z.itertuples()]
        x = z.macro_rmse_delta.to_numpy()
        low = x - z.macro_rmse_ci95_low.to_numpy()
        high = z.macro_rmse_ci95_high.to_numpy() - x
        ax.errorbar(
            x, np.arange(len(z)), xerr=np.vstack([low, high]), fmt="o", capsize=3
        )
        ax.set_yticks(np.arange(len(z)), labels)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel(
            f"{a.split.capitalize()} macro-family RMSE delta (variant − baseline; lower is better)"
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Declare baselines to estimate paired family effects",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
    ax.set_title(f"Paired family bootstrap effects ({a.split}; 95% CI)")
    fig.tight_layout()
    fig.savefig(a.out_dir / "feature_effects.png", dpi=180)
    plt.close(fig)
    print(
        f"wrote validation-screening outputs to {a.out_dir}"
        if a.split == "validation"
        else f"wrote frozen final test outputs to {a.out_dir}"
    )


if __name__ == "__main__":
    main()
