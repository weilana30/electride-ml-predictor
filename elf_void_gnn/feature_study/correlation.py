#!/usr/bin/env python3
"""Training-only Pearson/Spearman feature audit with pooled and balanced summaries.

Pooled node/edge rows are dependent within structures. Structure-balanced
within-group coefficients average one coefficient per structure; sample-summary
coefficients compare structure-level feature means across structures. These are
reported both pooled across structures and family-balanced. Cross-group rows
compare structure summaries rather than pretending unlike atoms, voids, and
edges are paired. All are descriptive associations, not causal importance.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import h5py
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from elf_void_gnn.data import ShardedGraphDataset, read_split_table

METHODS = ("pearson", "spearman")


def update_missingness(stats, group, feature, values, valid_values=None):
    values = np.asarray(values, float).reshape(-1)
    finite = np.isfinite(values)
    key = (group, feature)
    item = stats.setdefault(
        key,
        {"samples": 0, "rows": 0, "finite_rows": 0, "valid_rows": 0, "valid_observations": 0},
    )
    item["samples"] += 1
    item["rows"] += int(len(values))
    item["finite_rows"] += int(finite.sum())
    if valid_values is not None:
        valid = np.asarray(valid_values, float).reshape(-1)
        valid_finite = np.isfinite(valid)
        item["valid_rows"] += int(valid_finite.sum())
        item["valid_observations"] += int(np.nansum(valid))


def pair_corr(x, y, method):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, int(len(x))
    if method == "spearman":
        x = rankdata(x)
        y = rankdata(y)
    return float(np.corrcoef(x, y)[0, 1]), int(len(x))


def names_for(g, key, n, schema_names):
    if key not in g:
        if len(schema_names) != n:
            raise ValueError(
                f"Schema defines {len(schema_names)} names for {n} feature columns"
            )
        return schema_names
    values = g[key][...]
    values = [x.decode() if isinstance(x, bytes) else str(x) for x in values]
    return values if len(values) == n else schema_names


def record(out, group, xname, yname, method, estimator, r, n):
    if np.isfinite(r):
        out.append(
            dict(
                group=group,
                feature_a=xname,
                feature_b=yname,
                statistic=method,
                estimator=estimator,
                correlation=r,
                n_rows=n,
            )
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--max-per-family",
        type=int,
        default=100,
        help=(
            "Analyze a deterministic, hash-selected, family-stratified sample "
            "of at most this many training structures per family (default: 100). "
            "Use 0 for every training structure."
        ),
    )
    a = p.parse_args()
    if a.max_per_family < 0:
        p.error("--max-per-family must be nonnegative")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_split_table(a.dataset_root)["train"]
    all_train_count = len(train_rows)
    if a.max_per_family:
        by_family = {}
        for row in train_rows:
            by_family.setdefault(row["family_id"], []).append(row)
        train_rows = []
        for family_id, rows in sorted(by_family.items()):
            rows = sorted(
                rows,
                key=lambda row: hashlib.sha256(
                    f"elfnet-feature-correlation-v1:{family_id}:{row['sample_id']}".encode()
                ).digest(),
            )
            train_rows.extend(rows[: a.max_per_family])
        train_rows.sort(key=lambda row: row["sample_id"])
    ds = ShardedGraphDataset(a.dataset_root, train_rows)
    schema_path = a.dataset_root / "schema.json"
    if not schema_path.exists():
        schema_path = (
            Path(__file__).resolve().parents[2] / "elf_void_analysis" / "schema.json"
        )
    schema = json.loads(schema_path.read_text())
    node_schema = schema["graph_storage"]["node_features"]
    schema_names = {
        "atom": node_schema["atoms"]["feature_names_in_order"],
        "atom_validity": node_schema["atoms"]["feature_validity_names_in_order"],
        "void": node_schema["voids"]["feature_names_in_order"],
    }
    parts = {}
    sample_rows = {}
    missing_stats = {}
    target_balanced = {}
    target_parts = {}
    for shard, sid, family in ds.items:
        row = {}
        with h5py.File(shard, "r") as h:
            s = h[f"samples/{sid}"]
            for group, base in [("atom", s["atoms"]), ("void", s["voids"])]:
                x = np.asarray(base["features"][...], float)
                nms = names_for(base, "feature_names", x.shape[1], schema_names[group])
                cols = {n: x[:, i] for i, n in enumerate(nms)}
                if group == "atom" and "feature_validity" in base:
                    valid = np.asarray(base["feature_validity"][...], float)
                    vns = names_for(
                        base,
                        "feature_validity_names",
                        valid.shape[1],
                        schema_names["atom_validity"],
                    )
                    for j, n in enumerate(vns):
                        cols[f"validity.{n}"] = valid[:, j]
                        update_missingness(
                            missing_stats, "atom_validity", n, valid[:, j], valid[:, j]
                        )
                for n, v in cols.items():
                    parts.setdefault((group, n), []).append((sid, v))
                    finite = np.isfinite(v)
                    update_missingness(missing_stats, group, n, v)
                    row[f"{group}.{n}"] = (
                        float(np.nanmean(v)) if len(v) and finite.any() else np.nan
                    )
                if len(x):
                    row[f"{group}.__count"] = len(x)
                if group == "void":
                    y = np.asarray(s["targets/elf_at_center"][...], float)
                    target_parts[sid] = y
                    row["target.elf_at_center.mean"] = (
                        float(np.nanmean(y))
                        if len(y) and np.isfinite(y).any()
                        else np.nan
                    )
                    for i, n in enumerate(nms):
                        for method in METHODS:
                            r, nobs = pair_corr(x[:, i], y, method)
                            if np.isfinite(r):
                                target_balanced.setdefault((n, method), []).append(
                                    (r, nobs)
                                )
            for rel in ("atom_atom", "void_atom", "void_void"):
                key = f"{rel}_distance_angstrom"
                if key in s["edges"]:
                    v = np.asarray(s[f"edges/{key}"][...], float).reshape(-1)
                    parts.setdefault((rel, key), []).append((sid, v))
                    z = v[np.isfinite(v)]
                    row[f"edge.{rel}.mean"] = float(np.mean(z)) if len(z) else np.nan
                    row[f"edge.{rel}.std"] = float(np.std(z)) if len(z) else np.nan
                    update_missingness(missing_stats, rel, key, v)
        sample_rows[sid] = row
    # Within-group pooled and sample-balanced within-structure correlations.
    out = []
    for group in sorted({g for g, n in parts}):
        nms = sorted(n for g, n in parts if g == group)
        for i, left in enumerate(nms):
            for right in nms[i + 1 :]:
                pooled_x = np.concatenate([x for _, x in parts[group, left]])
                pooled_y = np.concatenate([x for _, x in parts[group, right]])
                for method in METHODS:
                    r, n = pair_corr(pooled_x, pooled_y, method)
                    record(out, group, left, right, method, "pooled_observations", r, n)
    # Cross-group associations use one row of group summaries per structure. No
    # artificial atom-to-void or edge-to-node row pairing is performed.
    summary_keys = sorted(
        {k for row in sample_rows.values() for k in row if not k.endswith(".__count")}
    )
    sample_ids = [sid for _, sid, _ in ds.items]
    sample_arrays = {
        k: np.asarray([sample_rows[sid].get(k, np.nan) for sid in sample_ids], float)
        for k in summary_keys
    }
    for i, left in enumerate(summary_keys):
        for right in summary_keys[i + 1 :]:
            for method in METHODS:
                r, n = pair_corr(sample_arrays[left], sample_arrays[right], method)
                record(
                    out,
                    "cross_group_sample_summary",
                    left,
                    right,
                    method,
                    "pooled_structure_means",
                    r,
                    n,
                )
    # Family-balanced counterpart: average structure summaries within each family
    # first, so families with more pressure/phase variants do not get extra weight.
    sample_family = {sid: family for _, sid, family in ds.items}
    family_ids = sorted(set(sample_family.values()))
    family_arrays = {
        k: np.asarray(
            [
                np.nanmean(
                    [
                        sample_rows[sid].get(k, np.nan)
                        for sid in sample_ids
                        if sample_family[sid] == family
                    ]
                )
                for family in family_ids
            ],
            float,
        )
        for k in summary_keys
    }
    for i, left in enumerate(summary_keys):
        for right in summary_keys[i + 1 :]:
            for method in METHODS:
                r, n = pair_corr(family_arrays[left], family_arrays[right], method)
                record(
                    out,
                    "cross_group_sample_summary",
                    left,
                    right,
                    method,
                    "family_balanced_family_means",
                    r,
                    n,
                )
    # Mean of per-structure node correlations: equal weight regardless of node count.
    for group in sorted({g for g, n in parts}):
        nms = sorted(n for g, n in parts if g == group)
        for i, left in enumerate(nms):
            for right in nms[i + 1 :]:
                per = {m: [] for m in METHODS}
                # Join by sample id so structures missing a relation never shift alignment.
                right_by_id = dict(parts[group, right])
                for sid, xa in parts[group, left]:
                    if sid not in right_by_id:
                        continue
                    ya = right_by_id[sid]
                    for method in METHODS:
                        r, n = pair_corr(xa, ya, method)
                        if np.isfinite(r):
                            per[method].append((r, n))
                for method, vals in per.items():
                    if vals:
                        record(
                            out,
                            group,
                            left,
                            right,
                            method,
                            "mean_within_structure",
                            float(np.mean([v for v, n in vals])),
                            int(sum(n for v, n in vals)),
                        )
    # Pooled void-node feature/target associations, with pairwise finite filtering.
    for (group, n), entries in parts.items():
        if group != "void":
            continue
        x = np.concatenate([v for sid, v in entries if sid in target_parts])
        y = np.concatenate(
            [target_parts[sid] for sid, v in entries if sid in target_parts]
        )
        for method in METHODS:
            r, nobs = pair_corr(x, y, method)
            record(
                out,
                "void_target",
                n,
                "elf_at_center",
                method,
                "pooled_void_observations",
                r,
                nobs,
            )
    for (n, method), vals in target_balanced.items():
        record(
            out,
            "void_target",
            n,
            "elf_at_center",
            method,
            "mean_within_structure",
            float(np.mean([v for v, _ in vals])),
            int(sum(k for _, k in vals)),
        )
    with (a.out_dir / "correlations.csv").open("w", newline="") as f:
        fields = [
            "group",
            "feature_a",
            "feature_b",
            "statistic",
            "estimator",
            "correlation",
            "n_rows",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    with (a.out_dir / "missingness.csv").open("w", newline="") as f:
        fields = [
            "group",
            "feature",
            "samples",
            "rows",
            "finite_rate",
            "valid_rate",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (group, feature), stats in sorted(missing_stats.items()):
            w.writerow(
                {
                    "group": group,
                    "feature": feature,
                    "samples": stats["samples"],
                    "rows": stats["rows"],
                    "finite_rate": (
                        stats["finite_rows"] / stats["rows"] if stats["rows"] else np.nan
                    ),
                    "valid_rate": (
                        stats["valid_observations"] / stats["valid_rows"]
                        if stats["valid_rows"]
                        else np.nan
                    ),
                }
            )

    import matplotlib.pyplot as plt

    heat_names = summary_keys
    heat = np.eye(len(heat_names), dtype=float)
    for i, left in enumerate(heat_names):
        for j in range(i):
            r, _ = pair_corr(sample_arrays[left], sample_arrays[heat_names[j]], "pearson")
            heat[i, j] = heat[j, i] = r
    fig_size = max(8, min(16, 0.32 * len(heat_names)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(heat, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(heat_names)), heat_names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(heat_names)), heat_names, fontsize=6)
    ax.set_title("Train-only Pearson correlation of structure-level feature means")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    fig.savefig(a.out_dir / "train_feature_correlation_heatmap.png", dpi=180)
    plt.close(fig)
    print(
        f"wrote {len(out)} correlation rows and {len(missing_stats)} missingness summaries "
        f"from {len(ds)} of {all_train_count} training structures to {a.out_dir}"
    )
    (a.out_dir / "analysis_meta.json").write_text(
        json.dumps(
            {
                "source_split": "train",
                "selection": "all" if not a.max_per_family else "deterministic_hash_stratified_by_family",
                "max_per_family": a.max_per_family or None,
                "structures_analyzed": len(ds),
                "training_structures_available": all_train_count,
                "family_count": len({family for _, _, family in ds.items}),
                "statistics": ["Pearson", "Spearman"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
