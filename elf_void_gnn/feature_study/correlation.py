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
import json
import sys
from pathlib import Path
import h5py
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from elf_void_gnn.data import ShardedGraphDataset, read_split_table

METHODS = ("pearson", "spearman")


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
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    ds = ShardedGraphDataset(a.dataset_root, read_split_table(a.dataset_root)["train"])
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
    missing = []
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
                        missing.append(
                            dict(
                                sample_id=sid,
                                family_id=family,
                                group="atom_validity",
                                feature=n,
                                rows=len(valid),
                                finite_rate=(
                                    float(np.isfinite(valid[:, j]).mean())
                                    if len(valid)
                                    else np.nan
                                ),
                                valid_rate=(
                                    float(np.nanmean(valid[:, j]))
                                    if len(valid)
                                    else np.nan
                                ),
                            )
                        )
                for n, v in cols.items():
                    parts.setdefault((group, n), []).append((sid, v))
                    finite = np.isfinite(v)
                    missing.append(
                        dict(
                            sample_id=sid,
                            family_id=family,
                            group=group,
                            feature=n,
                            rows=len(v),
                            finite_rate=float(finite.mean()) if len(v) else np.nan,
                            valid_rate=np.nan,
                        )
                    )
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
                    missing.append(
                        dict(
                            sample_id=sid,
                            family_id=family,
                            group=rel,
                            feature=key,
                            rows=len(v),
                            finite_rate=(
                                float(np.isfinite(v).mean()) if len(v) else np.nan
                            ),
                            valid_rate=np.nan,
                        )
                    )
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
            "sample_id",
            "family_id",
            "group",
            "feature",
            "rows",
            "finite_rate",
            "valid_rate",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(missing)
    print(
        f"wrote {len(out)} correlation rows and {len(missing)} missingness rows to {a.out_dir}"
    )


if __name__ == "__main__":
    main()
