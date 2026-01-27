# dataset_docgen_all.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

import pandas as pd


# ----------------------------
# Config
# ----------------------------
INPUT_DIR = os.environ.get("INPUT_DIR", "cleaned")
OUT_DIR = os.environ.get("OUT_DIR", str(Path(INPUT_DIR) / "_docs"))

SAMPLE_ROWS = 5
CHUNK_SIZE = 250_000
TOP_VALUE_COUNTS = 10


# Common join-ish columns to look for in “connections”
COMMON_KEY_CANDIDATES = [
    "timestamp", "time", "date", "datetime",
    "entity", "building_id", "building_key", "all_building",
    "meter_id", "site_id",
    "utility", "resource", "type",
]


def md_table(rows: List[List[str]], headers: List[str]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _try_parse_timestamp_minmax(csv_path: Path, col: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    tmin, tmax = None, None
    try:
        for chunk in pd.read_csv(csv_path, usecols=[col], chunksize=CHUNK_SIZE):
            ts = pd.to_datetime(chunk[col], errors="coerce")
            if ts.notna().any():
                cmin, cmax = ts.min(), ts.max()
                tmin = cmin if tmin is None else min(tmin, cmin)
                tmax = cmax if tmax is None else max(tmax, cmax)
    except Exception:
        return None, None
    return tmin, tmax


def profile_csv(csv_path: Path) -> Dict[str, Any]:
    # quick sample for schema + preview
    sample = pd.read_csv(csv_path, nrows=5000)
    cols = list(sample.columns)

    # row count (fast-ish): count lines
    nrows = None
    try:
        with csv_path.open("rb") as f:
            nrows = max(sum(1 for _ in f) - 1, 0)
    except Exception:
        pass

    dtypes = sample.dtypes.astype(str).to_dict()
    na_counts = sample.isna().sum().to_dict()

    # timestamp detection
    ts_col = None
    for c in ["timestamp", "datetime", "date", "time"]:
        if c in cols:
            ts_col = c
            break

    tmin = tmax = None
    if ts_col:
        tmin, tmax = _try_parse_timestamp_minmax(csv_path, ts_col)

    # key candidates present
    present_candidates = [c for c in COMMON_KEY_CANDIDATES if c in cols]

    # value stats (if present)
    value_stats = None
    if "value" in cols:
        v = pd.to_numeric(sample["value"], errors="coerce")
        if v.notna().any():
            value_stats = {
                "count_nonnull(sample)": int(v.notna().sum()),
                "mean(sample)": float(v.mean()),
                "std(sample)": float(v.std()),
                "min(sample)": float(v.min()),
                "p25(sample)": float(v.quantile(0.25)),
                "median(sample)": float(v.quantile(0.50)),
                "p75(sample)": float(v.quantile(0.75)),
                "max(sample)": float(v.max()),
            }
        else:
            value_stats = {"count_nonnull(sample)": 0}

    # top categories for a few likely categorical cols
    top_categories: Dict[str, List[Tuple[str, int]]] = {}
    for c in ["utility", "all_building", "building_id", "building_key", "entity", "type", "resource"]:
        if c in cols:
            vc = sample[c].astype(str).value_counts(dropna=True).head(TOP_VALUE_COUNTS)
            top_categories[c] = [(k, int(v)) for k, v in vc.items()]

    return {
        "file": csv_path.name,
        "stem": csv_path.stem,
        "path": str(csv_path.resolve()),
        "nrows": nrows,
        "ncols": len(cols),
        "columns": cols,
        "dtypes": dtypes,
        "na_counts_sample": na_counts,
        "timestamp_col": ts_col,
        "timestamp_min": None if tmin is None else str(tmin),
        "timestamp_max": None if tmax is None else str(tmax),
        "present_key_candidates": present_candidates,
        "value_stats_sample": value_stats,
        "top_categories_sample": top_categories,
        "head_preview": sample.head(SAMPLE_ROWS),
    }


def render_dataset_md(p: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# {p['file']}\n")
    lines.append("## File")
    lines.append(f"- Path: `{p['path']}`")
    lines.append(f"- Rows: {p['nrows'] if p['nrows'] is not None else 'unknown'}")
    lines.append(f"- Columns: {p['ncols']}\n")

    if p["timestamp_col"]:
        lines.append("## Time coverage")
        lines.append(f"- Timestamp column: `{p['timestamp_col']}`")
        lines.append(f"- Min: `{p['timestamp_min']}`")
        lines.append(f"- Max: `{p['timestamp_max']}`\n")

    lines.append("## Columns & dtypes (sample-based)")
    rows = []
    for c in p["columns"]:
        rows.append([c, p["dtypes"].get(c, ""), str(p["na_counts_sample"].get(c, 0))])
    lines.append(md_table(rows, ["column", "dtype", "NA count (sample)"]))
    lines.append("")

    lines.append("## Likely key columns present")
    lines.append("- " + (", ".join(p["present_key_candidates"]) if p["present_key_candidates"] else "None detected"))
    lines.append("")

    if p["value_stats_sample"] is not None:
        lines.append("## `value` summary (sample-based)")
        srows = [[k, "None" if v is None else (f"{v:.6g}" if isinstance(v, (int, float)) else str(v))]
                 for k, v in p["value_stats_sample"].items()]
        lines.append(md_table(srows, ["metric", "value"]))
        lines.append("")

    if p["top_categories_sample"]:
        lines.append("## Top categories (sample-based)")
        for col, pairs in p["top_categories_sample"].items():
            lines.append(f"### {col}")
            lines.append(md_table([[k, str(v)] for k, v in pairs], [col, "count"]))
            lines.append("")

    lines.append("## Preview (head)")
    lines.append("```text")
    lines.append(p["head_preview"].to_string(index=False))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def infer_connections(profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Creates:
    - shared columns between datasets
    - likely join keys (intersection of key candidates)
    - timestamp overlap if both have timestamp_min/max
    """
    by_name = {p["file"]: p for p in profiles}
    files = [p["file"] for p in profiles]

    # shared columns matrix (sparse)
    shared = []
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            a, b = by_name[files[i]], by_name[files[j]]
            inter = sorted(set(a["columns"]).intersection(b["columns"]))
            # keep meaningful ones (or show a small list)
            if inter:
                shared.append((a["file"], b["file"], inter))

    # likely join keys: intersection of key candidates
    join_keys = []
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            a, b = by_name[files[i]], by_name[files[j]]
            inter = sorted(set(a["present_key_candidates"]).intersection(b["present_key_candidates"]))
            if inter:
                join_keys.append((a["file"], b["file"], inter))

    # timestamp overlap (if possible)
    ts_overlap = []
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            a, b = by_name[files[i]], by_name[files[j]]
            if a["timestamp_col"] and b["timestamp_col"] and a["timestamp_min"] and b["timestamp_min"]:
                # parse back for overlap
                try:
                    a_min, a_max = pd.to_datetime(a["timestamp_min"]), pd.to_datetime(a["timestamp_max"])
                    b_min, b_max = pd.to_datetime(b["timestamp_min"]), pd.to_datetime(b["timestamp_max"])
                    overlap_start = max(a_min, b_min)
                    overlap_end = min(a_max, b_max)
                    has_overlap = overlap_start <= overlap_end
                    ts_overlap.append((a["file"], b["file"], str(overlap_start), str(overlap_end), has_overlap))
                except Exception:
                    pass

    return {"shared_columns": shared, "join_keys": join_keys, "timestamp_overlap": ts_overlap}


def render_all_md(profiles: List[Dict[str, Any]], connections: Dict[str, Any], input_dir: Path) -> str:
    lines = []
    lines.append("# All datasets\n")
    lines.append(f"Source folder: `{input_dir.resolve()}`\n")

    # Inventory
    lines.append("## Inventory")
    inv_rows = []
    for p in profiles:
        inv_rows.append([
            p["file"],
            str(p["nrows"] if p["nrows"] is not None else "unknown"),
            str(p["ncols"]),
            p["timestamp_col"] or "—",
            (p["timestamp_min"] or "—"),
            (p["timestamp_max"] or "—"),
        ])
    lines.append(md_table(inv_rows, ["file", "rows", "cols", "timestamp_col", "min_ts", "max_ts"]))
    lines.append("")

    # Connections: join keys
    lines.append("## How datasets are connected (inferred)\n")

    lines.append("### Likely join keys between datasets")
    if connections["join_keys"]:
        rows = []
        for a, b, keys in connections["join_keys"]:
            rows.append([a, b, ", ".join(keys)])
        lines.append(md_table(rows, ["dataset A", "dataset B", "shared key candidates"]))
    else:
        lines.append("- No shared key candidates found.\n")
    lines.append("")

    # Connections: timestamp overlap
    lines.append("### Timestamp overlap (if both have timestamps)")
    if connections["timestamp_overlap"]:
        rows = []
        for a, b, s, e, ok in connections["timestamp_overlap"]:
            rows.append([a, b, s, e, "yes" if ok else "no"])
        lines.append(md_table(rows, ["dataset A", "dataset B", "overlap_start", "overlap_end", "overlap?"]))
    else:
        lines.append("- Not enough timestamp info to compute overlap.\n")
    lines.append("")

    # Connections: shared columns (show limited list)
    lines.append("### Shared columns (limited list)")
    if connections["shared_columns"]:
        rows = []
        for a, b, inter in connections["shared_columns"]:
            show = inter[:15]
            more = "" if len(inter) <= 15 else f" (+{len(inter)-15} more)"
            rows.append([a, b, ", ".join(show) + more])
        lines.append(md_table(rows, ["dataset A", "dataset B", "shared columns"]))
    else:
        lines.append("- No shared columns found.\n")

    lines.append("")
    lines.append("## Notes")
    lines.append("- “Likely join keys” are inferred from common column names across files.")
    lines.append("- For energy-style long format, typical joins are on `timestamp` + (`all_building`/`building_id`) + `utility`.\n")
    return "\n".join(lines)


def main():
    in_path = Path(INPUT_DIR)
    out_path = Path(OUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)

    csvs = sorted(in_path.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in: {in_path.resolve()}")

    profiles = []
    for csv_path in csvs:
        p = profile_csv(csv_path)
        profiles.append(p)

        per_md = render_dataset_md(p)
        (out_path / f"{p['stem']}.md").write_text(per_md, encoding="utf-8")

    connections = infer_connections(profiles)
    all_md = render_all_md(profiles, connections, in_path)
    (out_path / "ALL_DATASETS.md").write_text(all_md, encoding="utf-8")

    # index
    index_lines = ["# Dataset docs", "", f"Source: `{in_path.resolve()}`", "", "## Files", ""]
    for p in profiles:
        index_lines.append(f"- **{p['file']}** → `{p['stem']}.md`")
    index_lines.append("")
    index_lines.append("- Combined doc → `ALL_DATASETS.md`")
    (out_path / "README.md").write_text("\n".join(index_lines), encoding="utf-8")

    print(f"✅ Wrote per-dataset docs + ALL_DATASETS.md to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
