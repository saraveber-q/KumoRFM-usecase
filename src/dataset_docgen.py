# dataset_docgen.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


# ----------------------------
# Config
# ----------------------------
DEFAULT_INPUT_DIR = "datasets/cleaned"   # change to wherever your CSVs live
DEFAULT_OUT_DIR = "docs/datasets/cleaned"
MAX_UNIQUE_TO_LIST = 25
SAMPLE_ROWS = 5
TOP_VALUE_COUNTS = 10


def _safe_read_csv(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read CSV with minimal assumptions."""
    return pd.read_csv(path, nrows=nrows)


def _maybe_parse_datetime(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if col not in df.columns:
        return None
    s = pd.to_datetime(df[col], errors="coerce", utc=False)
    if s.notna().any():
        return s
    return None


def _fmt_int(x: Any) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)


def _md_table(rows, headers) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def profile_csv(csv_path: Path) -> Dict[str, Any]:
    # lightweight peek
    head = _safe_read_csv(csv_path, nrows=2000)  # enough to infer common patterns
    df_full = None  # only load full file if needed for better stats

    info: Dict[str, Any] = {}
    info["path"] = str(csv_path.resolve())
    info["ncols"] = head.shape[1]

    # Try to get row count fast-ish
    # (pure pandas doesn't have a perfect fast rowcount for CSV; this is a pragmatic approach)
    # If files are huge, you can comment this out.
    try:
        with csv_path.open("rb") as f:
            nrows = sum(1 for _ in f) - 1  # minus header
        info["nrows"] = max(nrows, 0)
    except Exception:
        info["nrows"] = None

    # schema
    dtypes = head.dtypes.astype(str).to_dict()
    info["dtypes"] = dtypes

    # missingness (sample-based if huge)
    na_counts = head.isna().sum().to_dict()
    info["na_counts_sample"] = na_counts

    # detect timestamp column candidates
    ts_candidates = [c for c in ["timestamp", "time", "date", "datetime"] if c in head.columns]
    ts_col = ts_candidates[0] if ts_candidates else None
    info["timestamp_col"] = ts_col

    # parse timestamp range (use full file if timestamp exists and file isn't massive)
    ts_min = ts_max = None
    if ts_col:
        # chunk to get true min/max without loading everything
        try:
            tmin, tmax = None, None
            for chunk in pd.read_csv(csv_path, usecols=[ts_col], chunksize=250_000):
                ts = pd.to_datetime(chunk[ts_col], errors="coerce")
                if ts.notna().any():
                    cmin, cmax = ts.min(), ts.max()
                    tmin = cmin if tmin is None else min(tmin, cmin)
                    tmax = cmax if tmax is None else max(tmax, cmax)
            ts_min, ts_max = tmin, tmax
        except Exception:
            # fallback: sample-based
            ts = _maybe_parse_datetime(head, ts_col)
            if ts is not None:
                ts_min, ts_max = ts.min(), ts.max()
    info["timestamp_min"] = None if ts_min is None or pd.isna(ts_min) else str(ts_min)
    info["timestamp_max"] = None if ts_max is None or pd.isna(ts_max) else str(ts_max)

    # Identify likely key columns
    key_cols = [c for c in ["reading_id", "building_key", "all_building", "building_id", "utility"] if c in head.columns]
    info["key_cols"] = key_cols

    # Value column stats if present
    value_col = "value" if "value" in head.columns else None
    info["value_col"] = value_col
    if value_col:
        v = pd.to_numeric(head[value_col], errors="coerce")
        info["value_stats_sample"] = {
            "count_nonnull": int(v.notna().sum()),
            "mean": float(v.mean()) if v.notna().any() else None,
            "std": float(v.std()) if v.notna().any() else None,
            "min": float(v.min()) if v.notna().any() else None,
            "p25": float(v.quantile(0.25)) if v.notna().any() else None,
            "median": float(v.quantile(0.50)) if v.notna().any() else None,
            "p75": float(v.quantile(0.75)) if v.notna().any() else None,
            "max": float(v.max()) if v.notna().any() else None,
        }

    # Uniques / top categories for key cols (sample-based)
    uniques = {}
    top_values = {}
    for c in key_cols:
        s = head[c]
        uniques[c] = int(s.nunique(dropna=True))
        if s.dtype == "object" or c in ("all_building", "utility"):
            vc = s.astype(str).value_counts(dropna=True).head(TOP_VALUE_COUNTS)
            top_values[c] = [(k, int(v)) for k, v in vc.items()]
    info["uniques_sample"] = uniques
    info["top_values_sample"] = top_values

    # Provide a small preview head()
    info["head_preview"] = head.head(SAMPLE_ROWS)

    return info


def render_markdown(profile: Dict[str, Any]) -> str:
    file_name = profile["file"]
    nrows = profile.get("nrows")
    ncols = profile.get("ncols")
    ts_col = profile.get("timestamp_col")

    lines = []
    lines.append(f"# {file_name}")
    lines.append("")
    lines.append("## File")
    lines.append(f"- Path: `{profile.get('path')}`")
    lines.append(f"- Rows: {(_fmt_int(nrows) if nrows is not None else 'unknown')}")
    lines.append(f"- Columns: {ncols}")
    lines.append("")
    lines.append("## Columns & dtypes")
    rows = []
    for c, dt in profile["dtypes"].items():
        na = profile["na_counts_sample"].get(c, 0)
        rows.append([c, dt, _fmt_int(na)])
    lines.append(_md_table(rows, headers=["column", "dtype (sampled)", "NA count (sample)"]))
    lines.append("")

    lines.append("## Keys / identifiers (detected)")
    lines.append("- " + (", ".join(profile.get("key_cols") or []) if profile.get("key_cols") else "None detected"))
    lines.append("")

    if ts_col:
        lines.append("## Time coverage")
        lines.append(f"- Timestamp column: `{ts_col}`")
        lines.append(f"- Min: `{profile.get('timestamp_min')}`")
        lines.append(f"- Max: `{profile.get('timestamp_max')}`")
        lines.append("")

    if profile.get("value_col"):
        stats = profile.get("value_stats_sample", {})
        lines.append("## `value` summary (sample-based)")
        rows = [[k, "None" if v is None else (f"{v:.6g}" if isinstance(v, (int, float)) else str(v))] for k, v in stats.items()]
        lines.append(_md_table(rows, headers=["metric", "value"]))
        lines.append("")

    uniques = profile.get("uniques_sample", {})
    if uniques:
        lines.append("## Unique counts (sample-based)")
        rows = [[k, _fmt_int(v)] for k, v in uniques.items()]
        lines.append(_md_table(rows, headers=["column", "unique values"]))
        lines.append("")

    topvals = profile.get("top_values_sample", {})
    if topvals:
        lines.append("## Top categories (sample-based)")
        for col, pairs in topvals.items():
            lines.append(f"### {col}")
            rows = [[k, _fmt_int(v)] for k, v in pairs]
            lines.append(_md_table(rows, headers=[col, "count"]))
            lines.append("")

    lines.append("## Preview (head)")
    lines.append("```text")
    lines.append(profile["head_preview"].to_string(index=False))
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def write_docs(input_dir: str = DEFAULT_INPUT_DIR, out_dir: str = DEFAULT_OUT_DIR) -> None:
    in_path = Path(input_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csvs = sorted(in_path.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in: {in_path.resolve()}")

    index_lines = ["# Dataset docs", "", f"Source folder: `{in_path.resolve()}`", ""]
    index_lines.append("## Files")
    index_lines.append("")

    for csv_path in csvs:
        prof = profile_csv(csv_path)
        md = render_markdown(prof)

        md_name = csv_path.stem + ".md"
        (out_path / md_name).write_text(md, encoding="utf-8")

        # index entry
        rows = prof.get("nrows")
        cols = prof.get("ncols")
        ts_col = prof.get("timestamp_col")
        tmin = prof.get("timestamp_min")
        tmax = prof.get("timestamp_max")
        index_lines.append(f"- **{csv_path.name}** → `{md_name}`  "
                           f"(rows={rows if rows is not None else 'unknown'}, cols={cols}, "
                           f"ts={ts_col or '—'}, range={tmin or '—'}..{tmax or '—'})")

    (out_path / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"✅ Wrote docs to: {out_path.resolve()}")
    print(f"✅ Index: { (out_path / 'README.md').resolve() }")


if __name__ == "__main__":
    # Example:
    #   python dataset_docgen.py
    #
    # Or set env vars:
    #   INPUT_DIR=datasets/cleaned OUT_DIR=docs/datasets python dataset_docgen.py
    input_dir = os.environ.get("INPUT_DIR", DEFAULT_INPUT_DIR)
    out_dir = os.environ.get("OUT_DIR", DEFAULT_OUT_DIR)
    write_docs()
