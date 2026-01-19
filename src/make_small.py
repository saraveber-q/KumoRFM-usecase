# src/make_small.py
from __future__ import annotations

from pathlib import Path
import shutil
import pandas as pd


SKIP_AS_IS = {"metadata.csv"}  # copy as-is
TIME_FILES = {"weather.csv"}   # also filter by timestamp


def _ensure_datetime(s: pd.Series) -> pd.Series:
    # handles strings or already-datetime
    return pd.to_datetime(s, errors="coerce")


def _find_min_timestamp(csv_path: Path, ts_col: str = "timestamp", chunksize: int = 250_000) -> pd.Timestamp:
    min_ts = None
    for chunk in pd.read_csv(csv_path, usecols=[ts_col], chunksize=chunksize):
        ts = _ensure_datetime(chunk[ts_col])
        cur = ts.min()
        if pd.isna(cur):
            continue
        min_ts = cur if min_ts is None else min(min_ts, cur)
    if min_ts is None:
        raise ValueError(f"Could not find any valid timestamps in {csv_path.name}")
    return pd.Timestamp(min_ts)


def _select_top_buildings_in_window(
    csv_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    building_col: str = "all_building",
    ts_col: str = "timestamp",
    top_n: int = 50,
    chunksize: int = 250_000,
) -> set[str]:
    counts: dict[str, int] = {}
    usecols = [ts_col, building_col]

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        ts = _ensure_datetime(chunk[ts_col])
        mask = (ts >= start) & (ts < end)
        if not mask.any():
            continue
        sub = chunk.loc[mask, building_col].astype(str)
        vc = sub.value_counts(dropna=True)
        for k, v in vc.items():
            counts[k] = counts.get(k, 0) + int(v)

    if not counts:
        return set()

    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return {k for k, _ in top}


def make_small_dataset(
    input_dir: str = "datasets/cleaned",
    output_dir: str = "datasets/cleaned_small",
    months: int = 2,
    max_buildings: int | None = None,   # e.g. 50; set None to keep all buildings
    chunksize: int = 250_000,
) -> None:
    """
    Creates a smaller version of your cleaned datasets by:
      - keeping only the first `months` months (based on min timestamp per file)
      - (optional) keeping only the top `max_buildings` buildings in that window (utility tables)

    Assumes utility tables are long format with columns:
      timestamp, all_building, value, ...

    Writes filtered CSVs to output_dir.
    """

    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(in_dir.glob("*.csv")):
        name = csv_path.name
        out_path = out_dir / name
        print(name)

        # metadata: copy through
        if name in SKIP_AS_IS:
            shutil.copy2(csv_path, out_path)
            print(f"✅ copied as-is: {name}")
            continue

        # Determine time window for this file
        start = _find_min_timestamp(csv_path, ts_col="timestamp", chunksize=chunksize)
        end = start + pd.DateOffset(months=months)

        # For utility tables, optionally pick top buildings
        top_buildings: set[str] | None = None
        is_utility_like = name not in TIME_FILES and name not in SKIP_AS_IS

        if is_utility_like and max_buildings is not None:
            top_buildings = _select_top_buildings_in_window(
                csv_path,
                start=start,
                end=end,
                top_n=max_buildings,
                chunksize=chunksize,
            )
            print(f"ℹ️ {name}: keeping top {len(top_buildings)} buildings in first {months} months")

        # Stream-filter and write
        wrote_header = False
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            ts = _ensure_datetime(chunk["timestamp"])
            mask = (ts >= start) & (ts < end)

            if is_utility_like and top_buildings is not None and "all_building" in chunk.columns:
                mask = mask & chunk["all_building"].astype(str).isin(top_buildings)

            if not mask.any():
                continue

            out_chunk = chunk.loc[mask]
            out_chunk.to_csv(out_path, index=False, mode="a", header=not wrote_header)
            wrote_header = True

        if wrote_header:
            print(f"✅ wrote filtered: {name}  ({start.date()} → {end.date()})")
        else:
            # Still write an empty file with headers, so downstream code doesn't break
            # (take headers from original)
            head = pd.read_csv(csv_path, nrows=0)
            head.to_csv(out_path, index=False)
            print(f"⚠️ no rows matched window: {name} (wrote empty file with headers)")

    print("\nDone. Smaller files are in:", out_dir.resolve())


if __name__ == "__main__":
    # Example defaults: first 2 months, keep all buildings
    make_small_dataset(months=2, max_buildings=None)

    # If you ALSO want to cap buildings, uncomment:
    # make_small_dataset(months=2, max_buildings=50)
