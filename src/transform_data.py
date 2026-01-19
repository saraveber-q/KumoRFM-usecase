# src/transform_data.py
from __future__ import annotations

from pathlib import Path
import shutil
import pandas as pd


SKIP_AS_IS = {"weather.csv", "metadata.csv"}


def melt_cleaned_csvs(
    input_dir: str = "datasets/original",
    output_dir: str = "datasets/cleaned",
) -> None:
    """
    Reads CSVs from input_dir, melts wide utility files into long format,
    adds surrogate keys, and writes them to output_dir.

    Utility output columns:
      reading_id, timestamp, utility, all_building, building_key, value

    weather.csv and metadata.csv are copied unchanged (optionally with extra keys, see below).
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------- Pass 1: collect all building labels across all utility files (global key map) --------
    all_buildings = set()

    utility_files = [p for p in sorted(input_dir.glob("*.csv")) if p.name not in SKIP_AS_IS]
    for csv_path in utility_files:
        df = pd.read_csv(csv_path, nrows=1)
        ts_col = df.columns[0]
        building_cols = [c for c in df.columns if c != ts_col]
        all_buildings.update(building_cols)

    # Stable global building_key mapping (sorted for determinism)
    building_to_key = {b: i for i, b in enumerate(sorted(all_buildings), start=1)}

    # -------- Pass 2: melt + write files --------
    for csv_path in sorted(input_dir.glob("*.csv")):
        name = csv_path.name
        print(f"Processing {name}...")

        # Copy weather + metadata unchanged
        if name in SKIP_AS_IS:
            shutil.copy2(csv_path, output_dir / name)
            continue

        df = pd.read_csv(csv_path)

        # Assume first column is timestamp
        ts_col = df.columns[0]

        melted = df.melt(
            id_vars=[ts_col],
            var_name="all_building",
            value_name="value",
        ).rename(columns={ts_col: "timestamp"})

        # Add utility name from filename
        utility = csv_path.stem.replace("_cleaned", "")
        melted.insert(1, "utility", utility)

        # Standardize types
        melted["value"] = pd.to_numeric(melted["value"], errors="coerce")

        # Surrogate keys
        melted["building_key"] = melted["all_building"].map(building_to_key).astype("Int64")
        melted["reading_id"] = pd.RangeIndex(start=1, stop=len(melted) + 1, step=1, dtype="int64")

        # Nice column order
        melted = melted[["reading_id", "timestamp", "utility", "all_building", "building_key", "value"]]

        out_path = output_dir / name
        melted.to_csv(out_path, index=False)

    # -------- Optional: add surrogate keys to metadata (kept otherwise identical) --------
    metadata_path = output_dir / "metadata.csv"
    if metadata_path.exists():
        md = pd.read_csv(metadata_path)

        # If metadata has building_id, you can add a numeric building_key for convenience.
        # NOTE: This uses building_id mapping, not all_building labels (different concepts).
        if "building_id" in md.columns and "building_key" not in md.columns:
            md["building_key"] = md["building_id"].astype("category").cat.codes.astype("int64") + 1

        md.to_csv(metadata_path, index=False)

    print("✅ Melted utility CSVs + added surrogate keys. Output:", output_dir.resolve())


if __name__ == "__main__":
    melt_cleaned_csvs()
