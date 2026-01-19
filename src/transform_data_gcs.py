# transform_data_gcs.py
from __future__ import annotations

from pathlib import Path
import io
import pandas as pd

try:
    import gcsfs
except ImportError as e:
    raise SystemExit("Install gcsfs: pip install gcsfs") from e


SKIP_AS_IS = {"weather.csv", "metadata.csv"}


def _is_gcs(path: str) -> bool:
    return path.startswith("gs://")


def _list_csvs(input_dir: str) -> list[str]:
    if _is_gcs(input_dir):
        fs = gcsfs.GCSFileSystem()
        # Ensure trailing slash behavior
        prefix = input_dir.rstrip("/") + "/"
        files = fs.glob(prefix + "*.csv")
        return sorted(files)  # returns like 'bucket/path/file.csv'
    else:
        return sorted(str(p) for p in Path(input_dir).glob("*.csv"))


def _join_path(base: str, name: str) -> str:
    if _is_gcs(base):
        return base.rstrip("/") + "/" + name
    return str(Path(base) / name)


def _read_csv(path: str, **kwargs) -> pd.DataFrame:
    if _is_gcs(path):
        return pd.read_csv("gs://" + path if not path.startswith("gs://") else path, **kwargs)
    return pd.read_csv(path, **kwargs)


def _copy_file(src: str, dst: str) -> None:
    if _is_gcs(src) or _is_gcs(dst):
        fs = gcsfs.GCSFileSystem()
        s = src if src.startswith("gs://") else "gs://" + src
        d = dst if dst.startswith("gs://") else "gs://" + dst
        fs.copy(s, d)
    else:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(Path(src).read_bytes())


def _write_csv(df: pd.DataFrame, out_path: str) -> None:
    if _is_gcs(out_path):
        fs = gcsfs.GCSFileSystem()
        with fs.open(out_path, "w") as f:
            df.to_csv(f, index=False)
    else:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)


def melt_cleaned_csvs(
    input_dir: str = "datasets/original",
    output_dir: str = "datasets/cleaned",
) -> None:
    # List input CSVs
    all_files = _list_csvs(input_dir)

    # Pass 1: collect building labels across utility files
    all_buildings = set()
    utility_files = [p for p in all_files if Path(p).name not in SKIP_AS_IS]
    for p in utility_files:
        df = _read_csv(p, nrows=1)
        ts_col = df.columns[0]
        building_cols = [c for c in df.columns if c != ts_col]
        all_buildings.update(building_cols)

    building_to_key = {b: i for i, b in enumerate(sorted(all_buildings), start=1)}

    # Pass 2: process each file
    for p in all_files:
        name = Path(p).name
        out_path = _join_path(output_dir, name)
        print(f"Processing {name} -> {out_path}")

        if name in SKIP_AS_IS:
            _copy_file(p, out_path)
            continue

        df = _read_csv(p)
        ts_col = df.columns[0]

        melted = df.melt(
            id_vars=[ts_col],
            var_name="all_building",
            value_name="value",
        ).rename(columns={ts_col: "timestamp"})

        utility = Path(name).stem.replace("_cleaned", "")
        melted.insert(1, "utility", utility)
        melted["value"] = pd.to_numeric(melted["value"], errors="coerce")

        melted["building_key"] = melted["all_building"].map(building_to_key).astype("Int64")
        melted["reading_id"] = pd.RangeIndex(1, len(melted) + 1, dtype="int64")

        melted = melted[["reading_id", "timestamp", "utility", "all_building", "building_key", "value"]]
        _write_csv(melted, out_path)

    print("✅ Done.")


if __name__ == "__main__":
    # Example:
    # melt_cleaned_csvs("gs://my-bucket/datasets/original", "gs://my-bucket/datasets/cleaned")
    melt_cleaned_csvs()
