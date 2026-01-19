# make_small_gcs.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

try:
    import gcsfs
except ImportError as e:
    raise SystemExit("Install gcsfs: pip install gcsfs") from e


SKIP_AS_IS = {"metadata.csv"}
TIME_FILES = {"weather.csv"}


def _is_gcs(path: str) -> bool:
    return path.startswith("gs://")


def _ensure_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _list_csvs(input_dir: str) -> list[str]:
    if _is_gcs(input_dir):
        fs = gcsfs.GCSFileSystem()
        prefix = input_dir.rstrip("/") + "/"
        return sorted(fs.glob(prefix + "*.csv"))
    return sorted(str(p) for p in Path(input_dir).glob("*.csv"))


def _join_path(base: str, name: str) -> str:
    return base.rstrip("/") + "/" + name if _is_gcs(base) else str(Path(base) / name)


def _copy_file(src: str, dst: str) -> None:
    if _is_gcs(src) or _is_gcs(dst):
        fs = gcsfs.GCSFileSystem()
        s = src if src.startswith("gs://") else "gs://" + src
        d = dst if dst.startswith("gs://") else "gs://" + dst
        fs.copy(s, d)
    else:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(Path(src).read_bytes())


def _find_min_timestamp(csv_path: str, ts_col: str = "timestamp", chunksize: int = 250_000) -> pd.Timestamp:
    min_ts = None
    for chunk in pd.read_csv(csv_path, usecols=[ts_col], chunksize=chunksize):
        ts = _ensure_datetime(chunk[ts_col])
        cur = ts.min()
        if pd.isna(cur):
            continue
        min_ts = cur if min_ts is None else min(min_ts, cur)
    if min_ts is None:
        raise ValueError(f"Could not find any valid timestamps in {csv_path}")
    return pd.Timestamp(min_ts)


def make_small_dataset(
    input_dir: str = "datasets/cleaned",
    output_dir: str = "datasets/cleaned_small",
    months: int = 2,
    max_buildings: int | None = None,
    chunksize: int = 250_000,
) -> None:
    fs = gcsfs.GCSFileSystem() if _is_gcs(output_dir) else None

    for csv_path in _list_csvs(input_dir):
        name = Path(csv_path).name
        out_path = _join_path(output_dir, name)
        print(name)

        if name in SKIP_AS_IS:
            _copy_file(csv_path, out_path)
            print(f"✅ copied as-is: {name}")
            continue

        start = _find_min_timestamp(csv_path, ts_col="timestamp", chunksize=chunksize)
        end = start + pd.DateOffset(months=months)

        wrote_header = False

        # Open output stream (GCS or local) once, append chunks
        if _is_gcs(out_path):
            f = fs.open(out_path, "w")
        else:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            f = open(out_path, "w", newline="")

        try:
            for chunk in pd.read_csv(csv_path, chunksize=chunksize):
                ts = _ensure_datetime(chunk["timestamp"])
                mask = (ts >= start) & (ts < end)

                if not mask.any():
                    continue

                out_chunk = chunk.loc[mask]
                out_chunk.to_csv(f, index=False, header=not wrote_header)
                wrote_header = True

            if wrote_header:
                print(f"✅ wrote filtered: {name}  ({start.date()} → {end.date()})")
            else:
                # write empty with headers
                head = pd.read_csv(csv_path, nrows=0)
                head.to_csv(f, index=False)
                print(f"⚠️ no rows matched window: {name} (wrote empty file with headers)")
        finally:
            f.close()

    print("Done.")


if __name__ == "__main__":
    # Example:
    # make_small_dataset(
    #   input_dir="gs://my-bucket/datasets/cleaned",
    #   output_dir="gs://my-bucket/datasets/cleaned_small",
    # )
    make_small_dataset()
