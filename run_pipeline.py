from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from drive_sync import get_drive_service, sync_folder
from src.transform_data import melt_cleaned_csvs
from src.make_small import make_small_dataset
from src.drive_upload import upload_tree


def parse_args():
    p = argparse.ArgumentParser(description="Run KumoRFM end-to-end pipeline.")
    p.add_argument("--skip-sync", action="store_true", help="Skip downloading from Drive.")
    p.add_argument("--skip-transform", action="store_true", help="Skip transforming original -> cleaned.")
    p.add_argument("--skip-small", action="store_true", help="Skip generating cleaned_small.")
    p.add_argument("--skip-upload", action="store_true", help="Skip uploading results to Drive.")

    p.add_argument("--months", type=int, default=2, help="Number of months for cleaned_small window.")
    p.add_argument("--max-buildings", type=int, default=0, help="0 keeps all buildings; otherwise top N.")

    p.add_argument("--drive-cache", default="drive_cache", help="Local cache folder for Drive downloads.")
    p.add_argument("--local-datasets", default="datasets", help="Local datasets output folder.")
    return p.parse_args()


def main():
    args = parse_args()

    drive_folder_id = os.environ.get("KUMORFM_DRIVE_FOLDER_ID")
    if not drive_folder_id:
        sys.exit("❌ KUMORFM_DRIVE_FOLDER_ID is not set")

    drive_cache = Path(args.drive_cache)
    original_dir = drive_cache / "original"

    local_root = Path(args.local_datasets)
    cleaned_dir = local_root / "cleaned"
    cleaned_small_dir = local_root / "cleaned_small"

    print("\n=== KumoRFM pipeline started ===\n")

    service = get_drive_service()

    # 1) Sync
    if not args.skip_sync:
        print("🔄 Syncing data from Google Drive...")
        sync_folder(service, drive_folder_id, drive_cache)
    else:
        print("⏭️  Skipping sync")

    # 2) Transform
    if not args.skip_transform:
        print("\n🧪 Transforming original → cleaned...")
        melt_cleaned_csvs(input_dir=str(original_dir), output_dir=str(cleaned_dir))
    else:
        print("\n⏭️  Skipping transform")

    # 3) Small dataset
    if not args.skip_small:
        print("\n📉 Creating cleaned_small dataset...")
        max_buildings = None if args.max_buildings == 0 else args.max_buildings
        make_small_dataset(
            input_dir=str(cleaned_dir),
            output_dir=str(cleaned_small_dir),
            months=args.months,
            max_buildings=max_buildings,
        )
    else:
        print("\n⏭️  Skipping cleaned_small")

    # 4) Upload
    if not args.skip_upload:
        print("\n☁️ Uploading results back to Google Drive...")
        upload_tree(service, cleaned_dir, drive_folder_id, "cleaned")
        upload_tree(service, cleaned_small_dir, drive_folder_id, "cleaned_small")
    else:
        print("\n⏭️  Skipping upload")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
