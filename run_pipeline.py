from pathlib import Path
import os
import sys
import shutil

# ---- imports from your project ----
from drive_sync import get_drive_service, sync_folder
from src.transform_data import melt_cleaned_csvs
from src.make_small import make_small_dataset
from src.drive_upload import upload_folder


# ---- paths ----
DRIVE_DATASETS_ID = os.environ.get("KUMORFM_DRIVE_FOLDER_ID")
if not DRIVE_DATASETS_ID:
    sys.exit("❌ KUMORFM_DRIVE_FOLDER_ID is not set")

WORKDIR = Path("drive_cache")
ORIGINAL = WORKDIR / "original"

CLEANED = Path("datasets/cleaned")
CLEANED_SMALL = Path("datasets/cleaned_small")


def main():
    print("\n=== KumoRFM pipeline started ===\n")

    # 1. Sync Drive → local cache
    print("🔄 Syncing data from Google Drive...")
    service = get_drive_service()
    sync_folder(service, DRIVE_DATASETS_ID, WORKDIR)

    # 2. Transform original → cleaned
    print("\n🧪 Transforming original → cleaned...")
    CLEANED.mkdir(parents=True, exist_ok=True)
    melt_cleaned_csvs(
        input_dir=str(ORIGINAL),
        output_dir=str(CLEANED),
    )

    # 3. Create cleaned_small
    print("\n📉 Creating cleaned_small dataset...")
    CLEANED_SMALL.mkdir(parents=True, exist_ok=True)
    make_small_dataset(
        input_dir=str(CLEANED),
        output_dir=str(CLEANED_SMALL),
        months=2,
        max_buildings=None,
    )

    # 4. Upload results back to Drive
    print("\n☁️ Uploading results back to Google Drive...")
    upload_folder(service, CLEANED, DRIVE_DATASETS_ID)
    upload_folder(service, CLEANED_SMALL, DRIVE_DATASETS_ID)

    print("\n✅ Pipeline finished successfully")


if __name__ == "__main__":
    main()
