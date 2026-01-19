from pathlib import Path
import os
import shutil

from drive_sync import get_drive_service, sync_folder
from transform_data import melt_cleaned_csvs
from make_small import make_small_dataset
from drive_upload import upload_folder  # small helper, see below


DRIVE_DATASETS_ID = os.environ["KUMORFM_DRIVE_FOLDER_ID"]

WORKDIR = Path(".work")
LOCAL_ORIGINAL = WORKDIR / "original"
LOCAL_CLEANED = WORKDIR / "cleaned"
LOCAL_SMALL = WORKDIR / "cleaned_small"


def main():
    service = get_drive_service()

    # clean workspace
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    WORKDIR.mkdir()

    # 1. Download Drive/original → local
    sync_folder(service, DRIVE_DATASETS_ID, WORKDIR)

    # 2. Transform locally
    melt_cleaned_csvs(
        input_dir=LOCAL_ORIGINAL,
        output_dir=LOCAL_CLEANED,
    )

    make_small_dataset(
        input_dir=LOCAL_CLEANED,
        output_dir=LOCAL_SMALL,
    )

    # 3. Upload results back to Drive
    upload_folder(service, LOCAL_CLEANED, DRIVE_DATASETS_ID + "/cleaned")
    upload_folder(service, LOCAL_SMALL, DRIVE_DATASETS_ID + "/cleaned_small")

    print("✅ Drive pipeline complete")


if __name__ == "__main__":
    main()
