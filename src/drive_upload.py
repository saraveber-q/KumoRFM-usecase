# drive_upload.py
from pathlib import Path
from googleapiclient.http import MediaFileUpload


def upload_folder(service, local_dir: Path, drive_parent_id: str):
    for path in local_dir.rglob("*"):
        if path.is_dir():
            continue

        media = MediaFileUpload(
            path,
            mimetype="text/csv",
            resumable=True,
        )

        file_metadata = {
            "name": path.name,
            "parents": [drive_parent_id],
        }

        service.files().create(
            body=file_metadata,
            media_body=media,
            supportsAllDrives=True,
        ).execute()

