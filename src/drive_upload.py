from __future__ import annotations

import os
import time
import ssl
from pathlib import Path
from typing import Optional, Dict

from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError


# ---------- Drive helpers ----------

def _find_child_folder_id(service, parent_id: str, name: str) -> Optional[str]:
    q = (
        f"'{parent_id}' in parents and "
        f"mimeType='application/vnd.google-apps.folder' and "
        f"name='{name}' and trashed=false"
    )
    resp = service.files().list(
        q=q,
        fields="files(id,name)",
        pageSize=10,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def _ensure_folder(service, parent_id: str, name: str) -> str:
    existing = _find_child_folder_id(service, parent_id, name)
    if existing:
        return existing

    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    created = service.files().create(
        body=metadata,
        fields="id",
        supportsAllDrives=True,
    ).execute()
    return created["id"]


def _find_file_id(service, parent_id: str, name: str) -> Optional[str]:
    q = (
        f"'{parent_id}' in parents and "
        f"name='{name}' and "
        f"mimeType!='application/vnd.google-apps.folder' and "
        f"trashed=false"
    )
    resp = service.files().list(
        q=q,
        fields="files(id,name,modifiedTime,size)",
        pageSize=10,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def _upload_with_retries(request, filename: str, max_retries: int = 8):
    """
    Executes a resumable upload request with retries and progress output.
    """
    attempt = 0
    response = None

    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                percent = int(status.progress() * 100)
                print(f"⬆️ Uploading {filename}: {percent}%")
        except (ssl.SSLEOFError, ConnectionError, TimeoutError) as e:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_s = min(60, 2 ** attempt)
            print(f"⚠️ Upload interrupted ({type(e).__name__}), retrying in {sleep_s}s...")
            time.sleep(sleep_s)
        except HttpError as e:
            status_code = getattr(e.resp, "status", None)
            if status_code in (429, 500, 502, 503, 504):
                attempt += 1
                if attempt > max_retries:
                    raise
                sleep_s = min(60, 2 ** attempt)
                print(f"⚠️ HTTP {status_code}, retrying in {sleep_s}s...")
                time.sleep(sleep_s)
            else:
                raise

    print(f"✅ Finished uploading {filename}")
    return response
# ---------- Public API ----------

def upload_tree(
    service,
    local_dir: Path,
    drive_root_folder_id: str,
    drive_subpath: str,
    mimetype_default: str = "text/csv",
) -> None:
    """
    Upload local_dir contents into Drive folder:
      drive_root_folder_id / drive_subpath / (mirrored local structure)

    Example:
      upload_tree(service, Path("datasets/cleaned"), DATASETS_ID, "cleaned")
    """
    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")

    # Ensure subpath folders exist in Drive
    current_parent = drive_root_folder_id
    for part in Path(drive_subpath).parts:
        current_parent = _ensure_folder(service, current_parent, part)

    # Walk local files and create matching folders
    for path in local_dir.rglob("*"):
        if path.is_dir():
            # create folder in Drive mirroring structure
            rel = path.relative_to(local_dir)
            parent = current_parent
            for part in rel.parts:
                parent = _ensure_folder(service, parent, part)
            continue

        rel_parent = path.parent.relative_to(local_dir)
        parent = current_parent
        for part in rel_parent.parts:
            parent = _ensure_folder(service, parent, part)

        name = path.name
        file_id = _find_file_id(service, parent, name)

        media = MediaFileUpload(
            filename=str(path),
            mimetype=mimetype_default,
            resumable=True,
            chunksize=5 * 1024 * 1024,  # 5MB chunks = stable on flaky networks
        )

        if file_id:
            # Update existing file (prevents duplicates)
            print(f"⬆️ Updating: {drive_subpath}/{rel_parent}/{name}")
            req = service.files().update(
                fileId=file_id,
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            )
            _upload_with_retries(req, filename=name)

        else:
            # Create new file
            print(f"⬆️ Creating: {drive_subpath}/{rel_parent}/{name}")
            metadata = {"name": name, "parents": [parent]}
            req = service.files().create(
                body=metadata,
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            )
            _upload_with_retries(req, filename=name)

