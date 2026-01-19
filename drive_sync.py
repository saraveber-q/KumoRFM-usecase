from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional, List, Dict

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SCOPES = ["https://www.googleapis.com/auth/drive.file"]




def get_drive_service(
    client_secret_path: str = "secrets/client_secret.json",
    token_path: str = "secrets/token.json",
):
    token_file = Path(token_path)
    creds: Optional[Credentials] = None

    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secret_path, SCOPES
            )
            creds = flow.run_local_server(port=0)

        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(creds.to_json(), encoding="utf-8")

    return build("drive", "v3", credentials=creds)


def list_children(service, folder_id: str) -> List[Dict]:
    results = []
    page_token = None
    query = f"'{folder_id}' in parents and trashed = false"
    fields = "nextPageToken, files(id, name, mimeType)"

    while True:
        resp = service.files().list(
            q=query,
            fields=fields,
            pageToken=page_token,
            pageSize=1000,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()

        results.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return results


def download_file(service, file_id: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    request = service.files().get_media(
        fileId=file_id, supportsAllDrives=True
    )
    fh = io.FileIO(dest_path, "wb")
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    fh.close()


def sync_folder(service, folder_id: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    children = list_children(service, folder_id)

    for item in children:
        name = item["name"]
        mime = item["mimeType"]
        item_id = item["id"]

        if mime == "application/vnd.google-apps.folder":
            sync_folder(service, item_id, local_dir / name)
        else:
            # skip Google Docs/Sheets
            if mime.startswith("application/vnd.google-apps"):
                print(f"Skipping Google-native file: {name}")
                continue

            dest = local_dir / name
            print(f"Downloading: {dest}")
            download_file(service, item_id, dest)


if __name__ == "__main__":
    folder_id = os.environ.get("KUMORFM_DRIVE_FOLDER_ID")
    if not folder_id:
        raise SystemExit("KUMORFM_DRIVE_FOLDER_ID is not set")

    out_dir = Path(os.environ.get("KUMORFM_CACHE_DIR", "drive_cache"))

    service = get_drive_service()
    sync_folder(service, folder_id, out_dir)

    print(f"✅ Drive folder synced to: {out_dir.resolve()}")
