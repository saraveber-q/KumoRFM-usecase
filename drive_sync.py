from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Optional, List, Dict

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def clean_drive_folder_id(raw: str) -> str:
    """
    Accepts either:
      - a folder ID: 1abcDEF...
      - a full URL: https://drive.google.com/drive/folders/<ID>?...
    Returns the cleaned folder ID with query params removed.
    """
    if raw is None:
        return ""

    s = raw.strip().strip('"').strip("'")

    # If a full URL was pasted, extract after /folders/
    if "/folders/" in s:
        s = s.split("/folders/", 1)[1]

    # Remove query string / fragments
    s = s.split("?", 1)[0]
    s = s.split("#", 1)[0]
    s = s.split("&", 1)[0]

    # Extra safety: remove any whitespace
    s = s.strip()

    return s


def validate_drive_id(folder_id: str) -> None:
    """
    Very lightweight validation:
    Drive IDs are typically URL-safe base64-ish strings.
    """
    if not folder_id:
        raise ValueError("Empty folder_id after cleaning.")

    # Must not contain URL punctuation that indicates user pasted too much
    if any(ch in folder_id for ch in ["?", "&", "/", "=", " "]):
        raise ValueError(f"folder_id still contains invalid characters: {folder_id!r}")

    # Typical Drive IDs are 20+ chars. (Not a strict rule, but good guardrail.)
    if len(folder_id) < 15:
        raise ValueError(f"folder_id looks too short: {folder_id!r}")

    # Allowed characters check (letters, digits, underscore, dash)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", folder_id):
        raise ValueError(f"folder_id has unexpected characters: {folder_id!r}")


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
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
            # Prefer local server (opens browser). In headless/docker, fall back to
            # a no-browser local server flow that prints the auth URL.
            try:
                creds = flow.run_local_server(port=0)
            except Exception:
                print("No runnable browser found. Using manual auth flow.")
                print("Open the URL printed below on your host machine.")
                creds = flow.run_local_server(
                    host="localhost",      # this is what Google will use in redirect_uri
                    port=8080,
                    bind_addr="0.0.0.0",   # this is what the server listens on inside Docker
                    open_browser=False,
                )

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

    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
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
    raw = os.environ.get("KUMORFM_DRIVE_FOLDER_ID")
    if not raw:
        raise SystemExit("KUMORFM_DRIVE_FOLDER_ID is not set")

    folder_id = clean_drive_folder_id(raw)
    validate_drive_id(folder_id)

    print(f"📁 Using Drive folder id: {folder_id}")

    out_dir = Path(os.environ.get("KUMORFM_CACHE_DIR", "drive_cache"))

    service = get_drive_service()
    sync_folder(service, folder_id, out_dir)

    print(f"✅ Drive folder synced to: {out_dir.resolve()}")
