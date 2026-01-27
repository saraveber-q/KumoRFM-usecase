# KumoRFM – Data Pipeline

This repository contains a data pipeline that:
1. Downloads **private datasets from Google Drive**
2. Transforms them locally
3. Creates a smaller subset of the data
4. Uploads results back to Google Drive

The pipeline is designed to be run with **one command**.

---

## 📁 Project Structure

```
KumoRFM-usecase/
│
├─ run_pipeline.py          # Main entry point (recommended)
├─ drive_sync.py            # Download data from Google Drive
│
├─ src/
│   ├─ transform_data.py    # original → cleaned
│   ├─ make_small.py        # cleaned → cleaned_small
│   └─ drive_upload.py      # upload results back to Drive
│
├─ datasets/
│   ├─ cleaned/
│   └─ cleaned_small/
│
├─ drive_cache/             # Local cache of Drive data (auto-created)
│   └─ original/
│
├─ secrets/
│   ├─ client_secret.json
│   └─ token.json
│
├─ requirements.txt
└─ README.md
```

---

## 🔐 Google Drive Access

- Data is stored in **private Google Drive folders**
- Access is handled via **Google Drive API (OAuth)**
- Each user authenticates once with their own Google account
- No public links and no service accounts are used

---

## 1️⃣ Setup (one-time)

### Create and activate a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2️⃣ Google Drive configuration (one-time per user)

1. Obtain `client_secret.json` from the project owner
2. Place it in:
   ```
   secrets/client_secret.json
   ```
3. Ensure your Google account email is added as a **Test User**
4. Set the Drive folder ID (from the Drive URL)

**Windows**
```powershell
$env:KUMORFM_DRIVE_FOLDER_ID="YOUR_DRIVE_FOLDER_ID"
```

**macOS / Linux**
```bash
export KUMORFM_DRIVE_FOLDER_ID="YOUR_DRIVE_FOLDER_ID"
```

➡️ The first run will open a browser window for Google login.

---

## 3️⃣ Run the full pipeline (recommended)

```powershell
python run_pipeline.py
```

This will:
1. Download data from Google Drive → `drive_cache/`
2. Transform original data → `datasets/cleaned/`
3. Create a reduced dataset → `datasets/cleaned_small/`
4. Upload results back to Google Drive

---

## 4️⃣ Skipping steps (for faster iteration)

### Skip Drive download
```powershell
python run_pipeline.py --skip-sync
```

### Only regenerate `cleaned_small`
```powershell
python run_pipeline.py --skip-sync --skip-transform --skip-upload
```

### Only upload results
```powershell
python run_pipeline.py --skip-sync --skip-transform --skip-small
```

---

## 5️⃣ Optional parameters

### Change the time window (months)
```powershell
python run_pipeline.py --months 1
```

### Limit number of buildings
```powershell
python run_pipeline.py --max-buildings 50
```

Use `0` to keep all buildings (default).

---

## 6️⃣ Output locations

### Local
```
drive_cache/original/
datasets/cleaned/
datasets/cleaned_small/
```

### Google Drive
```
datasets/cleaned/
datasets/cleaned_small/
```

Uploads update existing files (no duplicates).

---

## 7️⃣ Troubleshooting

**403 / Access blocked**
- Your email is not added as a Test User

**Pipeline runs everything again**
- Use `--skip-*` flags

**Transform step is slow**
- Large CSVs → expected behavior
