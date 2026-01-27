## Project Structure

```
KumoRFM-usecase/
├── datasets/
│   ├── original/        # Raw input datasets (downloaded manually)
│   ├── cleaned/         # Transformed long-format datasets
│   └── cleaned_small/   # Reduced-size datasets for faster iteration
│
├── src/
│   ├── transform_data.py
│   ├── make_small.py
│   ├── drive_sync.py
│   ├── drive_upload.py
│   ├── dataset_docgen.py
│   └── dataset_docgen_all.py
│
├── run_pipeline.py
├── sanity_data_check.ipynb
├── requirements.txt
├── README.md
└── README_DataPipline_Sara.md
```

---

## Environment Setup

1. Create a new virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
pip install -r requirements.txt
pip install kumoai
```
2. Download original datasets
Download the raw (original) datasets and place them in:
**bash**
```powershell
datasets/original/
```
3. Data Transformation
Transform original datasets → cleaned format

This step:

* Converts wide CSVs into long format
* Adds surrogate keys
* Copies weather.csv and metadata.csv as-is
* Run from the project root:

**Windows (PowerShell)**
```powershell
python src/transform_data.py
```

**bash**
```powershell
datasets/cleaned/
```