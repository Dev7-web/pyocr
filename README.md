## Doc Pipeline Studio – Run Guide

Local‑first document processing and semantic image retrieval, now split cleanly into **backend** and **frontend**.

- **Backend** (Python, FastAPI, pipeline): `backend/`
- **Frontend** (static UI): `frontend/`

### Project layout

```text
pyocr/
├── backend/
│   ├── app.py              # Web app launcher (uvicorn wrapper)
│   ├── main.py             # CLI entrypoint
│   ├── requirements.txt    # Backend dependencies
│   ├── webapp/
│   │   ├── app.py          # FastAPI app
│   │   └── job_manager.py  # Job orchestration
│   ├── extractors/         # PDF / PPTX / DOCX / spreadsheet extractors
│   ├── image_processing/   # Embeddings + metadata generation
│   ├── models/             # Pydantic data models
│   ├── search/             # Semantic search
│   └── utils/              # Helpers, logging, file I/O
├── frontend/
│   ├── index.html          # Dashboard UI
│   ├── styles.css
│   └── app.js
└── README.md               # This file
```

### 1. Environment setup

- **Python**: 3.10+ recommended.
- From the **project root** (`pyocr/`), create and activate a virtualenv:

```bash
python -m venv .venv
.venv\Scripts\activate           # on Windows PowerShell
```

- Install backend dependencies (includes `uvicorn`, `fastapi`, etc.):

```bash
pip install -r backend/requirements.txt
```

If you previously saw `ModuleNotFoundError: No module named 'uvicorn'`, this step fixes it.

### 2. Run the web app (backend API + frontend UI)

From the **project root** (`pyocr/`) with the venv activated:

```bash
python -m backend.app
```

This runs `uvicorn` with the FastAPI app defined in `backend/webapp/app.py`. Then open:

- `http://127.0.0.1:8000` – main UI

From the browser:

1. Upload a supported document (`PDF`, `PPTX`, `DOCX`, `CSV`, `XLSX/XLS`).
2. Wait until the job status becomes **completed**.
3. Run semantic image searches in the search panel.

### 3. CLI usage (processing without the web UI)

Always invoke the CLI as a **module** from the project root so imports stay correct:

```bash
# Extract only
python -m backend.main extract --file sample.csv --output backend/output

# Full pipeline (extract + metadata + index)
python -m backend.main pipeline --file sample.csv --output backend/output

# Build index for an existing extracted document directory
python -m backend.main build-index --input backend/output/<document_folder>

# Search an existing index
python -m backend.main search --index backend/output/<document_folder> --query "circuit diagram"
```

> Note: Avoid running `python main.py` directly from inside `backend/`, because that bypasses the package context and can cause import errors. Use `python -m backend.main` from the project root instead.

### 4. Key API endpoints (when the web app is running)

- `GET /api/health`
- `POST /api/upload` – upload a document (multipart form)
- `GET /api/jobs` – list jobs
- `GET /api/jobs/{job_id}` – job status/details
- `GET /api/jobs/{job_id}/result` – extraction result JSON
- `GET /api/jobs/{job_id}/gallery` – image gallery metadata
- `POST /api/jobs/{job_id}/search` – semantic image search
- `DELETE /api/jobs/{job_id}` – delete job + artifacts

### 5. Output structure

Per processed document/job, artifacts are stored under `backend/output/`:

```text
backend/output/<job_id>/<document_name_ext>/
├── extraction_result.json
├── texts/full_text.txt
├── tables/*.csv
├── images/*
├── image_index.json
└── embeddings.npz
```

### 6. Optional system dependencies

For best results, install:

- **Tesseract OCR** – used by `pytesseract` for better image text extraction.
- **LibreOffice** – used as a fallback to render complex PPTX charts/SmartArt to images.

### 7. Running tests (if/when you add them back)

From the project root with the venv active:

```bash
pytest -q
```

