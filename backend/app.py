from __future__ import annotations

import sys
from pathlib import Path

import uvicorn


def main() -> None:
    # Ensure project root (one level above backend/) is on sys.path so that
    # the `backend` package is always importable, whether you run:
    #   - from project root:  python -m backend.app
    #   - from backend/:      python app.py
    backend_dir = Path(__file__).resolve().parent
    project_root = backend_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    uvicorn.run("backend.webapp.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
