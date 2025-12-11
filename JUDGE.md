# Judging / Reproduction Instructions

Thank you for reviewing our submission. This file contains concise steps for judges to reproduce the results and a summary of changes made to the original folder.

## Quick Reproduction (recommended)

1. Create a Python virtual environment and activate it (Windows PowerShell example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the pipeline (default behaviour uses `input_data/` and writes to `output_results/`):

```powershell
python pipeline.py
```

3. Expected outputs:
- `output_results/predictions.json` — the mandatory JSON submission file.
- `output_results/artifacts/` — audit images per site.
- `output_results/debug_views/` — visualization used for manual verification.

## Notes for Judges
- The repository includes a trained model (`solar_model.pt`) and some test artifacts for immediate evaluation.
- If you need to re-run at scale, consider using an environment with an NVIDIA GPU (recommended).
- Live image fetching requires internet access and may be subject to remote imagery provider rate limits.

## Changes I made to the original folder
The following changes were made while preparing this repository for judging and submission:

- Added: `.gitignore` — ignores future large/model/data files and common virtualenv folders.
- Added: `JUDGE.md` — this reproduction + judging instructions file.
- Updated: `README.md` — appended a short "Judging" pointer to this file.
- Git operations performed (no source files deleted):
  - Initialized local git repo (if not already present).
  - Added remote `origin` -> `https://github.com/prjjwal/ecoinnovators_26.git`.
  - Committed workspace files and pushed to `origin/main`.
  - Pulled and rebased remote changes as necessary to successfully push.

No existing source files were deleted or modified aside from the README append.

If you'd like, I can also:
- Remove the large image/test artifacts from the repo history (requires a force-push and coordination).
- Add a small demo script that runs inference on one sample and produces a single artifact for fast judging.

---
Thank you — please let me know which additional judging conveniences you'd like me to add.
