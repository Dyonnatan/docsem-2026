"""Repository locations. Override the data root with ``DFM_DATA``."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "prompts"
CONFIGS_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = Path(os.environ.get("DFM_DATA", REPO_ROOT / "data"))
