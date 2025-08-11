"""Configuration settings for the LUKAS NextGen project."""
from pathlib import Path

# Project root directory
ROOT = Path(__file__).resolve().parents[1]

# Model storage directories
MODELS_DIR = ROOT / "models"
LATEST_MODEL = MODELS_DIR / "latest.joblib"

# Evaluation report path
REPORT_PATH = MODELS_DIR / "eval_report.json"

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)