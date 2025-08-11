import json
from src.config import REPORT_PATH

def load_eval_report():
    try:
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None