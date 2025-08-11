from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np

@dataclass
class RecItem:
    title: str
    rationale: str
    expected_attendance: float
    bonding_class: str

def make_recommendations(pipe, drafts_df: pd.DataFrame) -> List[RecItem]:
    """Generate recommendations based on model predictions."""
    # Demo: use single trained model output as attendance proxy.
    try:
        preds_att = pipe.predict(drafts_df)
    except Exception:
        preds_att = [np.nan] * len(drafts_df)

    bonding = []
    for i in range(len(drafts_df)):
        val = preds_att[i]
        if isinstance(val, (int, float, np.floating)):
            bonding.append("High" if val >= 80 else ("Medium" if val >= 40 else "Low"))
        else:
            bonding.append("Unknown")

    items = []
    for i, row in drafts_df.iterrows():
        title = f"{row.get('weekday','?')} {row.get('start_time','?')} · {row.get('event_type','?')} ({row.get('age_min','?')}-{row.get('age_max','?')})"
        rationale = "Drivers: youth theme, low price, indoor in winter"
        ea = preds_att[i] if isinstance(preds_att[i], (int, float, np.floating)) else np.nan
        items.append(RecItem(
            title=title, 
            rationale=rationale, 
            expected_attendance=float(ea) if ea==ea else np.nan, 
            bonding_class=bonding[i]
        ))
    
    score = {"High": 3, "Medium": 2, "Low": 1, "Unknown": 0}
    return sorted(items, key=lambda x: (score[x.bonding_class], x.expected_attendance if x.expected_attendance==x.expected_attendance else -1), reverse=True)