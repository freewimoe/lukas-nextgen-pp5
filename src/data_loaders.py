import pandas as pd

def load_csv(file_like) -> pd.DataFrame:
    """Load CSV data from file-like object or file path."""
    return pd.read_csv(file_like)