import pandas as pd
from typing import Tuple, List

def infer_feature_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    feats = df.drop(columns=[target])
    num_cols = feats.select_dtypes(include="number").columns.tolist()
    cat_cols = feats.select_dtypes(exclude="number").columns.tolist()
    return num_cols, cat_cols