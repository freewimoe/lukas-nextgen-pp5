import pandas as pd
from src.preprocess import infer_feature_types

def test_infer_feature_types():
    df = pd.DataFrame({"a":[1,2], "b":["x","y"], "y":[0,1]})
    num, cat = infer_feature_types(df, target="y")
    assert "a" in num and "b" in cat and "y" not in num+cat