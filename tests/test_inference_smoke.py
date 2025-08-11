import pandas as pd
from src.model import build_pipeline, fit_and_save

def test_fit_predict_smoke(tmp_path, monkeypatch):
    df = pd.DataFrame({
        "feat_num":[1,2,3,4,5,6,7,8],
        "feat_cat":["a","a","b","b","a","b","a","a"],
        "y":[0,1,0,1,0,1,0,1]
    })
    pipes = build_pipeline(["feat_num"],["feat_cat"], model_name="logreg")
    m = fit_and_save(df, "y", pipes)
    assert "test" in m and ("accuracy" in m["test"] or "r2" in m["test"])