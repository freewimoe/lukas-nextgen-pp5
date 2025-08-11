from typing import List, Dict
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    r2_score, mean_absolute_error, mean_squared_error,
)
from src.config import MODELS_DIR, LATEST_MODEL, REPORT_PATH


def build_pipeline(num_cols: List[str], cat_cols: List[str], model_name: str = "logreg") -> dict:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    if model_name == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        reg = RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        clf = LogisticRegression(max_iter=1000)
        reg = LinearRegression()

    pipe_clf = Pipeline([("pre", pre), ("est", clf)])
    pipe_reg = Pipeline([("pre", pre), ("est", reg)])

    pipe_clf.task = "classification"
    pipe_reg.task = "regression"
    return {"clf": pipe_clf, "reg": pipe_reg}


def fit_and_save(df: pd.DataFrame, target: str, pipe_dict) -> Dict:
    X = df.drop(columns=[target])
    y = df[target]

    task = "regression" if np.issubdtype(y.dtype, np.number) else "classification"
    pipe = pipe_dict["reg" if task == "regression" else "clf"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None if task=="regression" else y
    )

    pipe.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, LATEST_MODEL)

    metrics = evaluate(pipe, X_train, y_train, X_test, y_test, task)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def evaluate(pipe: Pipeline, X_tr, y_tr, X_te, y_te, task: str) -> Dict:
    report = {"task": task}
    if task == "classification":
        ytr_pred = pipe.predict(X_tr)
        yte_pred = pipe.predict(X_te)
        report.update({
            "train": {
                "accuracy": float(accuracy_score(y_tr, ytr_pred)),
                "f1_weighted": float(f1_score(y_tr, ytr_pred, average="weighted")),
                "report": classification_report(y_tr, ytr_pred, output_dict=True),
            },
            "test": {
                "accuracy": float(accuracy_score(y_te, yte_pred)),
                "f1_weighted": float(f1_score(y_te, yte_pred, average="weighted")),
                "report": classification_report(y_te, yte_pred, output_dict=True),
            },
        })
    else:
        ytr_pred = pipe.predict(X_tr)
        yte_pred = pipe.predict(X_te)
        report.update({
            "train": {
                "r2": float(r2_score(y_tr, ytr_pred)),
                "mae": float(mean_absolute_error(y_tr, ytr_pred)),
                "rmse": float(mean_squared_error(y_tr, ytr_pred, squared=False)),
            },
            "test": {
                "r2": float(r2_score(y_te, yte_pred)),
                "mae": float(mean_absolute_error(y_te, yte_pred)),
                "rmse": float(mean_squared_error(y_te, yte_pred, squared=False)),
            },
        })
    return report