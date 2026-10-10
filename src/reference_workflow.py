"""New reference example; synthetic data, not a reproduced classroom experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import loguniform
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run(method="grid"):
    X, y = make_classification(n_samples=600, n_features=12,
        n_informative=6, n_redundant=2, weights=[0.85, 0.15], random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    pipe = Pipeline([("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42))])
    inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=43)
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)
    if method == "grid":
        search = GridSearchCV(pipe, {"model__C": [0.01, 0.1, 1, 10, 100],
            "model__class_weight": [None, "balanced"]},
            scoring="average_precision", cv=inner, n_jobs=1)
    else:
        search = RandomizedSearchCV(pipe,
            {"model__C": loguniform(1e-2, 1e2),
             "model__class_weight": [None, "balanced"]}, n_iter=20,
            scoring="average_precision", cv=inner, n_jobs=1, random_state=42)
    baseline = cross_val_score(pipe, X_dev, y_dev,
        cv=outer, scoring="average_precision", n_jobs=1)
    nested = cross_val_score(search, X_dev, y_dev,
        cv=outer, scoring="average_precision", n_jobs=1)
    search.fit(X_dev, y_dev)
    scores = search.best_estimator_.predict_proba(X_test)[:, 1]
    report = {
        "status": "new reference experiment on synthetic data",
        "method": method,
        "baseline_cv_ap_mean": float(np.mean(baseline)),
        "baseline_cv_ap_std_ddof0": float(np.std(baseline)),
        "nested_cv_ap_mean": float(np.mean(nested)),
        "nested_cv_ap_std_ddof0": float(np.std(nested)),
        "nested_outer_scores": nested.tolist(),
        "best_internal_cv_ap": float(search.best_score_),
        "best_params": search.best_params_,
        "holdout_ap": float(average_precision_score(y_test, scores)),
        "holdout_roc_auc": float(roc_auc_score(y_test, scores)),
    }
    output = Path("artifacts")
    output.mkdir(exist_ok=True)
    (output / f"{method}-reference.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["grid", "random"], default="grid")
    run(parser.parse_args().method)
