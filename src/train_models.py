"""CPU-only, group-aware regression experiments for a Mapillary index."""
import json
import numpy as np
from config import INDEX_DIR
from research import spatial_group


def train_regressors():
    meta, embeddings = INDEX_DIR / "metadata.json", INDEX_DIR / "embeddings.npy"
    if not meta.exists() or not embeddings.exists():
        raise ValueError("Build a Mapillary index before training.")
    rows = json.loads(meta.read_text(encoding="utf-8"))
    if not rows or any(row.get("source") != "Mapillary" for row in rows):
        raise ValueError("Only a Mapillary-only index is eligible for this experiment.")
    groups = [spatial_group(row) for row in rows]
    if len(rows) < 20 or len(set(groups)) < 5:
        raise ValueError("Need at least 20 images spanning 5 independent spatial groups; no results were created.")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.tree import DecisionTreeRegressor
    x, y = np.load(embeddings), np.array([[r["lat"], r["lon"]] for r in rows])
    cv = GroupKFold(n_splits=min(5, len(set(groups))))
    specs = {"decision_tree": (DecisionTreeRegressor(random_state=42), {"estimator__max_depth": [3, 6, None], "estimator__min_samples_leaf": [1, 2, 4]}),
             "random_forest": (RandomForestRegressor(n_estimators=150, max_features="sqrt", n_jobs=-1, random_state=42), {"estimator__max_depth": [None, 12], "estimator__min_samples_leaf": [1, 2]})}
    output = {"status": "completed", "samples": len(rows), "spatial_groups": len(set(groups)), "models": {}}
    for name, (base, grid) in specs.items():
        search = GridSearchCV(MultiOutputRegressor(base), grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
        search.fit(x, y, groups=groups)
        output["models"][name] = {"best_params": search.best_params_, "cv_negative_mae": float(search.best_score_)}
    (INDEX_DIR / "regression_experiments.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    try:
        print(json.dumps(train_regressors(), indent=2))
    except ValueError as error:
        print(f"UNAVAILABLE: {error}")
