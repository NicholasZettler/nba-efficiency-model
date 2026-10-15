# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · ML vs. empirical Bayes, tracked in MLflow
# MAGIC
# MAGIC **Replaces:** `06_ml.py` (same models, features, and walk-forward evaluation)
# MAGIC **Reads:** `nba_gold_ml_pairs`
# MAGIC **Writes:** `nba_gold_ml_predictions`, `nba_gold_ml_metrics`, and one MLflow run
# MAGIC
# MAGIC **New here:** every run is logged to MLflow — parameters, a nested child run per test season, pooled metrics, feature importance, and the final Ridge and gradient-boosting models. Open **Experiments** in the left sidebar to compare runs.
# MAGIC
# MAGIC Walk-forward evaluation: for each test season pair, train only on EARLIER pairs, then predict that pair. This mimics real use (you only ever know the past) and keeps a player's future seasons out of training.

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from scipy.stats import linregress
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Databricks turns on autologging by default, which would log a separate run for
# every fold's .fit(). Turn it off and log deliberately instead.
mlflow.autolog(disable=True)

SCHEMA = "workspace.default"

FEATURES = [
    "FG3_PCT_RAW", "FG3_PCT_SHRUNK", "FG3_PCT_2YR",
    "FG3_PCT_PREV", "FG3A_PREV", "HAS_PREV",
    "FT_PCT", "FTA",
    "FG3A", "FG3A_PG", "FG3A_RATE",
    "AGE", "G", "MIN_PG",
]
TARGET = "TARGET"
WEIGHT = "FG3A_NEXT"   # next-season 3P% is less noisy for high-volume shooters
MIN_TRAIN_PAIRS = 4    # first test season needs at least 4 seasons of training data
RIDGE_ALPHAS = np.logspace(-2, 3, 20)
GBM_PARAMS = dict(learning_rate=0.03, max_iter=400, max_depth=3,
                  min_samples_leaf=40, l2_regularization=1.0, random_state=0)


def load_table(name):
    return spark.table(f"{SCHEMA}.{name}").toPandas()


def save_table(pdf, name):
    (spark.createDataFrame(pdf)
          .write.mode("overwrite")
          .option("overwriteSchema", "true")
          .saveAsTable(f"{SCHEMA}.{name}"))
    print(f"wrote {len(pdf):,} rows -> {SCHEMA}.{name}")


def make_models():
    """Fresh, untrained models (rebuilt for every fold)."""
    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          RidgeCV(alphas=RIDGE_ALPHAS))
    gbm = HistGradientBoostingRegressor(**GBM_PARAMS)
    return {"ridge": ridge, "gbm": gbm}


def prep(df):
    """Missing previous season -> 0 (HAS_PREV tells the model it was missing)."""
    X = df[FEATURES].copy()
    X[["FG3_PCT_PREV", "FG3A_PREV"]] = X[["FG3_PCT_PREV", "FG3A_PREV"]].fillna(0)
    return X


def fit(name, model, train):
    """Fit with sample weights (a pipeline needs the weight routed to its last step)."""
    kw = "ridgecv__sample_weight" if name == "ridge" else "sample_weight"
    model.fit(prep(train), train[TARGET], **{kw: train[WEIGHT]})
    return model


def log_sklearn(model, name, signature):
    """Log a fitted model. MLflow 3 takes `name`, MLflow 2 takes `artifact_path`."""
    kw = dict(signature=signature, serialization_format="cloudpickle")
    try:
        mlflow.sklearn.log_model(model, name=name, **kw)
    except TypeError:
        mlflow.sklearn.log_model(model, artifact_path=name, **kw)


def score(pred, actual, weights):
    """Same metrics as 04."""
    err = pred - actual
    out = {"RMSE": np.sqrt(np.mean(err ** 2)),
           "wRMSE": np.sqrt(np.sum(weights * err ** 2) / np.sum(weights)),
           "MAE": np.mean(np.abs(err)), "slope": np.nan, "r": np.nan}
    if pred.max() > pred.min():
        fit_ = linregress(pred, actual)
        out.update(slope=fit_.slope, r=fit_.rvalue)
    return out


PREDICTIONS = {
    "raw 3P%": "FG3_PCT_RAW",
    "EB shrunk": "FG3_PCT_SHRUNK",
    "2-yr 3P%": "FG3_PCT_2YR",
    "league avg": "PRED_LEAGUE",
    "ridge": "PRED_RIDGE",
    "gbm": "PRED_GBM",
}
METRIC_KEY = {"raw 3P%": "raw", "EB shrunk": "eb", "2-yr 3P%": "two_yr",
              "league avg": "league_avg", "ridge": "ridge", "gbm": "gbm"}


def evaluate(df, group_type, group):
    """One metrics row per prediction method."""
    actual = df[TARGET].to_numpy()
    w = df[WEIGHT].to_numpy(dtype=float)
    rows = [{"GROUP_TYPE": group_type, "GROUP": group, "N": len(df), "METHOD": name,
             **score(df[col].to_numpy(dtype=float), actual, w)}
            for name, col in PREDICTIONS.items()]
    by = {r["METHOD"]: r["RMSE"] for r in rows}
    best = min(by, key=by.get)
    gain = 100 * (by["EB shrunk"] - by[best]) / by["EB shrunk"]
    print(f"{group:>24}  n={len(df):>5,}  best RMSE: {best:<10} ({gain:+.2f}% vs. EB shrunk)")
    return rows

# COMMAND ----------

df = load_table("nba_gold_ml_pairs")
print(f"loaded {len(df):,} pairs")

with mlflow.start_run(run_name="walk_forward_eb_vs_ml") as run:
    mlflow.log_params({
        "features": ",".join(FEATURES), "n_features": len(FEATURES),
        "target": "next-season 3P%", "sample_weight": WEIGHT,
        "min_train_pairs": MIN_TRAIN_PAIRS, "validation": "walk-forward by season",
        "ridge_alphas": f"logspace(-2, 3, {len(RIDGE_ALPHAS)})",
        **{f"gbm_{k}": v for k, v in GBM_PARAMS.items()},
    })

    # ---- walk-forward: train on earlier seasons, predict the next one ----
    starts = sorted(df["START"].unique())
    folds = []
    for test_start in starts[MIN_TRAIN_PAIRS:]:
        train = df[df["START"] < test_start]
        test = df[df["START"] == test_start].copy()
        test["PRED_LEAGUE"] = test["PRIOR_MEAN"]  # season t's prior mean, same as 04
        for name, model in make_models().items():
            fit(name, model, train)
            test[f"PRED_{name.upper()}"] = model.predict(prep(test))

        pair = test["PAIR"].iloc[0]
        with mlflow.start_run(run_name=f"fold {pair}", nested=True):
            mlflow.log_params({"test_pair": pair, "train_rows": len(train), "test_rows": len(test)})
            for method, col in PREDICTIONS.items():
                m = score(test[col].to_numpy(dtype=float), test[TARGET].to_numpy(),
                          test[WEIGHT].to_numpy(dtype=float))
                mlflow.log_metric(f"rmse_{METRIC_KEY[method]}", m["RMSE"])
        folds.append(test)
        print(f"  trained on {len(train):,} pairs -> predicted {pair}")

    preds = pd.concat(folds, ignore_index=True)

    # ---- metrics: pooled, by test season, by shooting volume ----
    print()
    rows = evaluate(preds, "pooled", "ALL TEST SEASONS POOLED")
    for name, grp in preds.groupby("PAIR"):
        rows += evaluate(grp, "test_season", name)
    preds["BUCKET"] = pd.cut(preds["FG3A"], bins=[0, 100, 200, 400, np.inf],
                             labels=["<100 3PA", "100-199", "200-399", "400+"])
    for name, grp in preds.groupby("BUCKET", observed=True):
        if len(grp) >= 20:
            rows += evaluate(grp, "volume_bucket", name)
    preds["BUCKET"] = preds["BUCKET"].astype(str)
    metrics = pd.DataFrame(rows)

    pooled = metrics[metrics["GROUP_TYPE"] == "pooled"].set_index("METHOD")
    for method, r in pooled.iterrows():
        key = METRIC_KEY[method]
        mlflow.log_metrics({f"pooled_rmse_{key}": r["RMSE"], f"pooled_wrmse_{key}": r["wRMSE"],
                            f"pooled_mae_{key}": r["MAE"]})
        if not np.isnan(r["r"]):
            mlflow.log_metric(f"pooled_r_{key}", r["r"])
    eb = pooled.loc["EB shrunk", "RMSE"]
    for key in ["ridge", "gbm"]:
        mlflow.log_metric(f"rmse_gain_pct_{key}_vs_eb", 100 * (eb - pooled.loc[key, "RMSE"]) / eb)

    # ---- feature importance: train on all but the last season, test on the last ----
    last = df["START"].max()
    tr, te = df[df["START"] < last], df[df["START"] == last]
    models = make_models()
    ridge = fit("ridge", models["ridge"], tr)
    coefs = pd.Series(ridge.named_steps["ridgecv"].coef_, index=FEATURES)
    gbm = fit("gbm", models["gbm"], tr)
    perm = permutation_importance(gbm, prep(te), te[TARGET], sample_weight=te[WEIGHT],
                                  scoring="neg_root_mean_squared_error",
                                  n_repeats=30, random_state=0)
    imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
    mlflow.log_dict(coefs.round(6).to_dict(), "ridge_standardized_coefficients.json")
    mlflow.log_dict(imp.round(6).to_dict(), "gbm_permutation_importance.json")

    # ---- final models: trained on every season, logged for reuse ----
    X_all = prep(df).astype(float)  # float schema so a missing value never breaks the signature
    for name, model in make_models().items():
        fit(name, model, df)
        log_sklearn(model, f"{name}_model", infer_signature(X_all, model.predict(X_all)))

    print(f"\nMLflow run: {run.info.run_id}")

# COMMAND ----------

save_table(preds, "nba_gold_ml_predictions")
save_table(metrics, "nba_gold_ml_metrics")

display(metrics[metrics["GROUP_TYPE"] == "pooled"])
display(coefs.reindex(coefs.abs().sort_values(ascending=False).index).round(4)
             .rename("ridge_coef").reset_index().rename(columns={"index": "feature"}))
display(imp.round(5).rename("gbm_perm_importance").reset_index()
           .rename(columns={"index": "feature"}))
