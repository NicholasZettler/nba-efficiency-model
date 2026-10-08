"""Does machine learning beat empirical Bayes at predicting next-season 3P%?

Walk-forward evaluation: for each test season pair, train only on EARLIER pairs,
then predict that pair. This mimics real use (you only ever know the past) and
keeps a player's future seasons out of training.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROC_DIR = Path("data/processed")
IN_PATH = PROC_DIR / "ml_pairs.csv"
OUT_PATH = PROC_DIR / "ml_predictions.csv"

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


def make_models():
    """Fresh, untrained models (rebuilt for every fold)."""
    ridge = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-2, 3, 20)),
    )
    gbm = HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_iter=400,
        max_depth=3,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=0,
    )
    return {"ridge": ridge, "gbm": gbm}


def prep(df):
    """Missing previous season -> 0 (HAS_PREV tells the model it was missing)."""
    X = df[FEATURES].copy()
    X[["FG3_PCT_PREV", "FG3A_PREV"]] = X[["FG3_PCT_PREV", "FG3A_PREV"]].fillna(0)
    return X


def score(pred, actual, weights):
    """Same metrics as 04_validate.py."""
    err = pred - actual
    out = {
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "wRMSE": np.sqrt(np.sum(weights * err ** 2) / np.sum(weights)),
        "MAE": np.mean(np.abs(err)),
        "slope": np.nan,
        "r": np.nan,
    }
    if pred.max() > pred.min():
        fit = linregress(pred, actual)
        out.update(slope=fit.slope, r=fit.rvalue)
    return out


def fit(name, model, train):
    """Fit with sample weights (a pipeline needs the weight routed to its last step)."""
    kw = "ridgecv__sample_weight" if name == "ridge" else "sample_weight"
    model.fit(prep(train), train[TARGET], **{kw: train[WEIGHT]})
    return model


def walk_forward(df):
    """Train on all pairs before each test year, predict that year."""
    starts = sorted(df["START"].unique())
    folds = []

    for test_start in starts[MIN_TRAIN_PAIRS:]:
        train = df[df["START"] < test_start]
        test = df[df["START"] == test_start].copy()

        # League-average baseline: season t's prior mean, same as 04_validate.py.
        test["PRED_LEAGUE"] = test["PRIOR_MEAN"]

        for name, model in make_models().items():
            fit(name, model, train)
            test[f"PRED_{name.upper()}"] = model.predict(prep(test))

        folds.append(test)
        print(f"  trained on {len(train):,} pairs -> predicted {test['PAIR'].iloc[0]}")

    return pd.concat(folds, ignore_index=True)


PREDICTIONS = {
    "raw 3P%": "FG3_PCT_RAW",
    "EB shrunk": "FG3_PCT_SHRUNK",
    "2-yr 3P%": "FG3_PCT_2YR",
    "league avg": "PRED_LEAGUE",
    "ridge": "PRED_RIDGE",
    "gbm": "PRED_GBM",
}


def report(df, label):
    actual = df[TARGET].to_numpy()
    w = df[WEIGHT].to_numpy(dtype=float)
    print(f"\n{label}   (n = {len(df):,})")
    print(f"{'':>11} {'RMSE':>8} {'wRMSE':>8} {'MAE':>8} {'slope':>7} {'r':>7}")
    print("-" * 54)
    rows = {}
    for name, col in PREDICTIONS.items():
        m = score(df[col].to_numpy(dtype=float), actual, w)
        rows[name] = m
        tail = (f"{'--':>7} {'--':>7}" if name == "league avg"  # constant per season: no slope
                else f"{m['slope']:>7.3f} {m['r']:>7.3f}")
        print(f"{name:>11} {m['RMSE']:>8.4f} {m['wRMSE']:>8.4f} {m['MAE']:>8.4f} {tail}")
    best = min(rows, key=lambda k: rows[k]["RMSE"])
    gain = 100 * (rows["EB shrunk"]["RMSE"] - rows[best]["RMSE"]) / rows["EB shrunk"]["RMSE"]
    print(f"  best RMSE: {best}  ({gain:+.2f}% vs. EB shrunk)")


def importance(df):
    """Which features matter? Fit on all but the last season, test on the last."""
    last = df["START"].max()
    train, test = df[df["START"] < last], df[df["START"] == last]
    models = make_models()

    ridge = fit("ridge", models["ridge"], train)
    coefs = pd.Series(ridge.named_steps["ridgecv"].coef_, index=FEATURES)
    print("\nRidge coefficients (standardized: change in 3P% per 1 SD of the feature):")
    print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).round(4).to_string())

    gbm = fit("gbm", models["gbm"], train)
    perm = permutation_importance(
        gbm, prep(test), test[TARGET], sample_weight=test[WEIGHT],
        scoring="neg_root_mean_squared_error", n_repeats=30, random_state=0,
    )
    imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
    print("\nGBM permutation importance (RMSE increase when the feature is shuffled):")
    print(imp.round(5).to_string())


def main():
    df = pd.read_csv(IN_PATH)
    print(f"loaded {len(df):,} pairs\n")
    print("walk-forward training:")
    preds = walk_forward(df)
    preds.to_csv(OUT_PATH, index=False)
    print(f"\nsaved predictions -> {OUT_PATH}")

    report(preds, "ALL TEST SEASONS POOLED")

    print("\n" + "=" * 54)
    print("BY TEST SEASON")
    for name, grp in preds.groupby("PAIR"):
        report(grp, name)

    print("\n" + "=" * 54)
    print("BY SEASON-T VOLUME  (ML should help most at low volume)")
    bins = [0, 100, 200, 400, np.inf]
    labels = ["<100 3PA", "100-199", "200-399", "400+"]
    preds["BUCKET"] = pd.cut(preds["FG3A"], bins=bins, labels=labels)
    for name, grp in preds.groupby("BUCKET", observed=True):
        if len(grp) >= 20:
            report(grp, str(name))

    print("\n" + "=" * 54)
    importance(df)


if __name__ == "__main__":
    main()
