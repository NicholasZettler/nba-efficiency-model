# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold: empirical Bayes shrinkage
# MAGIC
# MAGIC **Replaces:** `03_model.py` (same model)
# MAGIC **Reads:** `nba_silver_player_seasons`
# MAGIC **Writes:** `nba_gold_shrunk` (every player-season with its shrunk 3P%), `nba_gold_priors` (one fitted Beta prior per season)

# COMMAND ----------

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln

SCHEMA = "workspace.default"


def load_table(name):
    return spark.table(f"{SCHEMA}.{name}").toPandas()


def save_table(pdf, name):
    (spark.createDataFrame(pdf)
          .write.mode("overwrite")
          .option("overwriteSchema", "true")
          .saveAsTable(f"{SCHEMA}.{name}"))
    print(f"wrote {len(pdf):,} rows -> {SCHEMA}.{name}")


def neg_loglik(params, x, n):
    """Negative Beta-Binomial log-likelihood. params are log(alpha), log(beta)."""
    alpha, beta = np.exp(params)
    ll = betaln(x + alpha, n - x + beta) - betaln(alpha, beta)
    return -ll.sum()


def fit_mom(x, n):
    """Method-of-moments starting values from observed rates."""
    p = x / n
    m, v = p.mean(), p.var(ddof=1)
    total = max(m * (1 - m) / v - 1, 1.0)
    return m * total, (1 - m) * total


def fit_season(x, n):
    """MLE for alpha, beta. Returns (alpha, beta)."""
    a0, b0 = fit_mom(x, n)
    res = minimize(neg_loglik, x0=np.log([a0, b0]), args=(x, n), method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000})
    if not res.success:
        print(f"  warning: optimizer did not converge ({res.message})")
    return tuple(np.exp(res.x))

# COMMAND ----------

df = load_table("nba_silver_player_seasons")
print(f"loaded {len(df):,} player-seasons")

pieces, priors = [], []
for season, grp in df.groupby("SEASON_YEAR"):
    x = grp["FG3M"].to_numpy(dtype=float)
    n = grp["FG3A"].to_numpy(dtype=float)
    alpha, beta = fit_season(x, n)
    strength = alpha + beta

    out = grp.copy()
    out["ALPHA"] = alpha
    out["BETA"] = beta
    out["PRIOR_MEAN"] = alpha / strength
    out["FG3_PCT_SHRUNK"] = (out["FG3M"] + alpha) / (out["FG3A"] + strength)
    out["WEIGHT"] = out["FG3A"] / (out["FG3A"] + strength)
    out["SHIFT"] = out["FG3_PCT_SHRUNK"] - out["FG3_PCT_RAW"]
    pieces.append(out)
    priors.append({"SEASON_YEAR": season, "PLAYERS": len(grp), "ALPHA": alpha, "BETA": beta,
                   "PRIOR_MEAN": alpha / strength, "PRIOR_STRENGTH": strength})

result = pd.concat(pieces, ignore_index=True)
priors = pd.DataFrame(priors)
display(priors)

# COMMAND ----------

save_table(result, "nba_gold_shrunk")
save_table(priors, "nba_gold_priors")

latest = result[result["SEASON_YEAR"] == result["SEASON_YEAR"].max()]
cols = ["PLAYER_NAME", "FG3M", "FG3A", "FG3_PCT_RAW", "FG3_PCT_SHRUNK", "SHIFT"]
print("biggest downward corrections (small-sample hot shooters):")
display(latest.nsmallest(5, "SHIFT")[cols])
print("biggest upward corrections:")
display(latest.nlargest(5, "SHIFT")[cols])
