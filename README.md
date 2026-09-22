# Empirical Bayes Shrinkage for Small-Sample Shooting Rates

A player who goes 34-for-76 from three "shoots 44.7%," but 76 attempts say very
little about his true ability. This project fits a league-wide Beta prior to NBA
3-point data by maximum likelihood, shrinks each player's percentage toward it in
proportion to how little data he has, and tests whether the shrunk estimates
predict the next season better than raw percentages do.

## Method

Each player-season *i* has makes $x_i$ out of attempts $n_i$:

$$x_i \mid p_i \sim \text{Binomial}(n_i, p_i), \qquad p_i \sim \text{Beta}(\alpha, \beta)$$

$\alpha$ and $\beta$ are estimated separately for each season by maximizing the
Beta-Binomial marginal log-likelihood:

$$\ell(\alpha, \beta) = \sum_i \Big[\ln B(x_i + \alpha,\ n_i - x_i + \beta) - \ln B(\alpha, \beta)\Big]$$

The optimizer (Nelder-Mead) works on $\log\alpha, \log\beta$ so both stay positive,
starting from method-of-moments estimates. Each player's shrunk estimate is the
posterior mean:

$$\hat p_i = \frac{x_i + \alpha}{n_i + \alpha + \beta} = w_i \frac{x_i}{n_i} + (1 - w_i)\frac{\alpha}{\alpha + \beta}, \qquad w_i = \frac{n_i}{n_i + \alpha + \beta}$$

$\alpha + \beta$ works like a number of league-average "phantom attempts" added
to every player's record.

## Data

- Regular-season player game logs from stats.nba.com via
  [`nba_api`](https://github.com/swar/nba_api): 2018-19, 2019-20, and 2021-22
  through 2024-25. The 2020-21 season is not included.
- Collapsed to one row per player-season. Kept players with at least 1.5 3PA per
  game and 20 games, which leaves 1,894 player-seasons and 94.3% of all attempts.
  The smallest remaining sample is 33 attempts.

## Results

Fitted priors:

| Season  | Players | α      | β      | Prior mean | α + β |
|---------|--------:|-------:|-------:|-----------:|------:|
| 2018-19 | 302     | 153.49 | 277.69 | 0.3560     | 431.2 |
| 2019-20 | 302     | 137.21 | 244.92 | 0.3591     | 382.1 |
| 2021-22 | 331     | 113.27 | 207.90 | 0.3527     | 321.2 |
| 2022-23 | 312     | 138.63 | 245.00 | 0.3614     | 383.6 |
| 2023-24 | 314     | 168.86 | 290.09 | 0.3679     | 458.9 |
| 2024-25 | 333     | 125.85 | 222.16 | 0.3616     | 348.0 |

On average $\alpha + \beta \approx 390$. The model gives a player's own percentage
half the weight only once he has taken about 390 threes in a season. Few players
ever get there: even Anthony Edwards (811 attempts in 2024-25) is pulled about 30%
of the way toward the league average.

**Validation (season *t* predicting season *t*+1, four consecutive pairs).**
<!-- TODO: paste the ALL PAIRS POOLED table from 04_validate.py here -->
Shrinkage should matter least for high-volume shooters. Even among players with
400+ attempts, shrunk estimates cut next-season RMSE by 9.8% (0.0335 → 0.0302).
They also raise the calibration slope from 0.52 to 0.90, where 1.0 means perfectly
calibrated.

## Limitations

- A single prior per season treats every player as coming from the same pool.
  Poor-shooting players with small samples get pulled up toward league average;
  for example, a 14-for-72 season (.194) is shrunk to .333. A prior that depends
  on position or shot volume, or a two-component mixture, is the natural next step.
- The season-to-season movement in $\alpha + \beta$ (321 to 459) is probably
  estimation noise. Bootstrap confidence intervals would settle it.

## Run it

```bash
uv sync
uv run src/01_pull.py
uv run src/02_clean.py
uv run src/03_model.py
uv run src/04_validate.py
```

Run from the project root. `01_pull.py` caches each season in `data/raw/`, so reruns
skip the API. Data is not committed; `01_pull.py` rebuilds it in about 5–10 minutes.

| Script           | Does                                                        |
|------------------|-------------------------------------------------------------|
| `01_pull.py`     | stats.nba.com → one CSV of game logs per season             |
| `02_clean.py`    | game logs → filtered player-season totals                   |
| `03_model.py`    | fits each season's Beta prior, writes shrunk estimates      |
| `04_validate.py` | scores raw vs. shrunk against the next season               |

---

Nick Zettler · Data Science, University of Iowa
