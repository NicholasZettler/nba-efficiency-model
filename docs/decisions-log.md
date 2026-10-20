# Decisions log

## 2026-10-20: Add a forecast monitoring layer (Power BI + Excel)
- Why: target posting (Ingredion, Supply Chain Data Science Intern) asks for ML
  forecasts, anomaly detection in reporting, KPI driver analysis, Excel, and Power BI.
  The pipeline already forecasts; this presents it the way an operations team
  monitors a demand forecast.
- Alternatives considered: Excel-first stakeholder pack with a light Power BI page
  (less depth in the tool the posting names twice).

## 2026-10-20: Anomaly rule = 95% Beta-Binomial posterior predictive interval
- Rule: flag season t+1 if its makes fall outside the central 95% interval of
  BetaBinomial(n_next, alpha + x, beta + n - x).
- Why: comes straight from the existing model, so the flag rate doubles as a check
  on the model's honesty. A z-score on raw 3P% would ignore attempts on both sides.
- Discreteness makes the interval conservative: on data simulated from the model,
  coverage is 95.8%, so the expected flag rate is ~4.2%, not 5%.
- Result on real data (2,076 pairs): 4.6% flagged (45 hot, 50 cold), every season
  pair between 2.0% and 6.6%. PIT deciles are flat (8.8%–10.9%). The intervals are
  close to calibrated; low-volume shooters (<200 3PA) are flagged slightly more
  often (5.6–5.8%), consistent with EB trusting them a bit too much.
- Mid-point PIT used for the calibration histogram because makes are discrete.

## 2026-10-20: Reuse processed CSVs instead of re-pulling
- Copied data/processed from the older local copy (Documents/nba-shrinkage); row
  counts match the README (2,958 / 2,076 / 1,243). Saves the 15–20 minute API pull.
