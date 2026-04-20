# Day 1 — Time Series Foundations

## What I learned
- STL decomposition separates trend, seasonal, and residual. Multiplicative series become additive in log-space.
- Residual diagnostics (time plot, histogram, ACF, Ljung-Box) are required to validate any fitted model — "looks random" is not a test, Ljung-Box is.
- Baseline selection depends on horizon: SeasonalNaive wins short, RandomWalkWithDrift wins long when trend dominates.
- Nixtla libraries require `unique_id`, `ds`, `y` columns — forces long format and scales to many series.

## What surprised me
- RandomWalkWithDrift beat SeasonalNaive by ~50% on MAPE over a 60-month horizon. I expected seasonality to matter more than trend; over long horizons, trend compounds and wins.

## What I want to revisit
- Why does `unique_id` come as float64 in AirPassengersDF? When would this break at scale?
- The lag-1 autocorrelation of 0.3 in STL residuals — SARIMA AR(1) should handle this. Verify tomorrow.

## Floor-to-beat (this dataset)
- RWD MAPE = 0.14. Any model I try must beat this meaningfully.

## Libraries used:
 - General imports pandas, numpy, matplotlib
 - statsmodels for tsa, acf, pacf, acorr_ljungbox, plot_acf, plot_pacf
- statsforecast for StatsForecast and models - Naive, SeasonalNaive, WindowAverage, HistoricAverage, RandomWalkWithDrift

"Five naive baselines vs. 60-month test. RWD (light blue) wins not by being smart but by tracking the level correctly. SeasonalNaive (pink) matches seasonal shape but gets the level wrong. Trend dominates seasonality over long horizons — the key Day 1 insight."
