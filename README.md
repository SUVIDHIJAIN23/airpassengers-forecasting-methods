# AirPassengers Forecasting: A Methodological Comparison

A systematic comparison of classical and machine learning forecasting 
methods on the Box-Jenkins airline passengers dataset (1949-1960, 
monthly). This repository documents the tradeoffs between baseline, 
statistical (ETS, SARIMA), and gradient-boosted (LightGBM) approaches 
through residual diagnostics, rolling-origin cross-validation, and 
target transformation experiments.

## Headline Result

Rolling-origin cross-validated MAPE across 5 folds (1955-1959 test windows):

| Approach | Single-split MAPE | Notes |
|---|---|---|
| Random Walk with Drift (baseline) | 0.140 | Trend only, no seasonality |
| ETS MMM (raw-y) / MAM (log-y) | 0.096 | ETS ceiling across 6 specifications |
| AutoARIMA (2,0,0)(2,1,0)[12] | 0.075 | Beats manual Box-Jenkins selection |
| **LightGBM + log + differences [1,12]** | **0.037** | **Best single-split** |
| LightGBM (5-fold CV) | **0.060** | Robustly evaluated |

The LightGBM approach achieved a **2.3× improvement** in MAPE over 
classical statistical methods on held-out evaluation, via target 
transformations that address the tree-based extrapolation limitation.

## What This Project Demonstrates

**Classical forecasting workflow**
- Hypothesis-driven EDA before plotting
- STL decomposition and residual diagnostics
- Stationarity testing (ADF + KPSS) with opposing null hypotheses
- Box-Jenkins order identification from ACF/PACF patterns
- Conformal prediction intervals for models without analytical PIs

**Model comparison rigor**
- Six ETS specifications compared on raw-y and log-y
- Five SARIMA variants including AutoARIMA
- LightGBM with and without target transforms
- Residual diagnostics (time plot, histogram, ACF, Ljung-Box) at every stage

**Evaluation methodology**
- Rolling-origin cross-validation (not single train/test split)
- Per-fold error analysis revealing monotonic growth with horizon
- Scale-consistent comparison (back-transformation from log scale)

## Key Insights

**The overfitting paradox (ETS)**  
AutoETS on raw-y had the best in-sample AICc (767) but the worst test 
MAPE (0.227) — 2.3× worse than the manually-chosen MMM (0.100). AICc's 
complexity penalty under-penalized on 84 training observations, selecting 
a model that absorbed training noise. Domain-informed manual specification 
outperformed automated selection on this short series.

**Transform-spec coupling**  
Log-transform and multiplicative ETS components are alternative solutions 
to multiplicative seasonality, not complementary. Combining them 
(log-y + MMM) produced worse results than either path alone (raw-y + MMM 
or log-y + AAA). Preprocessing and model specification must be coherent.

**Horizon changes the winner**  
Over 60-month forecasts, RandomWalkWithDrift (trend only) beat 
SeasonalNaive (seasonality only) by ~50%. Long horizons favor correct 
trend over correct seasonality because trend errors compound.

**Tree-based extrapolation failure**  
A naive LightGBM on raw y produced forecasts locked at training-era 
levels — correct seasonal shape but wrong level, because trees cannot 
predict outside their training range. The fix: log + seasonal 
differencing, which models growth rates (scale-invariant) instead of 
absolute values.

**Single-split evaluation is optimistic**  
5-fold rolling-origin CV produced MAPE 0.060 vs. 0.037 on a single split 
— a 62% inflation. Per-fold errors grew monotonically (1.6% → 10.8%) as 
later folds forecast into the steeper-growth end of the series.

## Methods Compared

**Baselines (Section 4)**
- Naive, SeasonalNaive, HistoricAverage, RandomWalkWithDrift

**ETS family (Section 5)**
- Manual: MAM, MAdM (damped), MMM, MMdM (damped), AAA
- Automated: AutoETS
- Evaluated on both raw-y and log-y for coherence testing

**ARIMA family (Section 6)**
- Manual: SARIMA(1,1,1)(0,1,1)[12], SARIMA(0,1,1)(0,1,1)[12], 
  SARIMA(1,1,1)(1,1,1)[12], SARIMA(2,1,2)(0,1,1)[12]
- Automated: AutoARIMA (selected (2,0,0)(2,1,0)[12])

**Machine Learning (Section 7)**
- LightGBM with lag features (1, 3, 9, 12), rolling means, calendar features
- With and without target transforms (Differences([1, 12]))
- With and without log-transform

**Cross-Validation (Section 8)**
- mlforecast.cross_validation with h=12, n_windows=5, step_size=12

## What This Project Does Not Claim to Be

- **A production system.** No deployment, no monitoring, no retraining pipeline.
- **A novel methodology.** These are standard techniques from Hyndman, Nixtla documentation, and M-competition literature.
- **Multi-series scalable.** Single series only. For hierarchical retail forecasting at scale, see the M5 companion project [link forthcoming].
- **A definitive comparison.** Results are specific to this dataset; findings generalize with caveats.

## Reproduction

```bash
git clone https://github.com/SUVIDHIJAIN23/airpassengers-forecasting-methods.git
cd airpassengers-forecasting-methods
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
jupyter notebook forecasting_methods.ipynb
```

Notebook runs top-to-bottom with `Restart and Run All`. Runtime: ~5 minutes on a standard laptop.

## Repository Structure
