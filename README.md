# Freight Rate Prediction Challenge - solution

Predicts `posted_rate` for 12,000 loads (Nov-Dec 2025) from 48,000 labeled loads (Jan-Oct 2025).

**Result in one paragraph.** Out-of-time validation (train on the past, predict the next two months) gives a
**median error of 1.4 % and a MAPE of 1.7 % on clean labels (4.1 % raw, because ~1.4 % of labels are corrupted)**,
vs 3.3 % / 5.6 % for a plain LightGBM on the raw columns. The key to this is not model complexity but data work:
dropping `quote_signal` (it silently changes meaning over time), cleaning corrupted labels, using row distance, and
modelling the quarter-end rate surge. See `report/Freight_Rate_Report.pdf`.

## Layout
```
data/                  (not committed) put the provided input files here, see Run
src/features.py        feature engineering + label-free reference tables
src/model.py           FreightRateModel (additive model + GBM residual booster)
src/train_validate.py  time-aware validation, baselines, unseen-city test  -> outputs/
src/predict.py         final fit on all labeled data -> validation_predictions.csv + December inputs
src/eda_figures.py     figures for the report
src/make_report.py     builds report/Freight_Rate_Report.pdf
validation_predictions.csv   the 12,000 predictions (load_id,predicted_rate)
outputs/               validation tables/figures, december_predictions.csv
report/                PDF report
scorer_results/        candidate_december.png (made by score.py)
```

## Run
```bash
python -m pip install -r requirements.txt
python src/train_validate.py     # ~3 min: cross-validation tables + figure in outputs/
python src/predict.py            # fits final model, writes validation_predictions.csv, fills data/december_chart_inputs.csv
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python src/eda_figures.py && python src/make_report.py
```
Data files are expected in `data/` with underscore names (`train_test.csv`, `validation.csv`,
`validation_predictions_template.csv`, `december_chart_inputs.csv`).

## Method (short)
* **Split:** rolling-origin folds (train on everything before a cut, test on the next 2 months) + an unseen-city holdout.
  Never a random split - the target is a forecast and the process drifts.
* **Data issues handled:** `quote_signal` excluded; ~1.4 % corrupted labels (x0.2-0.3 / x3-5) detected and excluded from
  training; negative weights = sign errors (abs); weight capped at 47,500 and missing; missing `market_index`
  (replaced by a robust daily median); distance floor of 70 mi; per-row distance noise (rate follows the row distance);
  8 cities only in validation (premium interpolated from coordinates); synthetic city coordinates.
* **Model:** log(rate) = spline(log distance) + equipment + weight + origin/destination city premium + market index +
  linear trend + quarter-end ramp + day-of-week, fitted with ridge on clean labels, then a small LightGBM learns the
  remaining residual structure (no time/trend features, so it cannot extrapolate wrongly).
