"""Time-aware validation of the freight-rate model and its baselines.

    python src/train_validate.py

Why not a random K-fold?  The 12,000 validation loads are Nov-Dec 2025, *after* every training load (Jan-Oct).  The
final task is therefore a forecast, and the label-generating process drifts over time (market level, trend, quarter-end
surge, a quote_signal column that changes meaning).  A random split would put neighbouring days in train and test and
hide all of that.  We use:

  1. Rolling-origin ("expanding window") folds  - train on everything before a cut-off, test on the next two months
        (Mar-Apr, May-Jun, Jul-Aug, Sep-Oct) - the same 2-month horizon as the real validation file.
  2. Unseen-city folds  - 8 random cities are removed from training *entirely* (every lane touching them) and the model
        is scored on those lanes, mimicking the 8 cities (12% of loads) that exist only in validation.csv.

Two views of each metric are reported: RAW labels (what an external scorer sees) and CLEAN labels (rows whose label is
not one of the ~1.4% corrupted labels - see report).  Corrupted labels are unpredictable by construction, so RAW MAPE is
floored by them.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import build_reference, load, make_features  # noqa: E402
from model import FreightRateModel  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

FOLDS = [("Mar-Apr", 2, [3, 4]), ("May-Jun", 4, [5, 6]), ("Jul-Aug", 6, [7, 8]), ("Sep-Oct", 8, [9, 10])]
CLEAN_CUT = 0.5   # |log(actual/pred)| > 0.5 => corrupted label (inlier residual sd is ~0.02); only used for the figure


def metrics(actual, pred):
    ape = np.abs(pred - actual) / actual
    return dict(MAE=np.mean(np.abs(pred - actual)), RMSE=np.sqrt(np.mean((pred - actual) ** 2)),
                MAPE=100 * ape.mean(), MdAPE=100 * np.median(ape))


# ---------------------------------------------------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------------------------------------------------
class RatePerMile:
    """Naive baseline: median $/mile per equipment x distance decile."""
    def fit(self, df, y_log):
        d = df.assign(rpm=np.exp(y_log) / df["distance"])
        self.edges_ = np.quantile(d["distance"], np.linspace(0, 1, 11)[1:-1])
        d["bin"] = np.digitize(d["distance"], self.edges_)
        self.t_ = d.groupby(["equipment", "bin"])["rpm"].median()
        return self

    def predict(self, df):
        b = np.digitize(df["distance"], self.edges_)
        return df["distance"].values * self.t_.reindex(pd.MultiIndex.from_arrays([df["equipment"], b])).values


class PlainLGBM:
    """Typical 'throw the columns at a GBM' baseline (log target, L1 objective, no cleaning, no structure)."""
    def __init__(self, ref, use_quote=False):
        self.ref, self.use_quote = ref, use_quote
        self.cols = ["distance", "weight", "eq_Dry_Van", "eq_Flatbed", "eq_Reefer", "mi_row", "pickup_lat", "pickup_lon",
                     "delivery_lat", "delivery_lon", "dow"]

    def _x(self, df):
        X = make_features(df, self.ref)[self.cols].copy()
        X["weight"] = df["weight"].values   # raw weight (negatives and cap untouched)
        if self.use_quote:
            X["quote_signal"] = df["quote_signal"].values
        return X

    def fit(self, df, y_log):
        self.m = lgb.LGBMRegressor(objective="regression_l1", n_estimators=600, learning_rate=0.03, num_leaves=31,
                                   min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                                   n_jobs=4).fit(self._x(df), y_log)
        return self

    def predict(self, df):
        return np.exp(self.m.predict(self._x(df)))


def main():
    tr = load("data/train_test.csv")
    va = load("data/validation.csv")
    ref = build_reference([tr, va])   # label-free tables only (daily market index, lane distance, city coordinates)
    y = np.log(tr["posted_rate"].values)
    rate = tr["posted_rate"].values
    month = tr["date"].dt.month.values

    # one model-independent flag for corrupted labels so every contender is scored on the *same* clean rows
    is_clean = FreightRateModel(ref, use_gbm=False).fit(tr, y).keep_
    print(f"corrupted labels flagged: {(~is_clean).sum()} ({100 * (~is_clean).mean():.2f}%)")

    contenders = {
        "Naive $/mile table": lambda: RatePerMile(),
        "LightGBM, raw columns": lambda: PlainLGBM(ref),
        "LightGBM + quote_signal (trap)": lambda: PlainLGBM(ref, use_quote=True),
        "Additive model, no GBM": lambda: FreightRateModel(ref, use_gbm=False),
        "Additive model + GBM (FINAL)": lambda: FreightRateModel(ref),
    }

    rows, oof = [], {}
    for name, make in contenders.items():
        for fold, cut, test_m in FOLDS:
            a, b = month <= cut, np.isin(month, test_m)
            p = make().fit(tr[a], y[a]).predict(tr[b])
            clean = is_clean[b]
            raw_m, cl_m = metrics(rate[b], p), metrics(rate[b][clean], p[clean])
            rows.append(dict(model=name, fold=fold, n=int(b.sum()), **{f"raw_{k}": v for k, v in raw_m.items()},
                             **{f"clean_{k}": v for k, v in cl_m.items()}))
            if name.endswith("(FINAL)"):
                oof.setdefault("idx", []).append(np.where(b)[0]); oof.setdefault("pred", []).append(p)
        print(f"done {name}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "cv_forward_by_fold.csv", index=False)
    summary = res.groupby("model", sort=False).mean(numeric_only=True).drop(columns="n")
    summary.round(3).to_csv(OUT / "cv_forward_summary.csv")
    print("\n=== Rolling-origin summary (mean of 4 folds) ===")
    print(summary.round(2).to_string())

    # ---- unseen cities --------------------------------------------------------------------------------------------
    cities = np.array(sorted(set(tr.pickup) | set(tr.delivery)))
    ur = []
    for name in ("Additive model, no GBM", "Additive model + GBM (FINAL)", "LightGBM, raw columns"):
        for seed in range(5):
            hold = set(np.random.RandomState(seed).choice(cities, 8, replace=False))
            te = (tr.pickup.isin(hold) | tr.delivery.isin(hold)).values
            p = contenders[name]().fit(tr[~te], y[~te]).predict(tr[te])
            clean = is_clean[te]
            ur.append(dict(model=name, seed=seed, n=int(te.sum()), **{f"clean_{k}": v for k, v in metrics(rate[te][clean], p[clean]).items()}))
    ur = pd.DataFrame(ur)
    us = ur.groupby("model", sort=False).mean(numeric_only=True).drop(columns=["seed", "n"])
    us.round(3).to_csv(OUT / "cv_unseen_city_summary.csv")
    print("\n=== Unseen-city holdout (5 draws of 8 cities, clean labels) ===")
    print(us.round(2).to_string())

    # ---- figures ---------------------------------------------------------------------------------------------------
    idx = np.concatenate(oof["idx"]); pred = np.concatenate(oof["pred"])
    o = pd.DataFrame({"date": tr["date"].values[idx], "actual": rate[idx], "pred": pred}).sort_values("date")
    o["logerr"] = np.log(o.actual / o.pred)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    clean = (o.logerr.abs() < CLEAN_CUT).values
    ax[0].scatter(o.pred[clean], o.actual[clean], s=2, alpha=0.25, color="#064A56")
    lim = [50, o.actual[clean].max()]; ax[0].plot(lim, lim, color="#C0392B", lw=1); ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_title("Out-of-time: predicted vs actual (log-log)"); ax[0].set_xlabel("predicted rate ($)"); ax[0].set_ylabel("actual rate ($)")
    ax[1].hist(o.logerr.clip(-2, 2), bins=200, color="#064A56"); ax[1].set_yscale("log")
    ax[1].set_title("log(actual / predicted): a tight core + ~1.4%\ncorrupted labels (x0.2-0.3 or x3-5)"); ax[1].set_xlabel("log error")
    daily = o[clean].groupby("date").logerr.median()
    ax[2].plot(daily.index, daily.values * 100, color="#064A56"); ax[2].axhline(0, color="grey", lw=0.8)
    ax[2].set_title("Daily median % error on held-out folds\n(Mar: no quarter-end seen yet in train -> +4% miss; "
                    "once Q1 is in training, bias stays within ~+/-2%)");ax[2].set_ylabel("%")
    ax[2].tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(OUT / "fig_validation.png", dpi=160); plt.close(fig)

    json.dump({"forward": summary.round(3).to_dict("index"), "unseen_city": us.round(3).to_dict("index")},
              open(OUT / "cv_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
