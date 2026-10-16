"""Feature engineering shared by validation experiments, final training and prediction.

Design rules (each one is backed by a finding from the EDA, see report):
  * `quote_signal` is NEVER used: it equals rate/mile in some months, is anti-correlated with the
    rate in others, and is pure noise in the Nov-Dec validation file.
  * No absolute time index / day-of-year: validation (Nov-Dec) lies outside the training range, a
    tree model cannot extrapolate on it, and it would only memorise train-period seasonality.
  * Only label-free aggregates (daily market_index, lane median distance) cross the train/validation
    boundary, so there is no target leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]
WEIGHT_CAP = 47_500.0


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def load(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_reference(frames: list[pd.DataFrame]) -> dict:
    """Label-free lookup tables built from every file that contains features (train + validation)."""
    allx = pd.concat([f.drop(columns=["posted_rate"], errors="ignore") for f in frames], ignore_index=True)

    # market_index: noisy per row and ~7% missing -> robust daily level, smoothed and lagged versions
    daily = allx.groupby("date")["market_index"].median().sort_index()
    full = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D")).interpolate(limit_direction="both")
    mi = pd.DataFrame({"mi_day": full})
    mi["mi_3d"] = full.rolling(3, center=True, min_periods=1).mean()
    mi["mi_7d"] = full.rolling(7, center=True, min_periods=1).mean()
    mi["mi_14d_trail"] = full.rolling(14, min_periods=1).mean()
    mi["mi_28d_trail"] = full.rolling(28, min_periods=1).mean()
    mi["mi_chg_14d"] = full - full.shift(14).bfill()

    # lane median distance (de-noises the +-2% per-row distance noise and the clipped 70mi floor)
    allx["lane"] = allx["pickup"] + ">" + allx["delivery"]
    lane = allx.groupby("lane")["distance"].agg(lane_dist="median", lane_n="size")
    # city coordinates (one fixed pair per city)
    c1 = allx[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1)
    c2 = allx[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1)
    cities = pd.concat([c1, c2]).drop_duplicates("city").set_index("city")
    return {"mi": mi, "lane": lane, "cities": cities}


def make_features(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    df = df.copy()
    # rows that only carry city names (e.g. the fixed December chart inputs): look coordinates up, no row-level index
    for role in ("pickup", "delivery"):
        for c in ("lat", "lon"):
            if f"{role}_{c}" not in df:
                df[f"{role}_{c}"] = df[role].map(ref["cities"][c])
    if "market_index" not in df:
        df["market_index"] = np.nan
    X = pd.DataFrame(index=df.index)
    lane_key = df["pickup"] + ">" + df["delivery"]

    # --- distance ------------------------------------------------------------------------------
    X["distance"] = df["distance"]
    X["log_dist"] = np.log(df["distance"])
    ld = lane_key.map(ref["lane"]["lane_dist"])
    ln = lane_key.map(ref["lane"]["lane_n"]).fillna(0)
    # use lane median only when it is backed by >=3 observations, otherwise the row distance
    X["lane_dist"] = np.where(ln >= 3, ld, df["distance"])
    X["log_lane_dist"] = np.log(X["lane_dist"])
    X["dist_dev"] = df["distance"] / X["lane_dist"]
    X["short_floor"] = (df["distance"] <= 70.0).astype(int)
    X["geo_dist"] = haversine(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)

    # --- weight: sign flips (abs), 47,500 cap, missing -----------------------------------------
    w = df["weight"]
    X["weight"] = w.abs()                     # negative values are sign errors (|w| distribution matches)
    X["weight_missing"] = w.isna().astype(int)
    X["weight_neg"] = (w < 0).astype(int)
    X["weight_capped"] = (w.abs() >= WEIGHT_CAP).astype(int)

    # --- equipment -----------------------------------------------------------------------------
    for e in EQUIPMENT:
        X[f"eq_{e.replace(' ', '_')}"] = (df["equipment"] == e).astype(int)
    X["equipment"] = pd.Categorical(df["equipment"], categories=EQUIPMENT)

    # --- market --------------------------------------------------------------------------------
    X["mi_row"] = df["market_index"]
    X["mi_missing"] = df["market_index"].isna().astype(int)
    m = ref["mi"].reindex(df["date"].values)
    for c in m.columns:
        X[c] = m[c].values

    # --- calendar (cyclic effects only; no trend / day-of-year) --------------------------------
    X["dow"] = df["date"].dt.dayofweek
    X["is_weekend"] = (X["dow"] >= 5).astype(int)
    # freight rates ramp up over the last ~3 weeks of every quarter (seen in Q1, Q2, Q3 of the training data;
    # Q4 -> December is the same mechanism).  Days-to-quarter-end transfers to unseen months, month dummies would not.
    X["dtq"] = ((df["date"] + pd.offsets.QuarterEnd(0)) - df["date"]).dt.days.clip(upper=90)

    # --- geography (coordinates generalise to cities that never appear in train) ---------------
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        X[c] = df[c]
    return X
