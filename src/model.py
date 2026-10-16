"""FreightRateModel: robust-label additive model (+ optional GBM residual booster).

log(rate) = spline(log lane distance) + equipment + weight + premium(origin city) + premium(destination city)
            + b * market_index(day) + c * days_since_start + GBM residual correction
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import SplineTransformer

from features import make_features

T0 = pd.Timestamp("2025-01-01")
# |log residual - median| above this => corrupted label (the residual histogram has an empty gap between 0.3 and 0.7)
OUTLIER_CUT = 0.5


class FreightRateModel:
    def __init__(self, ref, use_trend=True, use_gbm=True, gbm_params=None, ridge_alpha=1e-3, clean_labels=True,
                 knn=4, gbm_cols=None, lin_mi=("mi_day",)):
        self.lin_mi = list(lin_mi)
        self.ref, self.use_trend, self.use_gbm, self.alpha = ref, use_trend, use_gbm, ridge_alpha
        self.clean_labels, self.knn = clean_labels, knn
        self.gbm_params = dict(objective="regression", n_estimators=600, learning_rate=0.03, num_leaves=15,
                               min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               reg_lambda=5.0, verbose=-1, n_jobs=4)
        if gbm_params:
            self.gbm_params.update(gbm_params)
        self.gbm_cols = gbm_cols or ["log_dist", "eq_Flatbed", "eq_Reefer", "weight", "mi_day", "mi_7d",
                                     "mi_28d_trail", "dtq", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
                                     "dow", "dist_dev"]

    # ---- design matrix ------------------------------------------------------------------------
    def _city_cols(self, df):
        M = np.zeros((len(df), len(self.cities_)))
        idx = {c: i for i, c in enumerate(self.cities_)}
        for col in ("pickup", "delivery"):
            ii = df[col].map(idx)
            ok = ii.notna().values
            M[np.where(ok)[0], ii[ok].astype(int).values] += 1.0
        return M

    def _base(self, df, X, fit=False):
        if fit:
            # cubic B-spline of log distance; linear extrapolation outside the training range
            self.spl_ = SplineTransformer(n_knots=9, degree=3, knots="quantile", extrapolation="linear", include_bias=True)
            self.spl_.fit(X[["log_dist"]])
            self.qknots_ = np.array([[0.0], [3.0], [7.0], [12.0], [18.0], [26.0], [40.0], [60.0], [90.0]])
            self.qspl_ = SplineTransformer(degree=3, knots=self.qknots_, extrapolation="constant", include_bias=False)
            self.qspl_.fit(X[["dtq"]])
        S = self.spl_.transform(X[["log_dist"]])
        w = X["weight"].fillna(self.wmed_) / 1e4
        cols = [S, X[["eq_Flatbed", "eq_Reefer"]].values, w.values[:, None],
                X[["weight_capped", "weight_missing", "weight_neg"]].values, X[self.lin_mi].values,
                self.qspl_.transform(X[["dtq"]]), np.eye(7)[X["dow"].values][:, 1:]]
        if self.use_trend:
            cols.append(((df["date"] - T0).dt.days.values / 100.0)[:, None])
        return np.hstack(cols)

    def fit(self, df, y_log):
        X = make_features(df, self.ref)
        self.wmed_ = float(X["weight"].median())
        self.cities_ = sorted(set(df["pickup"]) | set(df["delivery"]))
        keep = np.ones(len(df), bool)
        for it in range(2 if self.clean_labels else 1):
            base = self._base(df[keep], X[keep], fit=True)
            A = np.hstack([base, self._city_cols(df[keep])])
            self.rd_ = Ridge(alpha=self.alpha, fit_intercept=True).fit(A, y_log[keep])
            if self.clean_labels and it == 0:
                r = y_log - self.rd_.predict(np.hstack([self._base(df, X), self._city_cols(df)]))
                keep = np.abs(r - np.median(r)) < OUTLIER_CUT
        self.keep_ = keep
        nb = base.shape[1]
        prem = pd.Series(self.rd_.coef_[nb:], index=self.cities_)
        # identifiability: every row contains exactly two city terms -> centre premiums, move the mean to the intercept
        shift = prem.mean()
        self.rd_.intercept_ = self.rd_.intercept_ + 2 * shift
        prem = prem - shift
        self.rd_.coef_[nb:] = prem.values
        self.prem_ = prem
        # geo model for cities never seen in training (premium is a smooth function of position)
        cc = self.ref["cities"].loc[self.cities_, ["lat", "lon"]]
        self.geo_ = KNeighborsRegressor(n_neighbors=self.knn, weights="distance").fit(cc.values, prem.values)

        self.gbm_ = None
        if self.use_gbm:
            resid = y_log[keep] - self._linear(df[keep], X[keep])
            self.gbm_ = lgb.LGBMRegressor(**self.gbm_params).fit(X.loc[keep, self.gbm_cols], resid)
        return self

    def _linear(self, df, X):
        A = np.hstack([self._base(df, X), self._city_cols(df)])
        out = self.rd_.predict(A)
        for col in ("pickup", "delivery"):
            m = ~df[col].isin(self.cities_)
            if m.any():
                cc = self.ref["cities"].loc[df.loc[m, col].unique(), ["lat", "lon"]]
                pred = pd.Series(self.geo_.predict(cc.values), index=cc.index)
                out = out + np.where(m.values, df[col].map(pred).fillna(0.0).values, 0.0)
        return out

    def predict_log(self, df):
        X = make_features(df, self.ref)
        out = self._linear(df, X)
        if self.gbm_ is not None:
            out = out + self.gbm_.predict(X[self.gbm_cols])
        return out

    def predict(self, df):
        return np.exp(self.predict_log(df))
