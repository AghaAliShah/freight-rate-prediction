"""EDA figures used in the report:  python src/eda_figures.py"""
import sys, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from features import build_reference, load, make_features
from model import FreightRateModel
warnings.filterwarnings("ignore")

tr, va = load("data/train_test.csv"), load("data/validation.csv")
ref = build_reference([tr, va]); y = np.log(tr.posted_rate.values)
C = "#064A56"; R = "#C0392B"

# --- quote_signal: correlation with log(distance) by month (rate/mile falls with distance: a faithful quote must too)
cor = lambda g: np.corrcoef(np.log(g.quote_signal), np.log(g.distance))[0, 1]
ct = tr.groupby(tr.date.dt.to_period("M")).apply(cor); cv = va.groupby(va.date.dt.to_period("M")).apply(cor)
rpm = tr.assign(q=tr.quote_signal / (tr.posted_rate / tr.distance)); frac = rpm.groupby(rpm.date.dt.to_period("M")).q.apply(lambda s: (abs(s - 1) < .05).mean())
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
lab = [str(p)[5:] for p in list(ct.index) + list(cv.index)]
ax[0].bar(range(len(lab)), list(frac.values) + [np.nan] * 2, color=C)
ax[0].set_xticks(range(len(lab))); ax[0].set_xticklabels(lab); ax[0].set_ylim(0, 1.05)
ax[0].set_title("Share of loads where quote_signal = posted $/mile (+/-5%)\nTrain: ~98% in Jan-Mar, Jun, Sep; ~30% otherwise"); ax[0].set_xlabel("month 2025 (Nov-Dec = validation, no labels)")
ax[1].bar(range(len(lab)), list(ct.values) + list(cv.values), color=[C] * len(ct) + [R] * len(cv))
ax[1].axhline(0, color="grey", lw=.8); ax[1].set_xticks(range(len(lab))); ax[1].set_xticklabels(lab)
ax[1].set_title("corr(quote_signal, log distance) by month\nsign flips in train; ~0 in validation (red) -> pure noise"); ax[1].set_xlabel("month 2025")
fig.tight_layout(); fig.savefig("outputs/fig_quote_signal.png", dpi=160); plt.close(fig)

# --- quarter-end ramp: residual of the model *without* any calendar term
m = FreightRateModel(ref, use_gbm=False).fit(tr, y); X = make_features(tr, ref); k = m.keep_
X0 = X.assign(dtq=90)                      # switch the quarter-end term off -> residual shows what it has to explain
r = y - m._linear(tr, X0); r = r - np.median(r[k] [X.dtq[k].values > 60])
d = tr[k].assign(r=r[k], dtq=X.dtq[k].values, q=tr.date[k].dt.quarter.values)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for q, g in d.groupby("q"):
    if q < 4: ax.plot(*zip(*g[g.dtq <= 60].groupby("dtq").r.mean().mul(100).items()), label=f"Q{q} 2025", lw=1.8)
ax.invert_xaxis(); ax.set_xlabel("days until quarter end"); ax.set_ylabel("% above model"); ax.legend()
ax.set_title("Rates ramp up ~3 weeks before every quarter end"); fig.tight_layout(); fig.savefig("outputs/fig_quarter_end.png", dpi=160)
