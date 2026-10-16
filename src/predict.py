"""Fit the final model on ALL labeled loads, then write:
      validation_predictions.csv            (load_id,predicted_rate - 12,000 rows, template order)
      data/december_chart_inputs.csv        (predicted_rate filled for the 31 fixed December rows)

    python src/predict.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import build_reference, load  # noqa: E402
from model import FreightRateModel  # noqa: E402


def main():
    tr = load("data/train_test.csv")
    va = load("data/validation.csv")
    ref = build_reference([tr, va])          # label-free: daily market index, lane distance, city coordinates

    model = FreightRateModel(ref).fit(tr, np.log(tr["posted_rate"].values))
    print(f"trained on {model.keep_.sum():,} clean rows; {(~model.keep_).sum()} corrupted labels excluded")

    # ---- 12,000 validation loads -----------------------------------------------------------------------------------
    pred = model.predict(va)
    out = pd.read_csv("data/validation_predictions_template.csv")[["load_id"]]
    out = out.merge(pd.DataFrame({"load_id": va["load_id"], "predicted_rate": np.round(pred, 2)}), on="load_id", how="left")
    assert len(out) == 12_000 and out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all()
    out.to_csv("validation_predictions.csv", index=False)
    print("validation predictions:", out["predicted_rate"].describe().round(1).to_dict())

    # ---- fixed December chart (only the date changes) --------------------------------------------------------------
    dec = pd.read_csv("data/december_chart_inputs.csv")
    d = dec.drop(columns="predicted_rate").assign(date=lambda x: pd.to_datetime(x["date"]))
    dec["predicted_rate"] = np.round(model.predict(d), 2)
    dec.to_csv("data/december_chart_inputs.csv", index=False)
    print(dec[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
