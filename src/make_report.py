"""Build report/Freight_Rate_Report.pdf from the saved outputs (run after train_validate, predict, score, eda_figures)."""
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

Path("report").mkdir(exist_ok=True)
ss = getSampleStyleSheet()
B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.6, leading=13)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=17, textColor=colors.HexColor("#064A56"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, textColor=colors.HexColor("#064A56"), spaceBefore=10)
SM = ParagraphStyle("sm", parent=B, fontSize=8.3, leading=11, textColor=colors.HexColor("#455A60"))
bullet = lambda t: Paragraph("&bull;&nbsp;" + t, ParagraphStyle("bl", parent=B, leftIndent=12, firstLineIndent=-9))


def img(path, w=6.9):
    from PIL import Image as PI
    iw, ih = PI.open(path).size
    return Image(path, width=w * inch, height=w * inch * ih / iw)


def table(df, widths=None, bold=-1):
    t = Table([list(df.columns)] + df.values.tolist(), colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#064A56")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#9DAFB3")),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF3F4")]),
                           ("FONTNAME", (0, bold), (-1, bold), "Helvetica-Bold")]))
    return t


fw = pd.read_csv("outputs/cv_forward_summary.csv", index_col=0)
fold = pd.read_csv("outputs/cv_forward_by_fold.csv")
uc = pd.read_csv("outputs/cv_unseen_city_summary.csv", index_col=0)
dec = pd.read_csv("data/december_chart_inputs.csv")

res = fw[["raw_MAE", "raw_MAPE", "raw_MdAPE", "clean_MAE", "clean_MAPE", "clean_MdAPE"]].round(2).reset_index()
res.columns = ["Model", "MAE $", "MAPE %", "MdAPE %", "MAE $ (clean)", "MAPE % (clean)", "MdAPE % (clean)"]
fin = fold[fold.model.str.contains("FINAL")][["fold", "n", "raw_MAE", "raw_MAPE", "clean_MAE", "clean_MAPE", "clean_MdAPE"]].round(2)
fin.columns = ["Test fold (train = all earlier data)", "Loads", "MAE $", "MAPE %", "MAE $ (clean)", "MAPE % (clean)", "MdAPE % (clean)"]
ucr = uc[["clean_MAE", "clean_MAPE", "clean_MdAPE"]].round(2).reset_index(); ucr.columns = ["Model", "MAE $", "MAPE %", "MdAPE %"]

S = []
S += [Paragraph("Freight Rate Prediction - Approach, Validation and December Forecast", H1),
      Paragraph("Machine Learning Engineer assessment. Code: see GitHub repository (README.md for run instructions).", SM), Spacer(1, 6)]
S += [Paragraph("1. Summary", H2),
      Paragraph("Task: predict <b>posted_rate</b> for 12,000 loads dated Nov-Dec 2025 using 48,000 labeled loads from Jan-Oct 2025. "
                "Because validation lies entirely <i>after</i> training, this is a two-month-ahead forecast, and the data-generating process "
                "drifts over time. The result is driven by data understanding rather than model size: on out-of-time folds the final model "
                "reaches <b>1.4% median error / 1.7% MAPE on clean labels</b> (4.1% MAPE on raw labels, floored by ~1.4% corrupted labels that no model can "
                "predict), versus 3.0% / 3.3% for a plain LightGBM and 4.3% / 4.7% for a $/mile table. The estimated noise floor is about 1.8% MAPE.", B)]

S += [Paragraph("2. Data exploration and data-quality issues", H2),
      bullet("<b>quote_signal is a trap.</b> In Jan-Mar, Jun and Sep it equals posted $/mile to within ~1% (it would be a near-perfect leak). In Apr-May, "
             "Jul-Aug and Oct it is unrelated or inversely related to the rate. In validation it is pure noise: no relation with distance, equipment or the previous day "
             "(day-to-day autocorrelation 0.02 vs 0.93 in train). A model trusting it looks great in some CV folds and fails on the final data. "
             "<b>Decision: excluded.</b> (The 'LightGBM + quote_signal' row below is that experiment: it is worse, not better.)"),
      bullet("<b>~1.4% corrupted labels (677 rows).</b> Rates are off by a factor of 0.2-0.3 or 3-5.5 (random up/down) with no relation to any feature; the residual "
             "histogram has an empty gap between normal rows (|log error| &lt; 0.3) and these rows (&gt; 0.7). They are flagged by a first-pass fit and excluded from training "
             "(without this step the linear component degrades from 1.7% to 7.3% clean MAPE). They still count in the RAW metrics below."),
      bullet("<b>Weight:</b> 292 negative values (sign errors: |w| matches the positive distribution and the rate effect), a hard cap at 47,500 lb (1,204 rows), 300 missing. "
             "Handled with abs(), a cap flag and a missing flag; the rate rises ~3% per 10,000 lb."),
      bullet("<b>market_index:</b> noisy per row and 0.8% missing. Replaced by a robust <i>daily median</i> (built from features only, so no label leakage); "
             "the daily level drives the rate (+0.14 log-points per index point)."),
      bullet("<b>Distance:</b> each row carries ~2% noise around its lane's typical distance, and the rate follows the <i>row</i> distance (using the lane median instead "
             "raised residual noise by ~30%: sd 0.030 vs 0.0225); 69 rows are clipped at a 70-mile floor. City coordinates are one fixed (synthetic/distorted) pair per city; haversine vs distance is erratic for some "
             "lanes, so distance is trusted and coordinates are used only as location features."),
      bullet("<b>Unseen cities:</b> validation contains 8 cities absent from train (Chicago, Charlotte, Jackson, Norfolk, Laredo, Allentown, Knoxville, San Diego; 12% of loads). "
             "The origin and destination premium of a city are almost identical (corr 0.99) and fall with latitude (south-central cities +6%), so the premium of an unseen city is interpolated from nearby cities' coordinates."),
      bullet("<b>Calendar structure.</b> Rates ramp up over the last ~30 days of every quarter, peaking ~+4% at quarter end - identical in Q1, Q2, Q3 (figure below). "
             "Modeled with days-to-quarter-end (transfers to Q4/December; month dummies would not). A trend of ~+0.02%/day and a small day-of-week effect are also present."),
      bullet("No duplicates or missing ids; Nov-Dec market_index (0.93) lies within the training range.")]
S += [img("outputs/fig_quote_signal.png", 6.9), Spacer(1, 4), img("outputs/fig_quarter_end.png", 4.3)]

S += [PageBreak(), Paragraph("3. How the data was split and validated", H2),
      Paragraph("A random K-fold would place neighbouring days in both train and test, hide the drift, and reward the leaky quote_signal. "
                "Instead the validation mimics the real task:", B),
      bullet("<b>Rolling-origin (expanding-window) folds:</b> train on all data before a cut-off, test on the next two months (same horizon as Nov-Dec): "
             "Mar-Apr, May-Jun, Jul-Aug, Sep-Oct. Daily market-index aggregates are label-free and are the only information shared across the boundary."),
      bullet("<b>Unseen-city holdout:</b> 8 random cities (5 draws) are removed from training with every lane touching them, then scored - mirrors the 8 validation-only cities."),
      bullet("<b>Two views of each metric:</b> RAW labels (what an external scorer sees) and CLEAN labels (corrupted rows removed, <i>same rows for every model</i>). "
             "The metric used by the grader is not published, so MAE, MAPE and MdAPE are all reported. Predictions are the conditional median in log space, which is MAE-optimal and close to MAPE-optimal."),
      bullet("<b>Final fit</b> uses all 48,000 labeled loads; the last fold (Sep-Oct) is the best proxy of Nov-Dec performance. Hyper-parameters barely matter (plateau between 150 and 600 trees), "
             "so there is little tuning-on-test risk."),
      Spacer(1, 4), Paragraph("Rolling-origin results (mean of 4 folds)", H2), table(res, [1.9 * inch, .6 * inch, .65 * inch, .65 * inch, .85 * inch, .9 * inch, .95 * inch]),
      Spacer(1, 3), Paragraph("Final model by fold", H2), table(fin, [2.0 * inch, .5 * inch, .6 * inch, .6 * inch, .85 * inch, .95 * inch, .95 * inch]),
      Spacer(1, 3), Paragraph("Unseen-city holdout (clean labels)", H2), table(ucr, [2.5 * inch, .9 * inch, .9 * inch, .9 * inch], bold=2),
      Spacer(1, 6), img("outputs/fig_validation.png", 6.9),
      Paragraph("The Mar-Apr fold is the weakest because the training window (Jan-Feb) contains no quarter end yet, so the late-March surge is missed (+4% bias); "
                "after Q1 is in training the daily bias stays within about +/-2%.", SM)]

S += [Paragraph("4. Model", H2),
      Paragraph("<b>log(rate) = spline(log distance) + equipment + weight terms + premium(origin) + premium(destination) + b*market_index(day) + c*days + quarter-end spline + day-of-week</b>, "
                "fitted with ridge on clean labels; a small LightGBM (600 trees, 15 leaves) then models the residual using location, distance, weight, equipment, market level and day-of-week "
                "(deliberately without a time index so it cannot extrapolate wrongly). Why: the data are almost exactly additive in log space, so a structured model extrapolates in time "
                "(trend, quarter-end) and to unseen cities, which tree ensembles cannot; the GBM only mops up small interactions (-7% clean MAE). Plain GBMs lose 2x in accuracy because they memorise the corrupted labels, "
                "cannot extrapolate the trend, and (with quote_signal) learn a relationship that disappears.", B)]

S += [PageBreak(), Paragraph("5. Fixed December prediction chart (produced by the provided score.py)", H2),
      img("scorer_results/candidate_december.png", 6.9),
      Paragraph(f"Predictions range from ${dec.predicted_rate.min():.0f} (Dec 1) to ${dec.predicted_rate.max():.0f} (Dec 31). For the 31 fixed rows, the daily market-index level is looked up for each date "
                "(from the validation file's features), which creates the weekly wiggle; the rise through the month is the quarter-end ramp plus the trend. "
                "Limitations: the quarter-end pattern is extrapolated to Q4 from three quarters; holiday effects (Thanksgiving, Christmas) cannot be learned from Jan-Oct data and are not modeled.", B)]
SimpleDocTemplate("report/Freight_Rate_Report.pdf", pagesize=letter, leftMargin=.7 * inch, rightMargin=.7 * inch, topMargin=.6 * inch, bottomMargin=.6 * inch,
                  title="Freight Rate Prediction Report").build(S)
print("wrote report/Freight_Rate_Report.pdf")
