"""Build compact JS data bundle (data/dashboard_data.js) for the E-Commerce Telemetry Dashboard."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

raw_df = pd.read_csv(ROOT / "ecommerce_raw.csv")
df = pd.read_csv(ROOT / "ecommerce_clean.csv")
metrics_df = pd.read_csv(ROOT / "model_metrics.csv", index_col=0)
preds_df = pd.read_csv(ROOT / "predictions.csv")

num_cols = df.select_dtypes("number").columns.tolist()
train_i, test_i = train_test_split(df.index, test_size=0.2, random_state=42)
tr, te = df.loc[train_i], df.loc[test_i]
y_tr, y_te = tr["Purchase_Amount"].values, te["Purchase_Amount"].values

simple_f = ["Time_On_Site"]
multi_f = ["Time_On_Site", "Visits_Per_Month", "Pages_Viewed", "Cart_Items", "Previous_Purchase_Amount"]

m_simple = LinearRegression().fit(tr[simple_f], y_tr)
m_multi = LinearRegression().fit(tr[multi_f], y_tr)
cv_simple = cross_val_score(LinearRegression(), tr[simple_f], y_tr, cv=KFold(5, shuffle=True, random_state=1), scoring="r2")
cv_multi = cross_val_score(LinearRegression(), tr[multi_f], y_tr, cv=KFold(5, shuffle=True, random_state=1), scoring="r2")

# Gradient descent histories matching Task 7
M = tr[multi_f].values.astype(float)
mu, sd = M.mean(0), M.std(0)
Z = np.column_stack([np.ones(len(M)), (M - mu) / sd])

def descend(alpha, iters):
    w = np.zeros(Z.shape[1])
    hist = [w.copy()]
    costs = [float(np.sum((Z @ w - y_tr) ** 2) / (2 * len(y_tr)))]
    for _ in range(iters):
        w = w - alpha * (Z.T @ (Z @ w - y_tr)) / len(y_tr)
        hist.append(w.copy())
        costs.append(float(np.sum((Z @ w - y_tr) ** 2) / (2 * len(y_tr))))
    return w, np.array(hist), costs

gd_curves = {}
for al in [0.01, 0.05, 0.2]:
    _, _, j_hist = descend(al, 200)
    gd_curves[str(al)] = [round(v, 2) for v in j_hist]

w_1000, _, j_1000 = descend(0.05, 1000)
orig_gd = np.r_[w_1000[0] - np.sum(w_1000[1:] * mu / sd), w_1000[1:] / sd]
X_ne = np.column_stack([np.ones(len(M)), M])
theta_ne = np.linalg.solve(X_ne.T @ X_ne, X_ne.T @ y_tr)
gd_max_diff = float(np.abs(orig_gd - theta_ne).max())

# Test predictions map by Customer_ID
pred_map = dict(zip(preds_df["Customer_ID"], preds_df["Predicted"].round(2)))
test_id_set = set(preds_df["Customer_ID"])

# Q-Q points for test residuals (sampled evenly to 80 points for crisp SVG)
osm, osr = stats.probplot(preds_df["Residual"].values, dist="norm")
qq_idx = np.linspace(0, len(osm[0]) - 1, 80, dtype=int)
qq_pts = [[round(float(osm[0][i]), 3), round(float(osm[1][i]), 1)] for i in qq_idx]
qq_slope = float(osr[0])
qq_intercept = float(osr[1])

# LLN random draw indices (seed 7, matching Task 3) so browser can recompute LLN for any cutoff
y_all = df["Purchase_Amount"].values
lln_draw_idx = np.random.default_rng(7).integers(0, len(df), size=10000).tolist()
# Store pre-sampled 120 log-spaced indices (0..9999) + full draws only needed if exact? Actually 10,000 ints is tiny (~55KB)
# Keep full 10,000 drawn row indices so moving the High Spender cutoff updates the LLN curve live!

# Compact row encoding:
# [Customer_ID, gender(0=F,1=M), age, membership(0=Reg,1=Prem), device(0=Mob,1=Desk,2=Tab),
#  visits, time, pages, cart, prev, disc, rating, amount, pred_or_null]
g_map = {"Female": 0, "Male": 1}
m_map = {"Regular": 0, "Premium": 1}
d_map = {"Mobile": 0, "Desktop": 1, "Tablet": 2}

rows = []
for r in df.itertuples(index=False):
    cid = r.Customer_ID
    pred_val = float(pred_map[cid]) if cid in test_id_set else None
    rows.append([
        cid,
        g_map[r.Gender],
        int(r.Age),
        m_map[r.Membership],
        d_map[r.Device],
        int(r.Visits_Per_Month),
        round(float(r.Time_On_Site), 1),
        int(r.Pages_Viewed),
        int(r.Cart_Items),
        round(float(r.Previous_Purchase_Amount), 2),
        int(r.Discount_Percent),
        int(r.Review_Rating),
        round(float(r.Purchase_Amount), 2),
        pred_val,
    ])

# Residual bands and Shapiro-Wilk
bands = pd.qcut(preds_df["Actual"], 3, labels=["Low", "Medium", "High"])
band_means = preds_df.groupby(bands, observed=True)["Residual"].mean().round(2).to_dict()
shapiro_p = float(stats.shapiro(preds_df["Residual"]).pvalue)

# Welch t-test (full population)
a = df[df["Membership"] == "Premium"]["Purchase_Amount"]
b = df[df["Membership"] == "Regular"]["Purchase_Amount"]
t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
cohen_d = float((a.mean() - b.mean()) / np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2)))

payload = {
    "meta": {
        "rawRows": int(len(raw_df)),
        "cleanRows": int(len(df)),
        "cols": int(df.shape[1]),
        "duplicatesRemoved": int(raw_df.duplicated().sum()),
        "missingFilled": int(raw_df.isna().sum().sum()),
        "trainRows": int(len(tr)),
        "testRows": int(len(te)),
        "seed": 42,
        "missingByCol": {k: int(v) for k, v in raw_df.isna().sum()[lambda s: s > 0].items()},
    },
    "models": {
        "simple": {
            "intercept": round(float(m_simple.intercept_), 4),
            "coef": {"Time_On_Site": round(float(m_simple.coef_[0]), 4)},
            "testR2": round(float(metrics_df.loc["Simple", "R2"]), 4),
            "cvMean": round(float(cv_simple.mean()), 4),
            "cvStd": round(float(cv_simple.std()), 4),
            "mae": round(float(metrics_df.loc["Simple", "MAE"]), 4),
            "mse": round(float(metrics_df.loc["Simple", "MSE"]), 4),
            "rmse": round(float(metrics_df.loc["Simple", "RMSE"]), 4),
        },
        "multiple": {
            "intercept": round(float(m_multi.intercept_), 4),
            "coef": {k: round(float(v), 4) for k, v in zip(multi_f, m_multi.coef_)},
            "testR2": round(float(metrics_df.loc["Multiple", "R2"]), 4),
            "cvMean": round(float(cv_multi.mean()), 4),
            "cvStd": round(float(cv_multi.std()), 4),
            "mae": round(float(metrics_df.loc["Multiple", "MAE"]), 4),
            "mse": round(float(metrics_df.loc["Multiple", "MSE"]), 4),
            "rmse": round(float(metrics_df.loc["Multiple", "RMSE"]), 4),
        },
        "gd": {
            "mae": round(float(metrics_df.loc["Gradient descent", "MAE"]), 4),
            "mse": round(float(metrics_df.loc["Gradient descent", "MSE"]), 4),
            "rmse": round(float(metrics_df.loc["Gradient descent", "RMSE"]), 4),
            "testR2": round(float(metrics_df.loc["Gradient descent", "R2"]), 4),
            "maxDiffNormalEq": gd_max_diff,
            "finalCost": round(float(j_1000[-1]), 2),
            "initCost": round(float(j_1000[0]), 2),
            "curves": gd_curves,
        },
        "residuals": {
            "mean": round(float(preds_df["Residual"].mean()), 2),
            "overestimated": int((preds_df["Residual"] < 0).sum()),
            "underestimated": int((preds_df["Residual"] >= 0).sum()),
            "bands": band_means,
            "shapiroP": round(shapiro_p, 4),
            "qqPts": qq_pts,
            "qqSlope": round(qq_slope, 2),
            "qqIntercept": round(qq_intercept, 2),
        },
        "ttest": {
            "premMean": round(float(a.mean()), 2),
            "regMean": round(float(b.mean()), 2),
            "t": round(float(t_stat), 3),
            "p": float(p_val),
            "d": round(cohen_d, 3),
        },
    },
    "numCols": num_cols,
    "llnDrawIdx": lln_draw_idx,
    "rows": rows,
}

out_js = DATA_DIR / "dashboard_data.js"
out_js.write_text("window.DASHBOARD_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n", encoding="utf-8")
print("Wrote", out_js, "bytes:", out_js.stat().st_size, "rows:", len(rows))
