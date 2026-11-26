"""E-Commerce Customer Purchase Behaviour Analysis and Prediction (Tasks 1-8)
Run:  python ecommerce_project.py
Needs: numpy pandas scipy matplotlib seaborn scikit-learn
NOTE: the dataset is SYNTHETIC (simulated in this script), not real customer data."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from pathlib import Path

OUT = Path("plots"); OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")
def head(t): print("\n" + "="*70 + f"\n{t}\n" + "="*70)
def save(name): plt.tight_layout(); plt.savefig(OUT/name, dpi=150); plt.close()

# ------------------------------------------------------------ DATASET
def make_dataset(n=20000, seed=42):
    r = np.random.default_rng(seed)
    gender = r.choice(["Male","Female"], n)
    age = r.integers(18, 66, n)
    member = r.choice(["Regular","Premium"], n, p=[.7,.3])
    device = r.choice(["Mobile","Desktop","Tablet"], n, p=[.6,.3,.1])
    prem = (member=="Premium").astype(float)
    visits = np.clip(np.round(r.normal(10+3*prem, 4)), 1, None)
    time = np.clip(r.normal(18+0.6*(visits-10), 6), 2, None).round(1)
    pages = np.clip(np.round(0.7*time + r.normal(0,3,n)), 1, None)
    cart = np.clip(np.round(0.15*pages + r.normal(1,1.2,n)), 0, None)
    prev = np.clip(r.normal(2500+700*prem, 800), 200, None).round(2)
    disc = r.integers(0, 41, n)
    rating = r.integers(1, 6, n)
    amount = (200 + 35*time + 40*visits + 12*pages + 320*cart + 0.35*prev
              + 150*prem + r.normal(0,350,n))
    df = pd.DataFrame(dict(Customer_ID=[f"C{i:05d}" for i in range(1,n+1)], Gender=gender, Age=age,
        Membership=member, Device=device, Visits_Per_Month=visits, Time_On_Site=time,
        Pages_Viewed=pages, Cart_Items=cart, Previous_Purchase_Amount=prev,
        Discount_Percent=disc, Review_Rating=rating,
        Purchase_Amount=np.clip(amount,100,None).round(2)))
    # deliberately add data-quality problems so that cleaning is meaningful
    n_miss = max(1, int(0.025 * n))
    for col in ["Time_On_Site","Pages_Viewed","Previous_Purchase_Amount","Gender"]:
        df.loc[r.choice(n, n_miss, replace=False), col] = np.nan
    idx = r.choice(n, max(1, int(0.04 * n)), replace=False)
    df.loc[idx, "Device"] = df.loc[idx, "Device"].str.upper()
    n_dup = max(1, int(0.03 * n))
    return pd.concat([df, df.sample(n_dup, random_state=seed)]).sample(frac=1, random_state=seed).reset_index(drop=True)

raw = make_dataset(); raw.to_csv("ecommerce_raw.csv", index=False)

# ------------------------------------------------------------ TASK 1
head("TASK 1: DATA PREPROCESSING")
df = pd.read_csv("ecommerce_raw.csv")
print("Raw shape:", df.shape); print(df.head().to_string())
print("\nMissing values:\n", df.isna().sum()[lambda s: s>0].to_string())
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
df["Device"] = df["Device"].str.title()
num = df.select_dtypes("number").columns.tolist()
cat = [c for c in df.columns if c not in num and c != "Customer_ID"]
train_i, test_i = train_test_split(df.index, test_size=0.2, random_state=42)
for c in num:  # medians come from training rows only (no test-data leakage)
    df[c] = df[c].fillna(df.loc[train_i, c].median())
for c in cat:
    df[c] = df[c].fillna(df.loc[train_i, c].mode()[0])
df.to_csv("ecommerce_clean.csv", index=False)
print("Clean shape:", df.shape, "| missing left:", int(df.isna().sum().sum()))
print("Numerical:", num); print("Categorical:", cat)

# ------------------------------------------------------------ TASK 2
head("TASK 2: DESCRIPTIVE AND INFERENTIAL STATISTICS")
desc = pd.DataFrame({"Mean":df[num].mean(),"Median":df[num].median(),"Mode":df[num].mode().iloc[0],
    "Variance":df[num].var(),"Std_Dev":df[num].std()})
print(desc.round(2).to_string())
y = df.Purchase_Amount
print(f"\nPurchase amount: skewness={y.skew():.3f}, kurtosis={y.kurt():.3f}")
fig, ax = plt.subplots(1,2,figsize=(11,4))
sns.histplot(y, kde=True, ax=ax[0]); ax[0].set_title("Purchase amount distribution")
sns.boxplot(data=df, x="Membership", y="Purchase_Amount", ax=ax[1]); ax[1].set_title("Spending by membership")
save("01_distribution.png")
samples = {"Population": df,
  "Simple random": df.sample(frac=0.2, random_state=1),
  "Systematic": df.iloc[np.random.default_rng(1).integers(0,5)::5],
  "Stratified": df.groupby("Membership").sample(frac=0.2, random_state=1)}
print("\nSampling comparison:")
for k,s in samples.items(): print(f"  {k:14s} n={len(s):5d} mean={s.Purchase_Amount.mean():.2f}")
a = df[df.Membership=="Premium"].Purchase_Amount; b = df[df.Membership=="Regular"].Purchase_Amount
t,p = stats.ttest_ind(a,b,equal_var=False)
d = (a.mean()-b.mean())/np.sqrt(((len(a)-1)*a.var()+(len(b)-1)*b.var())/(len(a)+len(b)-2))
print("\nWelch t-test | H0: Premium and Regular mean spending are equal | H1: they differ")
print(f"  Premium mean={a.mean():.2f}, Regular mean={b.mean():.2f}")
print(f"  t={t:.3f}, p={p:.3g}, Cohen's d={d:.3f} ->", "Reject H0" if p<0.05 else "Fail to reject H0")

# ------------------------------------------------------------ TASK 3
head("TASK 3: PROBABILITY ANALYSIS")
HIGH = round(float(y.quantile(0.75)), -2); T = 20
H = y > HIGH; E = df.Time_On_Site > T
print(f"'High spender' = purchase above {HIGH:.0f}; 'engaged' = more than {T} minutes on site")
print(f"P(High) = {H.mean():.4f}")
print(f"P(High | Engaged) = {H[E].mean():.4f}   P(High | Not engaged) = {H[~E].mean():.4f}")
P = df.Membership=="Premium"
pA, pBA, pBn = P.mean(), H[P].mean(), H[~P].mean()
pB = pBA*pA + pBn*(1-pA); post = pBA*pA/pB
print("\nBayes: probability that a high spender is a Premium member")
print(f"  P(Premium)={pA:.4f}, P(High|Premium)={pBA:.4f}, P(High|Regular)={pBn:.4f}, P(High)={pB:.4f}")
print(f"  P(Premium|High) = {post:.4f}   (direct count check: {P[H].mean():.4f})")
draws = np.random.default_rng(7).choice(H.astype(int), 10000, replace=True)
running = np.cumsum(draws)/np.arange(1,len(draws)+1)
print("\nLaw of large numbers (random draws with replacement):")
for k in [10,100,1000,10000]: print(f"  n={k:5d}  observed={running[k-1]:.4f}  true={H.mean():.4f}")
plt.figure(figsize=(8,4)); plt.plot(running,label="Observed proportion"); plt.axhline(H.mean(),c="red",ls="--",label="True proportion")
plt.xscale("log"); plt.xlabel("Number of draws"); plt.ylabel("P(High spender)"); plt.title("Law of large numbers"); plt.legend(); save("02_lln.png")

# ------------------------------------------------------------ TASK 4
head("TASK 4: CORRELATION ANALYSIS")
pear = df[num].corr("pearson"); spear = df[num].corr("spearman")
ct = pd.DataFrame({"Pearson":pear.Purchase_Amount,"Spearman":spear.Purchase_Amount}).drop("Purchase_Amount")
ct = ct.sort_values("Pearson", key=abs, ascending=False)
print(ct.round(3).to_string()); print("Strongest factors:", ", ".join(ct.index[:3]))
for m,name in [(pear,"03_pearson.png"),(spear,"04_spearman.png")]:
    plt.figure(figsize=(9,7)); sns.heatmap(m,annot=True,fmt=".2f",cmap="coolwarm",vmin=-1,vmax=1)
    plt.title(name[3:-4].title()+" correlation matrix"); save(name)

# ------------------------------------------------------------ TASK 5
head("TASK 5: REGRESSION ANALYSIS")
simple_f = ["Time_On_Site"]
multi_f = ["Time_On_Site","Visits_Per_Month","Pages_Viewed","Cart_Items","Previous_Purchase_Amount"]
tr, te = df.loc[train_i], df.loc[test_i]; y_tr = tr.Purchase_Amount.values; y_te = te.Purchase_Amount.values
models = {}
for name,f in [("Simple",simple_f),("Multiple",multi_f)]:
    m = LinearRegression().fit(tr[f], y_tr); models[name] = (m, m.predict(te[f]))
    cv = cross_val_score(LinearRegression(), tr[f], y_tr, cv=KFold(5,shuffle=True,random_state=1), scoring="r2")
    print(f"{name} regression: intercept={m.intercept_:.3f}")
    for n_,c in zip(f,m.coef_): print(f"    {n_:26s} {c:.4f}")
    print(f"    test R2={m.score(te[f],y_te):.4f} | 5-fold CV R2={cv.mean():.4f} +/- {cv.std():.4f}")
plt.figure(figsize=(7,5)); plt.scatter(te.Time_On_Site,y_te,alpha=.5,label="Actual")
xs = pd.DataFrame({"Time_On_Site":np.linspace(df.Time_On_Site.min(),df.Time_On_Site.max(),100)})
plt.plot(xs,models["Simple"][0].predict(xs),c="red",label="Fitted line")
plt.xlabel("Time on site (min)"); plt.ylabel("Purchase amount"); plt.title("Simple linear regression"); plt.legend(); save("05_simple_regression.png")

# ------------------------------------------------------------ TASK 6
head("TASK 6: LINEAR ALGEBRA WITH NUMPY")
M = tr[multi_f].values.astype(float)
v1, v2 = M[0], M[1]
print("Customer vector 1:", v1); print("Customer vector 2:", v2)
print("Data matrix shape:", M.shape)
print("Matrix addition (first 2 rows + next 2 rows):\n", M[:2]+M[2:4])
print("Dot product v1.v2 =", round(float(np.dot(v1,v2)),2))
print("Matrix multiplication X(3x5) @ X^T(5x3):\n", (M[:3]@M[:3].T).round(1))
X = np.column_stack([np.ones(len(M)), M])
theta = np.linalg.solve(X.T@X, X.T@y_tr)
print("\nNormal equation  theta = (X^T X)^-1 X^T y")
print("  NumPy  :", theta.round(4))
m = models["Multiple"][0]; print("  sklearn:", np.r_[m.intercept_, m.coef_].round(4))
print("  First prediction as dot product:", round(float(X[0]@theta),2))

# ------------------------------------------------------------ TASK 7
head("TASK 7: CALCULUS AND GRADIENT DESCENT")
print("Cost      J(theta) = (1/2m) * sum((X.theta - y)^2)")
print("Gradient  dJ/dtheta = (1/m) * X^T (X.theta - y)")
print("Update    theta = theta - alpha * gradient")
mu, sd = M.mean(0), M.std(0)
Z = np.column_stack([np.ones(len(M)), (M-mu)/sd])
cost = lambda w: np.sum((Z@w-y_tr)**2)/(2*len(y_tr))
def descend(alpha, iters):
    w = np.zeros(Z.shape[1]); hist=[w.copy()]; J=[cost(w)]
    for _ in range(iters):
        w = w - alpha*(Z.T@(Z@w-y_tr))/len(y_tr); hist.append(w.copy()); J.append(cost(w))
    return w, np.array(hist), J
w, hist, J = descend(0.05, 1000)
for k in [0,1,10,50,100,500,1000]: print(f"iter {k:4d}  cost={J[k]:14.2f}  theta={hist[k].round(2)}")
orig = np.r_[w[0]-np.sum(w[1:]*mu/sd), w[1:]/sd]
print("GD parameters (original units):", orig.round(4)); print("Max difference from normal equation:", f"{np.abs(orig-theta).max():.2e}")
fig, ax = plt.subplots(1,3,figsize=(15,4))
for al in [0.01,0.05,0.2]: ax[0].plot(descend(al,200)[2],label=f"alpha={al}")
ax[0].set(title="Cost vs iteration",xlabel="Iteration",ylabel="J"); ax[0].set_yscale("log"); ax[0].legend()
ax[1].plot(J[:200]); ax[1].set(title="Cost (alpha=0.05)",xlabel="Iteration")
for i,n_ in enumerate(["Intercept"]+multi_f): ax[2].plot(hist[:200,i],label=n_)
ax[2].set(title="Parameter changes (standardised)",xlabel="Iteration"); ax[2].legend(fontsize=7); save("06_gradient_descent.png")

# ------------------------------------------------------------ TASK 8
head("TASK 8: MODEL EVALUATION AND RESIDUAL ANALYSIS")
Zte = np.column_stack([np.ones(len(te)), (te[multi_f].values-mu)/sd])
preds = {"Simple":models["Simple"][1], "Multiple":models["Multiple"][1], "Gradient descent":Zte@w}
rows = {}
for n_,pr in preds.items():
    res = y_te-pr; mse = np.mean(res**2)
    rows[n_] = dict(MAE=np.mean(np.abs(res)), MSE=mse, RMSE=np.sqrt(mse), R2=1-np.sum(res**2)/np.sum((y_te-y_te.mean())**2))
metrics = pd.DataFrame(rows).T; print(metrics.round(4).to_string()); metrics.to_csv("model_metrics.csv")
comp = pd.DataFrame({"Customer_ID":te.Customer_ID,"Actual":y_te,"Predicted":preds["Multiple"].round(2)})
comp["Residual"] = (comp.Actual-comp.Predicted).round(2)
comp["Status"] = np.where(comp.Residual<0,"Overestimated","Underestimated")
comp.to_csv("predictions.csv", index=False)
print("\nActual vs predicted (first 10):\n", comp.head(10).to_string(index=False))
print("\nResidual = Actual - Predicted (negative means overestimate)")
print(f"Mean residual = {comp.Residual.mean():.2f}"); print(comp.Status.value_counts().to_string())
bands = pd.qcut(comp.Actual,3,labels=["Low","Medium","High"])
print("\nMean residual by actual spending band:\n", comp.groupby(bands,observed=True).Residual.mean().round(2).to_string())
print(f"Shapiro-Wilk p-value for residuals = {stats.shapiro(comp.Residual).pvalue:.4f}")
fig, ax = plt.subplots(2,2,figsize=(11,8))
ax[0,0].scatter(comp.Actual,comp.Predicted,alpha=.5); lim=[comp.Actual.min(),comp.Actual.max()]
ax[0,0].plot(lim,lim,"r--"); ax[0,0].set(title="Actual vs predicted",xlabel="Actual",ylabel="Predicted")
ax[0,1].scatter(comp.Predicted,comp.Residual,alpha=.5); ax[0,1].axhline(0,c="red",ls="--"); ax[0,1].set(title="Residual plot",xlabel="Predicted",ylabel="Residual")
sns.histplot(comp.Residual,kde=True,ax=ax[1,0]); ax[1,0].set_title("Residual distribution")
stats.probplot(comp.Residual,plot=ax[1,1]); ax[1,1].set_title("Q-Q plot"); save("07_residuals.png")
print("\nDONE. Files: CSVs in this folder, charts in plots/")
