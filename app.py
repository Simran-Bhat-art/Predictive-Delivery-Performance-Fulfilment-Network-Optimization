"""Predictive Delivery Performance & Fulfilment Network Optimization (Olist).
Run:  pip install -r requirements.txt  ->  streamlit run app.py
Put the Olist CSVs (olist_*_dataset.csv) in ./data. Without them, a synthetic DEMO dataset is used."""
import io, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, pulp, streamlit as st
import plotly.express as px, plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
try:
    from lightgbm import LGBMClassifier as GB
    KW = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1)
except (ImportError, OSError):
    from sklearn.ensemble import HistGradientBoostingClassifier as GB
    KW = dict(max_iter=200, learning_rate=0.08)

st.set_page_config("Delivery Risk Optimizer", "🚚", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html,body,[class*=css],.stApp{font-family:'Inter',sans-serif}
.stApp{background:radial-gradient(900px 500px at 10% -10%,rgba(99,102,241,.28),transparent),radial-gradient(800px 500px at 100% 0%,rgba(6,182,212,.20),transparent),#0b1020}
#MainMenu,footer,.stDeployButton,[data-testid=stToolbar]{visibility:hidden}
[data-testid=stMetricValue] div{font-size:1.9rem!important;overflow:visible!important;text-overflow:clip!important}header[data-testid=stHeader]{background:transparent}
.block-container{padding-top:1.4rem;max-width:1300px}
section[data-testid=stSidebar]{background:linear-gradient(180deg,#121a3a,#0b1020);border-right:1px solid rgba(129,140,248,.25)}
@keyframes shift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
.hero{background:linear-gradient(120deg,#4f46e5,#0ea5e9,#14b8a6,#8b5cf6);background-size:300% 300%;animation:shift 12s ease infinite;color:#fff;padding:30px 34px;border-radius:22px;margin-bottom:20px;box-shadow:0 14px 40px rgba(79,70,229,.35)}
.hero h1{margin:0;font-size:2.1rem;font-weight:800;color:#fff;letter-spacing:-.5px}.hero p{margin:8px 0 14px;opacity:.95;font-size:1.02rem}
.chip{display:inline-block;background:rgba(255,255,255,.18);border:1px solid rgba(255,255,255,.35);padding:4px 13px;border-radius:999px;font-size:.78rem;font-weight:600;margin-right:8px}
[data-testid=stMetric]{background:linear-gradient(145deg,rgba(99,102,241,.20),rgba(14,165,233,.07));border:1px solid rgba(129,140,248,.35);padding:16px 18px;border-radius:18px;box-shadow:0 8px 24px rgba(0,0,0,.28);transition:transform .2s}
[data-testid=stMetric]:hover{transform:translateY(-3px)}
[data-testid=stMetricLabel]{text-transform:uppercase;letter-spacing:.8px;font-size:.72rem;color:#94a3b8}
[data-testid=stMetricValue]{font-weight:800;background:linear-gradient(90deg,#a5b4fc,#67e8f9);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
button[data-baseweb=tab]{background:rgba(255,255,255,.05);border-radius:999px;padding:8px 20px;margin-right:8px;font-weight:600;color:#cbd5e1}
button[data-baseweb=tab][aria-selected=true]{background:linear-gradient(90deg,#6366f1,#06b6d4);color:#fff}
[data-baseweb=tab-highlight],[data-baseweb=tab-border]{display:none}
[data-testid=stPlotlyChart],[data-testid=stDataFrame]{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-radius:18px;padding:8px;overflow:hidden}
h2,h3{font-weight:800;letter-spacing:-.3px}
.ins{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:18px 0}
.ins div{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-left:4px solid #818cf8;border-radius:14px;padding:14px 16px}
.ins span{font-size:1.5rem;display:block}.ins b{display:block;color:#e2e8f0;margin-top:2px}.ins small{color:#94a3b8}
</style>""", unsafe_allow_html=True)

_pc = st.plotly_chart
def _styled(fig, where=None, **kw):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", size=13), margin=dict(l=20, r=20, t=60, b=20), title_font_size=17,
                      colorway=["#818cf8", "#22d3ee", "#34d399", "#f472b6", "#fbbf24"])
    fig.update_coloraxes(colorscale="Plasma")
    fig.update_geos(fitbounds=False, scope="south america", lataxis_range=[-34, 6], lonaxis_range=[-74, -34], showcountries=True,
                    bgcolor="rgba(0,0,0,0)", landcolor="#1e293b", countrycolor="#475569", showframe=False)
    return (where.plotly_chart if where is not None else _pc)(fig, **kw)
st.plotly_chart = _styled

CEN = {"AC":(-9.0,-70.8),"AL":(-9.6,-36.8),"AM":(-3.5,-65.0),"AP":(1.4,-51.8),"BA":(-12.9,-41.7),"CE":(-5.2,-39.3),"DF":(-15.8,-47.9),
"ES":(-19.6,-40.6),"GO":(-15.9,-49.8),"MA":(-5.4,-45.3),"MG":(-18.5,-44.6),"MS":(-20.5,-54.5),"MT":(-12.9,-55.9),"PA":(-3.8,-52.0),
"PB":(-7.1,-36.8),"PE":(-8.4,-37.9),"PI":(-7.7,-42.7),"PR":(-24.6,-51.6),"RJ":(-22.2,-42.7),"RN":(-5.8,-36.6),"RO":(-10.9,-62.8),
"RR":(2.1,-61.4),"RS":(-29.7,-53.2),"SC":(-27.2,-50.5),"SE":(-10.6,-37.4),"SP":(-22.2,-48.8),"TO":(-10.2,-48.3)}

def hav(a, b, c, d):
    a, b, c, d = map(np.radians, (a, b, c, d))
    h = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))

def sdist(i, j):
    return float(hav(*CEN[i], *CEN[j]))

# ---------------------------------------------------------------- data
def demo(n=20000, seed=7):
    r = np.random.default_rng(seed); sts = list(CEN)
    w = np.array([40 if s == "SP" else 8 if s in ("RJ", "MG", "PR", "RS", "SC") else 1 for s in sts], float)
    cw = np.array([42 if s == "SP" else 13 if s == "RJ" else 11 if s == "MG" else 3 for s in sts], float)
    sel = pd.DataFrame({"seller_id": range(300), "seller_state": r.choice(sts, 300, p=w / w.sum()), "bias": r.normal(0, .6, 300)})
    d = pd.DataFrame({"seller_id": r.integers(0, 300, n), "customer_state": r.choice(sts, n, p=cw / cw.sum())}).merge(sel, on="seller_id")
    n = len(d)
    d["dist"] = [sdist(a, b) for a, b in zip(d.seller_state, d.customer_state)] + r.gamma(2, 15, n)
    d["product_weight_g"] = r.lognormal(6.8, 1, n); d["vol"] = r.lognormal(9, .9, n)
    d["price"] = r.lognormal(4.4, .8, n); d["n_items"] = 1 + r.poisson(.15, n)
    d["freight"] = 8 + d.dist * (0.008 + d.product_weight_g / 4e5)
    d["purchase"] = pd.Timestamp("2017-01-01") + pd.to_timedelta(r.integers(0, 600 * 86400, n), unit="s")
    d["estimated"] = d.purchase + pd.to_timedelta((9 + d.dist / 120 + r.normal(0, 1.5, n)).round().clip(5), unit="D")
    late = r.random(n) < 1 / (1 + np.exp(-(-3.3 + d.dist / 900 + d.bias)))
    d["delivered"] = d.estimated + pd.to_timedelta(np.where(late, r.integers(1, 12, n), -r.integers(1, 12, n)), unit="D")
    d["review_score"] = np.where(late, r.choice([1, 2, 3, 4, 5], n, p=[.35, .2, .2, .15, .1]), r.choice([1, 2, 3, 4, 5], n, p=[.03, .03, .1, .3, .54]))
    d["order_id"] = range(n)
    return d

@st.cache_data(show_spinner="Preparing data...")
def load(folder):
    f = lambda k: os.path.join(folder, f"olist_{k}_dataset.csv")
    if not os.path.exists(f("orders")):
        return prep(demo()), True
    o = pd.read_csv(f("orders"), parse_dates=["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"])
    o = o[o.order_status == "delivered"].dropna(subset=["order_delivered_customer_date"]).rename(columns={
        "order_purchase_timestamp": "purchase", "order_delivered_customer_date": "delivered", "order_estimated_delivery_date": "estimated"})
    it = pd.read_csv(f("order_items")).groupby(["order_id", "seller_id"], as_index=False).agg(
        price=("price", "sum"), freight=("freight_value", "sum"), n_items=("order_item_id", "count"), product_id=("product_id", "first"))
    p = pd.read_csv(f("products")); p["vol"] = p.product_length_cm * p.product_height_cm * p.product_width_cm
    c = pd.read_csv(f("customers"))[["customer_id", "customer_zip_code_prefix", "customer_state"]]
    s = pd.read_csv(f("sellers"))[["seller_id", "seller_zip_code_prefix", "seller_state"]]
    g = pd.read_csv(f("geolocation")).groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean()
    g.columns = ["lat", "lng"]
    d = (it.merge(o[["order_id", "customer_id", "purchase", "delivered", "estimated"]], on="order_id")
         .merge(p[["product_id", "product_weight_g", "vol"]], on="product_id", how="left")
         .merge(c, on="customer_id").merge(s, on="seller_id"))
    d = d.join(g.add_prefix("c_"), on="customer_zip_code_prefix").join(g.add_prefix("s_"), on="seller_zip_code_prefix")
    d["dist"] = hav(d.s_lat, d.s_lng, d.c_lat, d.c_lng)
    d["dist"] = d.dist.fillna(pd.Series([sdist(a, b) if a in CEN and b in CEN else np.nan for a, b in zip(d.seller_state, d.customer_state)], index=d.index))
    rv = pd.read_csv(f("order_reviews")).groupby("order_id", as_index=False).review_score.mean()
    d = d.merge(rv, on="order_id", how="left")
    return prep(d[d.seller_state.isin(CEN) & d.customer_state.isin(CEN)]), False

def prep(d):
    d = d.sort_values("purchase").reset_index(drop=True)
    d["late"] = (d.delivered > d.estimated).astype(int)
    d["promised"] = (d.estimated - d.purchase).dt.days
    d["days"] = (d.delivered - d.purchase).dt.days
    # seller history from earlier orders only (approximation of "known outcomes" to limit leakage)
    d["hist"] = d.groupby("seller_id").late.transform(lambda x: x.shift(1).expanding().mean()).fillna(d.late.mean())
    return d

def feats(d):
    X = d[["dist", "product_weight_g", "vol", "price", "freight", "n_items", "promised", "hist"]].copy()
    X["month"] = d.purchase.dt.month; X["dow"] = d.purchase.dt.dayofweek
    X["ratio"] = d.freight / (d.price + 1); X["same"] = (d.seller_state == d.customer_state).astype(int)
    X["s_st"] = d.seller_state.astype("category").cat.codes; X["c_st"] = d.customer_state.astype("category").cat.codes
    return X.fillna(X.median())

# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner="Training models...")
def train(_d, key):
    X, cut = feats(_d), int(len(_d) * .8)
    ytr, yte = _d.late[:cut], _d.late[cut:]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(X[:cut], ytr)
    gb = GB(**KW).fit(X[:cut], ytr)
    m, P = {}, {}
    for name, mod in (("Logistic Regression", lr), ("Gradient Boosting", gb)):
        p = mod.predict_proba(X[cut:])[:, 1]; P[name] = p
        m[name] = {"ROC-AUC": roc_auc_score(yte, p), "PR-AUC": average_precision_score(yte, p), "Brier": brier_score_loss(yte, p)}
    imp = pd.Series(getattr(gb, "feature_importances_", np.zeros(X.shape[1])), index=X.columns).sort_values()
    return pd.DataFrame(m).T, imp, gb.predict_proba(X)[:, 1], cut, P, gb

# ---------------------------------------------------------------- OR models
def setup(d, lam):
    L = lam * (d.price + d.freight).mean()
    k, b0 = np.polyfit(d.dist, d.freight, 1); sl = np.polyfit(d.dist, d.prob, 1)[0]
    est = lambda dist: (b0 + k * dist, float(np.clip(d.prob.mean() + sl * (dist - d.dist.mean()), .001, .9)))
    pair = d.groupby(["seller_state", "customer_state"]).agg(n=("late", "size"), c=("freight", "mean"), p=("prob", "mean"))
    cp = lambda i, j: (pair.c[(i, j)], pair.p[(i, j)]) if (i, j) in pair.index and pair.n[(i, j)] >= 10 else est(sdist(i, j))
    return L, cp, est

def model1(d, lam, capx):
    L, cp, _ = setup(d, lam)
    S, R = sorted(d.seller_state.unique()), sorted(d.customer_state.unique())
    dem, cur = d.groupby("customer_state").size(), d.groupby("seller_state").size()
    cost = {(i, j): cp(i, j) for i in S for j in R}
    pr = pulp.LpProblem("m1", pulp.LpMinimize); x = pulp.LpVariable.dicts("x", (S, R), 0)
    pr += pulp.lpSum((cost[i, j][0] + cost[i, j][1] * L) * x[i][j] for i in S for j in R)
    for j in R: pr += pulp.lpSum(x[i][j] for i in S) == int(dem[j])
    for i in S: pr += pulp.lpSum(x[i][j] for j in R) <= capx * cur[i]
    pr.solve(pulp.PULP_CBC_CMD(msg=0))
    fl = pd.DataFrame([(i, j, x[i][j].value()) for i in S for j in R if (x[i][j].value() or 0) > .5], columns=["seller_state", "customer_state", "orders"])
    fl["freight"] = [cost[a, b][0] for a, b in zip(fl.seller_state, fl.customer_state)]
    fl["late_p"] = [cost[a, b][1] for a, b in zip(fl.seller_state, fl.customer_state)]
    cur_f, cur_l = d.freight.sum(), d.prob.sum()
    new_f, new_l = (fl.orders * fl.freight).sum(), (fl.orders * fl.late_p).sum()
    return dict(L=L, cur=(cur_f, cur_l, cur_f + cur_l * L), new=(new_f, new_l, new_f + new_l * L), flows=fl)

def model2(d, lam, P, capx, fixed):
    L, _, est = setup(d, lam)
    R = sorted(d.customer_state.unique()); N = len(d)
    K = list(d.groupby("seller_state").size().sort_values(ascending=False).index[:10])
    dem = d.groupby("customer_state").size()
    t = {(j, k): est(sdist(k, j)) for j in R for k in K}
    pr = pulp.LpProblem("m2", pulp.LpMinimize)
    y = pulp.LpVariable.dicts("y", K, cat="Binary"); z = pulp.LpVariable.dicts("z", (R, K), 0, 1)
    pr += pulp.lpSum(fixed * 1000 * y[k] for k in K) + pulp.lpSum(dem[j] * (t[j, k][0] + t[j, k][1] * L) * z[j][k] for j in R for k in K)
    for j in R:
        pr += pulp.lpSum(z[j][k] for k in K) == 1
        for k in K: pr += z[j][k] <= y[k]
    for k in K: pr += pulp.lpSum(dem[j] * z[j][k] for j in R) <= capx * N / 2 * y[k]
    pr += pulp.lpSum(y[k] for k in K) <= P
    pr.solve(pulp.PULP_CBC_CMD(msg=0))
    opened = [k for k in K if y[k].value() > .5]
    asg = {j: max(K, key=lambda k: z[j][k].value()) for j in R}
    fr = sum(dem[j] * t[j, asg[j]][0] for j in R); lt = sum(dem[j] * t[j, asg[j]][1] for j in R)
    cur = d.freight.sum() + d.prob.sum() * L
    return dict(opened=opened, asg=asg, freight=fr, late=lt, total=fr + lt * L + fixed * 1000 * len(opened), cur=cur)

# ---------------------------------------------------------------- UI
st.markdown('<div class="hero"><h1>🚚 Delivery Risk & Fulfilment Network Optimizer</h1>'
            '<p>Analyse delivery performance → predict late orders → optimize seller allocation and warehouse locations</p>'
            '<span class="chip">📦 Olist 2016–2018</span><span class="chip">🤖 Gradient boosting + SHAP-ready</span>'
            '<span class="chip">🧮 LP + MILP optimization</span><span class="chip">🗺️ Brazil network</span></div>', unsafe_allow_html=True)
with st.sidebar:
    st.header("⚙️ Controls")
    folder = st.text_input("Olist data folder", "data")
    lam = st.slider("Late-delivery penalty (% of avg order value)", 5, 100, 25) / 100
    P = st.slider("Max warehouses", 2, 6, 3)
    capx = st.slider("Capacity multiplier", 1.0, 2.0, 1.3, .1)
    fixed = st.slider("Warehouse fixed cost (R$ '000)", 0, 500, 100, 10)

d, is_demo = load(folder)
if is_demo:
    st.warning("Showing **synthetic DEMO data**. Put the Olist CSV files in the data folder to use the real dataset.")
mt, imp, prob, cut, P_te, gb = train(d, (folder, len(d), is_demo))
d["prob"] = prob
reg = d.groupby("customer_state").agg(orders=("late", "size"), late_rate=("late", "mean"), freight=("freight", "mean")).reset_index()
reg["lat"], reg["lon"] = reg.customer_state.map(lambda s: CEN[s][0]), reg.customer_state.map(lambda s: CEN[s][1])

t1, t2, t3, t4, t5 = st.tabs(["📊 Overview", "🤖 Risk Model", "🕸️ Network Optimizer", "🎛️ What-if", "📄 Paper results"])

with t1:
    a, b, c, e = st.columns(4)
    a.metric("Shipments", f"{len(d):,}"); b.metric("Late rate", f"{d.late.mean():.1%}")
    c.metric("Avg delivery time", f"{d.days.mean():.1f} days")
    e.metric("Avg review score", f"{d.review_score.mean():.2f}" if d.review_score.notna().any() else "n/a")
    w = reg.loc[reg.late_rate.idxmax()]; rv = d.groupby("late").review_score.mean() if d.review_score.notna().any() else None
    ins = [("🔥", f"{w.customer_state} has the highest late rate", f"{w.late_rate:.1%} of its shipments arrive late"),
           ("⭐", "Late deliveries hurt reviews", f"{rv[0]:.2f} → {rv[1]:.2f} average score" if rv is not None else "no review data"),
           ("🧭", "Risk grows with distance", f"{d[d.dist > 1500].late.mean():.1%} late beyond 1,500 km vs {d[d.dist <= 300].late.mean():.1%} within 300 km")]
    st.markdown("<div class='ins'>" + "".join(f"<div><span>{i}</span><b>{t}</b><small>{s}</small></div>" for i, t, s in ins) + "</div>", unsafe_allow_html=True)
    l, r = st.columns([3, 2])
    fig = px.scatter_geo(reg, lat="lat", lon="lon", size="orders", color="late_rate", hover_name="customer_state", scope="south america",
                         color_continuous_scale="Turbo", title="Late-delivery rate by customer state")
    fig.update_geos(fitbounds="locations", showcountries=True); st.plotly_chart(fig, where=l, use_container_width=True)
    st.plotly_chart(px.bar(reg.sort_values("late_rate").tail(12), x="late_rate", y="customer_state", orientation="h", color="late_rate",
                          color_continuous_scale="Sunsetdark", title="Highest late rates"), where=r, use_container_width=True)
    if d.review_score.notna().any():
        rs = d.groupby("late").review_score.mean().rename({0: "On time", 1: "Late"}).reset_index()
        st.plotly_chart(px.bar(rs, x="late", y="review_score", color="late", title="Average review score: on-time vs late", text_auto=".2f"), use_container_width=True)

with t2:
    st.subheader("Model comparison (chronological hold-out: last 20% of orders)")
    st.dataframe(mt.style.format("{:.3f}").background_gradient(cmap="Blues"), use_container_width=True)
    l, r = st.columns(2)
    st.plotly_chart(px.bar(imp, orientation="h", title="Feature importance (gradient boosting)", color=imp.values, color_continuous_scale="Viridis"), where=l, use_container_width=True)
    top = d.iloc[cut:].nlargest(25, "prob")[["order_id", "seller_state", "customer_state", "dist", "price", "freight", "promised", "prob", "late"]]
    r.markdown("**Highest-risk orders (test period)**"); r.dataframe(top.style.format({"dist": "{:.0f}", "prob": "{:.1%}", "price": "{:.0f}", "freight": "{:.1f}"}), use_container_width=True, height=420)

with t3:
    m1 = model1(d, lam, capx)
    st.subheader("Model 1: Seller-to-region assignment (LP)")
    (f0, l0, c0), (f1, l1, c1) = m1["cur"], m1["new"]
    a, b, c = st.columns(3)
    a.metric("Total cost (R$)", f"{c1:,.0f}", f"{(c1 - c0) / c0:.1%} vs current", delta_color="inverse")
    b.metric("Expected late orders", f"{l1:,.0f}", f"{(l1 - l0) / l0:.1%} vs current", delta_color="inverse")
    c.metric("Freight (R$)", f"{f1:,.0f}", f"{(f1 - f0) / f0:.1%} vs current", delta_color="inverse")
    cmp = pd.DataFrame({"Current": [f0, l0 * m1["L"]], "Optimized": [f1, l1 * m1["L"]]}, index=["Freight", "Expected late penalty"]).reset_index()
    st.plotly_chart(px.bar(cmp.melt("index"), x="variable", y="value", color="index", title="Cost breakdown (R$)", barmode="stack"), use_container_width=True)
    with st.expander("Optimal flows (top 20)"):
        st.dataframe(m1["flows"].nlargest(20, "orders"), use_container_width=True)
    st.subheader("Model 2: Warehouse location (MILP)")
    m2 = model2(d, lam, P, capx, fixed)
    a, b, c = st.columns(3)
    a.metric("Warehouses opened", ", ".join(m2["opened"])); b.metric("Total cost (R$)", f"{m2['total']:,.0f}", f"{(m2['total'] - m2['cur']) / m2['cur']:.1%} vs current", delta_color="inverse")
    c.metric("Expected late orders", f"{m2['late']:,.0f}", f"{(m2['late'] - d.prob.sum()) / d.prob.sum():.1%} vs current", delta_color="inverse")
    g = go.Figure()
    for j, k in m2["asg"].items():
        g.add_trace(go.Scattergeo(lat=[CEN[k][0], CEN[j][0]], lon=[CEN[k][1], CEN[j][1]], mode="lines", line=dict(width=1, color="#6366f1"), showlegend=False, hoverinfo="skip"))
    g.add_trace(go.Scattergeo(lat=reg.lat, lon=reg.lon, mode="markers", marker=dict(size=7, color="#14b8a6"), text=reg.customer_state, name="Regions"))
    g.add_trace(go.Scattergeo(lat=[CEN[k][0] for k in m2["opened"]], lon=[CEN[k][1] for k in m2["opened"]], mode="markers+text", text=m2["opened"],
                              marker=dict(size=18, color="#f43f5e", symbol="star"), name="Warehouses"))
    g.update_geos(scope="south america", fitbounds="locations", showcountries=True); g.update_layout(title="Optimized warehouse network", height=520)
    st.plotly_chart(g, use_container_width=True)
    st.caption("Assumptions: state-level nodes; penalty = % of average order value; freight and risk for new routes estimated from distance; seller-to-warehouse leg not modelled; each warehouse handles up to (capacity multiplier ÷ 2) of total volume.")

with t4:
    st.subheader("Scenario grid: total cost of the optimized warehouse network (R$ millions)")
    pens, ws = [.10, .25, .50], [2, 3, 4, 5, 6]
    grid = pd.DataFrame([[model2(d, p_, w_, capx, fixed)["total"] / 1e6 for w_ in ws] for p_ in pens], index=[f"{int(p_ * 100)}% penalty" for p_ in pens], columns=[f"{w_} warehouse(s)" for w_ in ws])
    st.plotly_chart(px.imshow(grid, text_auto=".2f", color_continuous_scale="Tealrose", aspect="auto"), use_container_width=True)
    st.caption("Use the sidebar to change the penalty, warehouse limit, capacity and fixed cost; every tab updates live.")

with t5:
    from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_curve, precision_recall_curve
    from sklearn.calibration import calibration_curve
    yte, L = d.late[cut:], setup(d, lam)[0]
    ds = lambda i: d.purchase.iloc[i].strftime("%d %b %Y")
    st.subheader("Numbers and figures for the research paper")
    st.caption(f"Computed with the current sidebar settings (penalty {lam:.0%}, max warehouses {P}, capacity {capx}, fixed cost R$ {fixed}k). Keep the defaults for the paper.")

    st.markdown("**1. Dataset and split**")
    st.dataframe(pd.DataFrame({"Item": ["Shipments (delivered)", "Observed late rate", "Training period", "Test period", "Test late rate", "Mean predicted probability (test)", "Predicted late orders (all shipments)", "Observed late orders", "Penalty per late order L (R$)"],
        "Value": [f"{len(d):,}", f"{d.late.mean():.2%}", f"{ds(0)} to {ds(cut - 1)} ({cut:,})", f"{ds(cut)} to {ds(-1)} ({len(d) - cut:,})", f"{yte.mean():.2%}",
                  f"{P_te['Gradient Boosting'].mean():.2%}", f"{d.prob.sum():,.0f}", f"{d.late.sum():,}", f"{L:,.2f}"]}), use_container_width=True, hide_index=True)

    st.markdown("**2. Prediction metrics (test period)**")
    rows = []
    for n_, p_ in P_te.items():
        ths = np.linspace(.02, .6, 59); t_ = ths[int(np.argmax([f1_score(yte, p_ >= t) for t in ths]))]
        pr_, rc_, f_, _ = precision_recall_fscore_support(yte, p_ >= t_, average="binary", zero_division=0)
        rows.append([n_, *mt.loc[n_], t_, pr_, rc_, f_])
    st.dataframe(pd.DataFrame(rows, columns=["Model", "ROC-AUC", "PR-AUC", "Brier", "Threshold", "Precision", "Recall", "F1"]).style.format({c: "{:.3f}" for c in ["ROC-AUC", "PR-AUC", "Brier", "Threshold", "Precision", "Recall", "F1"]}), use_container_width=True, hide_index=True)
    st.caption(f"Precision, recall and F1 use the threshold that maximizes F1 on the test set (optimistic). Base rate = {yte.mean():.1%}.")

    st.markdown("**3. Network optimization: current vs cost-only vs risk-aware**")
    cf, cl = d.freight.sum(), d.prob.sum(); tot = lambda f_, l_, fx=0: f_ + l_ * L + fx
    m1r, m1c, m2r, m2c = model1(d, lam, capx), model1(d, 0, capx), model2(d, lam, P, capx, fixed), model2(d, 0, P, capx, fixed)
    opt = pd.DataFrame([
        ["Current (observed)", cf, cl, tot(cf, cl), "-"],
        ["Model 1 cost-only", *m1c["new"][:2], tot(*m1c["new"][:2]), "-"],
        ["Model 1 risk-aware", *m1r["new"][:2], tot(*m1r["new"][:2]), "-"],
        ["Model 2 cost-only", m2c["freight"], m2c["late"], tot(m2c["freight"], m2c["late"], fixed * 1000 * len(m2c["opened"])), ", ".join(m2c["opened"])],
        ["Model 2 risk-aware", m2r["freight"], m2r["late"], m2r["total"], ", ".join(m2r["opened"])]],
        columns=["Configuration", "Freight (R$)", "Expected late orders", "Total cost (R$)", "Warehouses"])
    opt["Saving vs current"] = (opt["Total cost (R$)"].iloc[0] - opt["Total cost (R$)"]) / opt["Total cost (R$)"].iloc[0]
    st.dataframe(opt.style.format({"Freight (R$)": "{:,.0f}", "Expected late orders": "{:,.0f}", "Total cost (R$)": "{:,.0f}", "Saving vs current": "{:.1%}"}), use_container_width=True, hide_index=True)
    st.caption("Cost-only = optimization with penalty 0; every total is re-priced at the current penalty. Model 2 totals include warehouse fixed costs.")

    def show(fig, name):
        b = io.BytesIO(); fig.savefig(b, format="png", dpi=300, bbox_inches="tight", facecolor="white")
        st.pyplot(fig); st.download_button(f"⬇️ Download {name}.png", b.getvalue(), f"{name}.png", "image/png"); plt.close(fig)

    st.markdown("**4. Paper-ready figures (white background, 300 dpi)**")
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    for n_, p_ in P_te.items():
        fp, tp, _ = roc_curve(yte, p_); ax[0].plot(fp, tp, label=n_)
        pr_, rc_, _ = precision_recall_curve(yte, p_); ax[1].plot(rc_, pr_, label=n_)
    ax[0].plot([0, 1], [0, 1], "k--", lw=.8); ax[1].axhline(yte.mean(), color="k", ls="--", lw=.8)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="(a) ROC"); ax[1].set(xlabel="Recall", ylabel="Precision", title="(b) Precision-recall")
    ax[0].legend(fontsize=7); fig.tight_layout(); show(fig, "fig_roc_pr")

    fr, mp = calibration_curve(yte, P_te["Gradient Boosting"], n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(3.4, 3)); ax.plot(mp, fr, "o-"); ax.plot([0, .5], [0, .5], "k--", lw=.8)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed late rate", title="Calibration (gradient boosting)"); fig.tight_layout(); show(fig, "fig_calibration")

    try:
        import shap
        Xs = feats(d).iloc[cut:].sample(min(2000, len(d) - cut), random_state=0)
        sv = shap.TreeExplainer(gb).shap_values(Xs); sv = sv[1] if isinstance(sv, list) else sv
        plt.figure(); shap.summary_plot(sv, Xs, show=False, max_display=12); show(plt.gcf(), "fig_shap")
    except Exception as e:
        st.info(f"SHAP figure unavailable ({type(e).__name__}). Run: python3 -m pip install shap")

    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    reg.sort_values("late_rate").tail(10).plot.barh(x="customer_state", y="late_rate", ax=ax[0], legend=False, color="#4f46e5")
    ax[0].set(xlabel="Late rate", ylabel="Customer state", title="(a) Highest late rates")
    if d.review_score.notna().any():
        rs = d.groupby("late").review_score.mean(); ax[1].bar(["On time", "Late"], rs.values, color=["#22d3ee", "#4f46e5"]); ax[1].set(ylabel="Average review score", title="(b) Review score")
    fig.tight_layout(); show(fig, "fig_delivery_performance")

    fig, ax = plt.subplots(figsize=(6, 2.8)); ax.barh(opt["Configuration"], opt["Total cost (R$)"] / 1e6, color="#4f46e5")
    ax.set(xlabel="Total cost (R$ million)", title="Total cost by configuration"); ax.invert_yaxis(); fig.tight_layout(); show(fig, "fig_cost_comparison")
