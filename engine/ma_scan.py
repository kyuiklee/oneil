"""일회성 분석 (머지 금지): 이동평균선(5·20·60·120)과 바닥 거래량 — 10배+/3~10배/3배 미만 비교 + 신호로 썼을 때 1년 성과"""
import os, sys, gzip, json, pickle, time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import fund as FD

ROOT = D.ROOT; START = pd.Timestamp("2025-01-01"); t0 = time.time()
hist = pickle.load(gzip.open(os.path.join(ROOT, "cache", "hist.pkl.gz"), "rb"))
meta = D.load_universe().set_index("code")
try:
    raw = pickle.load(gzip.open(os.path.join(ROOT, "cache", "fund.pkl.gz"), "rb"))["raw"]
except Exception:
    raw = None
def og_at(code, d):
    if raw is None: return np.nan
    try: fs = FD.series(raw, code)
    except Exception: return np.nan
    if fs is None or not len(fs): return np.nan
    fs = fs[fs.index <= d]
    if not len(fs) or (d - fs.index[-1]).days > 200: return np.nan
    v = fs["og"].iloc[-1]; return float(v) if np.isfinite(v) else np.nan

def prep(h):
    h = h[["Open", "High", "Low", "Close", "Volume"]].astype(float).copy()
    h.index = pd.to_datetime(h.index).normalize(); h = h[~h.index.duplicated(keep="last")].sort_index()
    c = h["Close"]; A = {k: h[k].to_numpy() for k in ("Open", "High", "Low", "Close", "Volume")}
    A["Low"] = np.where(A["Low"] > 0, A["Low"], A["Close"]); A["idx"] = h.index
    for n in (5, 20, 60, 120): A[f"m{n}"] = c.rolling(n).mean().to_numpy()
    M = np.vstack([A["m5"], A["m20"], A["m60"], A["m120"]])
    A["spread"] = (M.max(0) - M.min(0)) / A["Close"] * 100                       # 4개 이평 간격 (종가 대비 %)
    A["spread3"] = (np.vstack([A["m20"], A["m60"], A["m120"]]).max(0) - np.vstack([A["m20"], A["m60"], A["m120"]]).min(0)) / A["Close"] * 100
    A["bull"] = (A["m5"] > A["m20"]) & (A["m20"] > A["m60"]) & (A["m60"] > A["m120"])   # 정배열
    A["bear"] = (A["m5"] < A["m20"]) & (A["m20"] < A["m60"]) & (A["m60"] < A["m120"])   # 역배열
    A["v50"] = pd.Series(A["Volume"]).shift(1).rolling(50, min_periods=20).mean().to_numpy()
    A["ph"] = pd.Series(A["High"]).shift(1).rolling(250, min_periods=120).max().to_numpy()
    A["chg"] = np.r_[np.nan, A["Close"][1:] / A["Close"][:-1] - 1] * 100
    return A

P = {c: prep(h) for c, h in hist.items() if len(h) > 300}
print(f"준비 {time.time() - t0:.0f}초")
rows = []
for code, A in P.items():
    idx = A["idx"]; s = idx.searchsorted(START)
    if len(idx) - s < 120 or s < 250: continue
    lo, hi, cl, v = A["Low"], A["High"], A["Close"], A["Volume"]
    run_min = np.minimum.accumulate(lo[s:]); ratio = hi[s:] / run_min
    j = int(np.nanargmax(ratio)) + s; L = int(np.argmin(lo[s:j + 1])) + s; mult = float(ratio[j - s])
    F = next((t for t in range(L + 1, j + 1) if np.isfinite(A["ph"][t]) and cl[t] >= A["ph"][t]), None)
    if F is None or F - L < 10 or L < 130: continue
    r = dict(code=code, mult=mult, LF=F - L)
    # --- 이평 배열·밀집 ---
    r["bear_L"] = bool(A["bear"][L]); r["spread_L"] = A["spread"][L]
    r["bear_days_pre"] = int(A["bear"][L - 60:L].sum())                      # 바닥 전 60일 중 역배열 일수
    b = np.where(A["bull"][L + 1:F + 1])[0]; r["d_bull"] = int(b[0] + 1) if len(b) else np.nan
    r["bull_before_F"] = bool(len(b))
    r["bull_F"] = bool(A["bull"][F]); r["spread_F"] = A["spread"][F]
    r["min_spread_LF"] = float(np.nanmin(A["spread3"][L:F + 1]))             # 바닥~신고가 사이 20·60·120 최소 간격
    r["min_spread_preF"] = float(np.nanmin(A["spread3"][max(L, F - 20):F + 1]))  # 신고가 직전 20일
    gc = np.where((A["m20"][L + 1:F + 1] > A["m60"][L + 1:F + 1]) & (A["m20"][L:F] <= A["m60"][L:F]))[0]
    r["d_gc2060"] = int(gc[0] + 1) if len(gc) else np.nan
    gc2 = np.where((A["m60"][L + 1:F + 1] > A["m120"][L + 1:F + 1]) & (A["m60"][L:F] <= A["m120"][L:F]))[0]
    r["d_gc60120"] = int(gc2[0] + 1) if len(gc2) else np.nan
    r["below120_L"] = (cl[L] / A["m120"][L] - 1) * 100                        # 바닥 때 120일선 이격
    r["m120_slope_F"] = (A["m120"][F] / A["m120"][F - 20] - 1) * 100          # 신고가 때 120일선 기울기
    # --- 거래량 ---
    v50 = A["v50"][L]
    if v50 > 0:
        r["v_Lday"] = v[L] / v50                                              # 바닥 당일
        r["v_spike5"] = v[L - 5:L + 6].max() / v50                            # 바닥 ±5일 최대
        r["v_pre20"] = v[L - 20:L].mean() / v50                               # 바닥 직전 20일 (투매)
        r["v_post20"] = v[L + 1:L + 21].mean() / v50                          # 바닥 후 20일
        r["v_dry"] = pd.Series(v[L:F + 1]).rolling(10).mean().min() / v50     # 바닥~신고가 사이 가장 조용한 10일
        r["v_F"] = v[F] / A["v50"][F] if A["v50"][F] > 0 else np.nan          # 첫 신고가 당일
    sh = meta["shares"].get(code, np.nan)
    r["turn_L"] = v[L] / sh * 100 if np.isfinite(sh) and sh > 0 else np.nan   # 바닥 당일 회전율 %
    rows.append(r)
df = pd.DataFrame(rows)
df["grp"] = np.where(df["mult"] >= 10, "10배+", np.where(df["mult"] >= 3, "3~10배", "3배 미만"))
med = lambda x: None if x.dropna().empty else round(float(x.dropna().median()), 2)
pc = lambda m: round(float(m.mean()) * 100)
def summ(g):
    o = {"n": len(g), "LF": med(g.LF)}
    for c in ["spread_L", "bear_days_pre", "below120_L", "d_bull", "d_gc2060", "d_gc60120", "min_spread_LF", "min_spread_preF", "spread_F", "m120_slope_F",
              "v_Lday", "v_spike5", "v_pre20", "v_post20", "v_dry", "v_F", "turn_L"]:
        o[c] = med(g[c])
    o["%역배열@바닥"] = pc(g.bear_L); o["%정배열 신고가전"] = pc(g.bull_before_F); o["%정배열@신고가"] = pc(g.bull_F)
    o["%밀집<5%(바닥~신고가)"] = pc(g.min_spread_LF < 5); o["%밀집<3%"] = pc(g.min_spread_LF < 3); o["%신고가직전밀집<5%"] = pc(g.min_spread_preF < 5)
    o["%바닥거래량3배↑"] = pc(g.v_spike5 >= 3); o["%바닥거래량5배↑"] = pc(g.v_spike5 >= 5); o["%조용한구간<0.7"] = pc(g.v_dry < 0.7)
    o["%신고가거래량2배↑"] = pc(g.v_F >= 2)
    return o
S = {k: summ(df[df.grp == k]) for k in ("10배+", "3~10배", "3배 미만")}
print("== 이평선·거래량 (중앙값 / 비율%)")
for k in S["10배+"]:
    print(f"{k:<22}" + "".join(f"{str(S[g][k]):>10}" for g in S))

# ---------- 신호로 썼을 때 (미래 정보 없이) ----------
W = 250; ev = []
for code, A in P.items():
    n = len(A["Close"]); lo, cl, v, idx = A["Low"], A["Close"], A["Volume"], A["idx"]
    if n < W + 60: continue
    Lw = np.full(n, -1); Lw[W - 1:] = np.arange(n - W + 1) + sliding_window_view(lo, W).argmin(axis=1)
    amt20 = pd.Series(cl * v / 1e8).rolling(20).mean().to_numpy(); last = {}
    for t in range(W + 5, n):
        L = Lw[t]; ago = t - L
        if ago < 10 or ago > 150 or cl[t] < 1000 or not (amt20[t] >= 3) or not np.isfinite(A["ph"][L]): continue
        dd = (lo[L] / A["ph"][L] - 1) * 100
        if dd > -40: continue
        ty = []
        if A["bull"][t] and not A["bull"][L:t].any(): ty.append("정배열 첫 전환")
        if np.nanmin(A["spread3"][t - 10:t]) < 5 and cl[t] > max(A["m20"][t], A["m60"][t], A["m120"][t]) and cl[t - 1] <= max(A["m20"][t - 1], A["m60"][t - 1], A["m120"][t - 1]) \
                and A["v50"][t] > 0 and v[t] >= 1.5 * A["v50"][t]: ty.append("이평 밀집 돌파")
        if A["m20"][t] > A["m60"][t] and A["m20"][t - 1] <= A["m60"][t - 1] and not (A["m20"][L:t] > A["m60"][L:t]).any(): ty.append("골든크로스 20·60")
        for k in ty:
            if t - last.get(k, -999) < 120: continue
            last[k] = t
            fc, fh = cl[t + 1:t + 251], A["High"][t + 1:t + 251]
            e = dict(type=k, date=f"{idx[t]:%Y-%m-%d}", code=code, nf=len(fc), og=og_at(code, idx[t]))
            if len(fc):
                e["mx"] = fh.max() / cl[t]; e["r40"] = (fc[min(40, len(fc)) - 1] / cl[t] - 1) * 100; e["r250"] = (fc[-1] / cl[t] - 1) * 100
            ev.append(e)
E = pd.DataFrame(ev); full = E[E.nf >= 250]
print(f"== 이평 신호 1년 성과 (깊은 하락 -40%↓ 뒤 바닥 후 10~150일, 1년 추적 {len(full)}건)")
for k in ("정배열 첫 전환", "이평 밀집 돌파", "골든크로스 20·60"):
    for nm, g in (("전체", full[full.type == k]), ("+영업익30", full[(full.type == k) & (full.og >= 30)])):
        if not len(g): continue
        print(f"  {k:<10} {nm:<7} {len(g):5d}건 | 1년 승률 {(g.r250 > 0).mean()*100:3.0f}% 중앙 {g.r250.median():+5.0f}% 평균 {g.r250.mean():+5.0f}% | 40일 승률 {(g.r40 > 0).mean()*100:3.0f}% 중앙 {g.r40.median():+5.1f}% | 2배 {(g.mx >= 2).mean()*100:3.0f}% 5배 {(g.mx >= 5).mean()*100:3.0f}%")
print(f"완료 {time.time() - t0:.0f}초")
