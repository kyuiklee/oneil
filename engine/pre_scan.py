"""일회성 분석 (머지 금지): 10배 종목이 '바닥 ~ 첫 52주 신고가' 사이에 보인 신호
1) 기술: 2025-01-01 이후 최대 상승 구간의 저점 L, 그 뒤 첫 종가 52주 신고가 F 사이 특징을 그룹별 비교 (10배+/3~10배/3배 미만)
2) 예측: 같은 신호를 과거 모든 종목에 미래 정보 없이 적용했을 때 이후 1년 성과 → site/data/pre_ev.json
"""
import os, sys, gzip, json, pickle, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import fund as FD

ROOT = D.ROOT
START = pd.Timestamp("2025-01-01")
t0 = time.time()
hist = pickle.load(gzip.open(os.path.join(ROOT, "cache", "hist.pkl.gz"), "rb"))
meta = D.load_universe().set_index("code")
try:
    raw = pickle.load(gzip.open(os.path.join(ROOT, "cache", "fund.pkl.gz"), "rb"))["raw"]
except Exception as e:
    print("실적 캐시 없음", e); raw = None

clean = {}
for c, h in hist.items():
    h = h[["Open", "High", "Low", "Close", "Volume"]].astype(float).copy()
    h.index = pd.to_datetime(h.index).normalize()
    clean[c] = h[~h.index.duplicated(keep="last")].sort_index()
CL = pd.DataFrame({c: h["Close"] for c, h in clean.items() if len(h) > 260}).sort_index().ffill(limit=5)
roc = lambda n: CL / CL.shift(n) - 1
RSR = ((0.4 * roc(63) + 0.2 * roc(126) + 0.2 * roc(189) + 0.2 * roc(252)).rank(axis=1, pct=True) * 98 + 1)
print(f"준비 {time.time() - t0:.0f}초")

FS = {}
def fser(code):
    if raw is None: return None
    if code not in FS:
        try: FS[code] = FD.series(raw, code)
        except Exception: FS[code] = None
    return FS[code]

def og_at(code, d):
    fs = fser(code)
    if fs is None or not len(fs): return np.nan
    fs = fs[fs.index <= d]
    if not len(fs) or (d - fs.index[-1]).days > 200: return np.nan
    v = fs["og"].iloc[-1]; return float(v) if np.isfinite(v) else np.nan

def rs_at(code, d):
    if code in RSR and d in RSR.index:
        v = RSR.at[d, code]; return float(v) if np.isfinite(v) else np.nan
    return np.nan

def arrays(h):
    o, hi, lo, cl, v = (h[k].to_numpy() for k in ("Open", "High", "Low", "Close", "Volume"))
    lo = np.where(lo > 0, lo, cl)
    s = pd.Series(cl)
    return dict(o=o, hi=hi, lo=lo, cl=cl, v=v, idx=h.index,
                ma50=s.rolling(50).mean().to_numpy(), ma200=s.rolling(200, min_periods=150).mean().to_numpy(),
                ph=pd.Series(hi).shift(1).rolling(250, min_periods=120).max().to_numpy(),
                v50=pd.Series(v).shift(1).rolling(50, min_periods=20).mean().to_numpy(),
                chg=np.r_[np.nan, cl[1:] / cl[:-1] - 1] * 100)

# ---------------- 1) 기술 비교 ----------------
rows = []
for code, h in clean.items():
    h = h[h.index >= START - pd.Timedelta(days=420)]
    s = h.index.searchsorted(START)
    if len(h) - s < 120 or s < 250: continue
    A = arrays(h); lo, hi, cl, v = A["lo"], A["hi"], A["cl"], A["v"]
    run_min = np.minimum.accumulate(lo[s:]); ratio = hi[s:] / run_min
    j = int(np.nanargmax(ratio)) + s; L = int(np.argmin(lo[s:j + 1])) + s; mult = float(ratio[j - s])
    F = None
    for t in range(L + 1, j + 1):
        if np.isfinite(A["ph"][t]) and cl[t] >= A["ph"][t]: F = t; break
    if F is None or F - L < 10: continue
    idx = A["idx"]; seg = slice(L + 1, F + 1)
    r = dict(code=code, name=str(meta["name"].get(code, "")), mult=mult, days_LF=F - L)
    r["pre_dd"] = (lo[L] / A["ph"][L] - 1) * 100 if np.isfinite(A["ph"][L]) else np.nan
    r["gain_F"] = (cl[F] / lo[L] - 1) * 100
    for k in (10, 20, 40):
        r[f"ret{k}"] = (cl[min(L + k, F)] / lo[L] - 1) * 100 if L + k <= F else np.nan
    # 200일선·50일선 회복
    above200 = np.where(cl[L + 1:F + 1] > A["ma200"][L + 1:F + 1])[0]
    r["d_ma200"] = int(above200[0] + 1) if len(above200) else np.nan
    r["ma200_gap_L"] = (lo[L] / A["ma200"][L] - 1) * 100 if np.isfinite(A["ma200"][L]) else np.nan
    up50 = np.where(A["ma50"][L + 6:F + 1] > A["ma50"][L + 1:F - 4])[0] if F - L > 6 else []
    r["d_ma50up"] = int(up50[0] + 6) if len(up50) else np.nan
    # 상대강도 변화
    r["rs_L"] = rs_at(code, idx[L])
    r["rs_20"] = rs_at(code, idx[min(L + 20, F)]); r["rs_40"] = rs_at(code, idx[min(L + 40, F)])
    rsF = [rs_at(code, idx[t]) for t in range(L, F + 1)]
    hit80 = [i for i, x in enumerate(rsF) if x >= 80]
    r["d_rs80"] = hit80[0] if hit80 else np.nan
    r["rs_F"] = rsF[-1]
    # 거래량·장대양봉
    v50L = A["v50"][L]
    vs = v[seg]; ch = A["chg"][seg]
    r["vol_LF"] = float(vs.mean() / v50L) if v50L > 0 else np.nan
    up = ch > 0
    r["ud_LF"] = float(vs[up].sum() / max(vs[~up].sum(), 1))
    big = (ch >= 10) & (vs >= 3 * A["v50"][seg])
    r["n_power"] = int(big.sum())
    r["d_power"] = int(np.where(big)[0][0] + 1) if big.any() else np.nan
    r["n_limit"] = int((ch >= 29).sum())
    r["n_gap5"] = int(((A["o"][seg] / cl[L:F] - 1) * 100 >= 5).sum())
    # 바닥 다지기: 저점 후 다시 밀린 깊이 (저점 대비), 저점 재시험
    if F - L > 15:
        k = L + 5 + int(np.argmax(cl[L + 5:F - 4])) if F - 4 > L + 5 else L + 5
        r["pullback"] = (lo[k:F + 1].min() / cl[k] - 1) * 100 if F > k else np.nan
        r["higher_low"] = (lo[L + 10:F + 1].min() / lo[L] - 1) * 100
    # 실적: 저점 시점과 첫 신고가 시점
    r["og_L"] = og_at(code, idx[L]); r["og_F"] = og_at(code, idx[F])
    # 저점 이후 첫 '영업이익 +30%↑' 공시가 신고가 전에 나왔는지
    fs = fser(code)
    if fs is not None and len(fs):
        g = fs[(fs.index > idx[L]) & (fs.index <= idx[F]) & (fs["og"] >= 30)]
        r["earn_before_F"] = bool(len(g))
        g2 = fs[(fs.index > idx[L]) & (fs["og"] >= 30)]
        r["d_earn"] = int((g2.index[0] - idx[L]).days) if len(g2) else np.nan
    sh = meta["shares"].get(code, np.nan)
    r["mcap_L"] = float(lo[L] * sh / 1e8) if np.isfinite(sh) else np.nan
    rows.append(r)

df = pd.DataFrame(rows)
df["grp"] = np.where(df["mult"] >= 10, "10배+", np.where(df["mult"] >= 3, "3~10배", "3배 미만"))
med = lambda s: None if s.dropna().empty else round(float(s.dropna().median()), 1)
pc = lambda m: round(float(m.mean()) * 100, 0)
def summ(g):
    out = {"n": len(g)}
    for c in ["days_LF", "pre_dd", "gain_F", "ret10", "ret20", "ret40", "ma200_gap_L", "d_ma200", "d_ma50up", "rs_L", "rs_20", "rs_40", "d_rs80",
              "rs_F", "vol_LF", "ud_LF", "n_power", "d_power", "n_limit", "n_gap5", "pullback", "higher_low", "og_L", "og_F", "d_earn", "mcap_L"]:
        if c in g: out[c] = med(g[c])
    out["%ret20>=30"] = pc(g["ret20"] >= 30); out["%rs40>=80"] = pc(g["rs_40"] >= 80); out["%power"] = pc(g["n_power"] > 0)
    out["%limit"] = pc(g["n_limit"] > 0); out["%og_L>=30"] = pc(g["og_L"] >= 30); out["%earn_before_F"] = pc(g.get("earn_before_F", pd.Series(dtype=bool)).fillna(False).astype(bool))
    out["%higher_low>=0"] = pc(g["higher_low"] >= 0) if "higher_low" in g else None
    return out
S = {k: summ(df[df["grp"] == k]) for k in ("10배+", "3~10배", "3배 미만")}
print("== 바닥~첫 신고가 사이 특징 (중앙값 / 비율%)")
keys = list(S["10배+"].keys())
print(f"{'항목':<16}" + "".join(f"{k:>12}" for k in S))
for k in keys:
    print(f"{k:<16}" + "".join(f"{str(S[g].get(k)):>12}" for g in S))

# ---------------- 2) 예측: 미래 정보 없이 '바닥 후 초기 신호' 이벤트 ----------------
ev = []
W = 250
for code, h in clean.items():
    if len(h) < W + 60: continue
    A = arrays(h); lo, hi, cl, v, idx = A["lo"], A["hi"], A["cl"], A["v"], A["idx"]
    n = len(cl)
    from numpy.lib.stride_tricks import sliding_window_view
    Lw = np.full(n, -1); Lw[W - 1:] = np.arange(n - W + 1) + sliding_window_view(lo, W).argmin(axis=1)
    amt20 = pd.Series(cl * v / 1e8).rolling(20).mean().to_numpy()
    last = {}
    for t in range(W + 5, n):
        L = Lw[t]; ago = t - L
        if ago < 5 or ago > 120 or cl[t] < 1000 or not (amt20[t] >= 3): continue
        if not np.isfinite(A["ph"][L]): continue
        dd = (lo[L] / A["ph"][L] - 1) * 100
        if dd > -40: continue
        types = []
        if cl[t] > A["ma200"][t] and cl[t - 1] <= A["ma200"][t - 1] and np.all(cl[L + 1:t] <= A["ma200"][L + 1:t]): types.append("ma200")
        if A["chg"][t] >= 10 and A["v50"][t] > 0 and v[t] >= 3 * A["v50"][t] and not np.any((A["chg"][L + 1:t] >= 10) & (v[L + 1:t] >= 3 * A["v50"][L + 1:t])): types.append("power")
        if ago >= 20 and (cl[t] / lo[L] - 1) * 100 >= 30 and (cl[t - 1] / lo[L] - 1) * 100 < 30 and t - L <= 40: types.append("ret30")
        if not types: continue
        rs = rs_at(code, idx[t])
        for ty in types:
            if t - last.get(ty, -999) < 120: continue
            last[ty] = t
            fc, fh = cl[t + 1:t + 251], hi[t + 1:t + 251]
            e = dict(type=ty, date=f"{idx[t]:%Y-%m-%d}", code=code, name=str(meta["name"].get(code, "")), ago=int(ago), dd=round(dd),
                     gain=round((cl[t] / lo[L] - 1) * 100), rs=None if not np.isfinite(rs) else int(rs),
                     og=None if not np.isfinite(og_at(code, idx[t])) else round(og_at(code, idx[t])),
                     mcl=None, nf=int(len(fc)))
            sh = meta["shares"].get(code, np.nan)
            if np.isfinite(sh): e["mcl"] = round(lo[L] * sh / 1e8)
            if len(fc):
                e["mx"] = round(float(fh.max() / cl[t]), 2); e["dn"] = round(float((lo[t + 1:t + 251].min() / cl[t] - 1) * 100))
                for k in (40, 120, 250): e[f"r{k}"] = round(float((fc[min(k, len(fc)) - 1] / cl[t] - 1) * 100), 1)
            ev.append(e)
E = pd.DataFrame(ev)
json.dump({"summary": S, "ev": ev}, open(os.path.join(ROOT, "site", "data", "pre_ev.json"), "w", encoding="utf-8"), ensure_ascii=False)
full = E[E["nf"] >= 250]
print(f"== 초기 신호 이벤트 {len(E)}건 (1년 추적 {len(full)})")
for ty in ("ma200", "power", "ret30"):
    for nm, g in (("전체", full[full.type == ty]), ("+RS80", full[(full.type == ty) & (full.rs >= 80)]),
                  ("+영업익30", full[(full.type == ty) & (full.og >= 30)]), ("+RS80+영업익30", full[(full.type == ty) & (full.rs >= 80) & (full.og >= 30)])):
        if not len(g): print(f"  {ty} {nm}: 0"); continue
        print(f"  {ty:6} {nm:<12} {len(g):5d}건 | 1년 승률 {(g.r250 > 0).mean()*100:3.0f}% 중앙 {g.r250.median():+5.0f}% 평균 {g.r250.mean():+5.0f}% | "
              f"40일 승률 {(g.r40 > 0).mean()*100:3.0f}% 중앙 {g.r40.median():+5.1f}% | 2배 {(g.mx >= 2).mean()*100:3.0f}% 5배 {(g.mx >= 5).mean()*100:3.0f}% | 최대하락 {g.dn.median():.0f}%")
print(f"완료 {time.time() - t0:.0f}초")
