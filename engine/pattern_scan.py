"""일회성 분석: 2025년 이후 10배 종목의 '저점 전후' 차트·수급·실적 특징을 나머지 종목과 비교
- 모든 종목에 같은 방식 적용: 2025-01-01 이후 '그날 고가 ÷ 그 전까지 최저가'가 최대인 구간의 저점(L)과 고점(P)
- 그룹: 10배 이상 / 3~10배 / 3배 미만
- 결과: site/data/patterns.json (+ run.log 요약)
"""
import os, sys, gzip, json, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fund as FD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START = pd.Timestamp("2025-01-01")

hist = pickle.load(gzip.open(os.path.join(ROOT, "cache", "hist.pkl.gz"), "rb"))
meta = pd.read_csv(os.path.join(ROOT, "engine", "tickers.csv"), dtype={"code": str}).set_index("code")
try:
    raw = pickle.load(gzip.open(os.path.join(ROOT, "cache", "fund.pkl.gz"), "rb"))["raw"]
except Exception as e:
    print("실적 캐시 없음:", e); raw = None

# ---------- RS(상대강도) 순위용 종가 패널 ----------
closes = {}
for c, h in hist.items():
    h = h[~h.index.duplicated(keep="last")].sort_index()
    if len(h) > 260:
        closes[c] = h["Close"].astype(float)
CL = pd.DataFrame(closes).sort_index().ffill(limit=5)
roc = lambda n: CL / CL.shift(n) - 1
RSR = (0.4 * roc(63) + 0.2 * roc(126) + 0.2 * roc(189) + 0.2 * roc(252)).rank(axis=1, pct=True) * 98 + 1


def troughs(lo, a, b, w=10):
    """구간 [a,b) 안의 지역 최저점 인덱스 (앞뒤 w일 중 가장 낮은 날)"""
    out = []
    for i in range(max(a, w), min(b, len(lo) - w)):
        if lo[i] == lo[i - w:i + w + 1].min():
            out.append(i)
    return out


def double_bottom(lo, hi, vol, L):
    """저점 L 앞 120일 또는 뒤 60일 안에, 저점 가격 ±7% 이내의 다른 바닥(앞뒤 10일 최저)이 20일 이상 떨어져 있고
    두 바닥 사이 반등이 15% 이상이면 쌍바닥. 두 번째 바닥 거래량이 첫 번째보다 적은지도 기록"""
    p = lo[L]
    for t in troughs(lo, max(0, L - 120), min(len(lo), L + 61)):
        if abs(t - L) < 20 or abs(lo[t] / p - 1) > 0.07:
            continue
        a, b = sorted((t, L))
        if hi[a:b + 1].max() / max(lo[a], lo[b]) - 1 < 0.15:
            continue
        v = lambda i: vol[max(0, i - 2):i + 3].mean()
        return True, bool(v(b) < v(a)), int(t - L)
    return False, False, None


def cup_handle(lo, hi, cl, vol, L, end):
    """L 이후 250일 안의 첫 '컵앤핸들' 돌파.
    컵: 왼쪽 고점(피벗 P) → 깊이 12~35% → 오른쪽이 P의 95%까지 회복, 컵 길이 25~150일
    핸들: 오른쪽 고점 뒤 5~25일, 컵 위쪽 절반에서 고점 대비 12% 이내 눌림
    돌파: 종가가 P 위, 거래량 50일 평균의 1.3배 이상"""
    v50 = pd.Series(vol).shift(1).rolling(50, min_periods=20).mean().to_numpy()
    for t in range(L + 30, min(end, L + 250, len(cl))):
        if t < 160 or not (v50[t] > 0) or vol[t] < 1.3 * v50[t]:
            continue
        P = hi[t - 150:t].max()
        if cl[t] <= P:
            continue
        for hl in range(5, 26):                                # 핸들 길이
            c = t - hl
            if hi[c] < 0.95 * P:
                continue
            seg = hi[t - 150:c - 24]
            idx = np.where(seg >= 0.95 * P)[0]
            if not len(idx):
                continue
            a = t - 150 + idx[0]
            if not (25 <= c - a <= 150):
                continue
            B = lo[a:c].min(); depth = (P - B) / P
            if not (0.12 <= depth <= 0.35):
                continue
            hmin = lo[c:t].min()
            if hmin >= B + 0.5 * (P - B) and hmin >= 0.88 * hi[c]:
                return True, int(t - L), round(depth * 100), int(c - a)
    return False, None, None, None


rows = []
for code, h in hist.items():
    h = h[~h.index.duplicated(keep="last")].sort_index()
    h = h[h.index >= START - pd.Timedelta(days=400)]
    s = h.index.searchsorted(START)
    if len(h) - s < 120 or s < 60:
        continue
    lo = h["Low"].astype(float).to_numpy(); hi = h["High"].astype(float).to_numpy()
    cl = h["Close"].astype(float).to_numpy(); vol = h["Volume"].astype(float).to_numpy()
    lo = np.where(lo > 0, lo, cl)
    run_min = np.minimum.accumulate(lo[s:])
    ratio = hi[s:] / run_min
    j = int(np.nanargmax(ratio)) + s
    L = int(np.argmin(lo[s:j + 1])) + s
    mult = float(ratio[j - s])
    if L < 60:
        continue
    v50 = vol[L - 55:L - 5].mean()
    # 1) 저점 전후 거래량
    vol_spike = float(vol[L - 5:L + 6].max() / v50) if v50 > 0 else np.nan          # 저점 ±5일 최대 거래량 배수
    vol_after = float(vol[L + 1:L + 41].mean() / v50) if v50 > 0 and L + 41 <= len(vol) else np.nan  # 저점 후 40일 평균 거래량 배수
    e = min(L + 41, len(cl))
    up = cl[L + 1:e] > cl[L:e - 1]
    seg_v = vol[L + 1:e]
    ud_after = float(seg_v[up].sum() / max(seg_v[~up].sum(), 1)) if len(seg_v) else np.nan  # 저점 후 상승일/하락일 거래량
    # 2) 저점 전 하락 폭 (52주 고점 대비)
    pre_dd = float(lo[L] / hi[max(0, L - 250):L + 1].max() - 1) * 100
    # 3) 쌍바닥
    db, db_vol_lower, db_gap = double_bottom(lo, hi, vol, L)
    # 4) 컵앤핸들
    ch, ch_days, ch_depth, ch_len = cup_handle(lo, hi, cl, vol, L, j + 1)
    # 5) 저점 후 첫 52주 신고가 (종가 기준)
    fh = None
    for t in range(L + 1, min(j + 1, len(cl))):
        if cl[t] >= hi[max(0, t - 250):t].max():
            fh = t; break
    fh_days = int(fh - L) if fh else None
    fh_gain = float(cl[fh] / lo[L] - 1) * 100 if fh else None
    rs_fh = None; og_fh = None
    if fh:
        d = h.index[fh]
        if code in RSR and d in RSR.index and np.isfinite(RSR.at[d, code]):
            rs_fh = float(RSR.at[d, code])
        if raw is not None:
            fs = FD.series(raw, code)
            if fs is not None and len(fs):
                fs = fs[fs.index <= d]
                if len(fs) and (d - fs.index[-1]).days <= 200 and np.isfinite(fs["og"].iloc[-1]):
                    og_fh = float(fs["og"].iloc[-1])
    # 6) 저점 시가총액 (현재 주식 수 기준 근사)
    sh = meta["shares"].get(code, np.nan)
    mcap_low = float(lo[L] * sh / 1e8) if np.isfinite(sh) else np.nan
    rows.append(dict(code=code, name=str(meta["name"].get(code, "")), mkt=str(meta["market"].get(code, "")), mult=round(mult, 2),
                     low_date=f"{h.index[L]:%Y-%m-%d}", peak_date=f"{h.index[j]:%Y-%m-%d}",
                     vol_spike=vol_spike, vol_after=vol_after, ud_after=ud_after, pre_dd=pre_dd,
                     db=db, db_vol_lower=db_vol_lower, db_gap=db_gap, ch=ch, ch_days=ch_days, ch_depth=ch_depth, ch_len=ch_len,
                     fh_days=fh_days, fh_gain=fh_gain, rs_fh=rs_fh, og_fh=og_fh, mcap_low=mcap_low))

df = pd.DataFrame(rows)
df["grp"] = np.where(df["mult"] >= 10, "10배+", np.where(df["mult"] >= 3, "3~10배", "3배 미만"))


def summ(g):
    f = lambda s: None if s.dropna().empty else round(float(s.dropna().median()), 2)
    pc = lambda m: round(float(m.mean()) * 100, 1)
    return {
        "n": int(len(g)),
        "저점±5일 최대거래량(50일평균 배수) 중앙값": f(g["vol_spike"]),
        "저점 거래량 3배↑ 비율%": pc(g["vol_spike"] >= 3),
        "저점후 40일 평균거래량 배수 중앙값": f(g["vol_after"]),
        "저점후 40일 상승일/하락일 거래량 중앙값": f(g["ud_after"]),
        "저점 전 52주고점 대비 하락% 중앙값": f(g["pre_dd"]),
        "쌍바닥 비율%": pc(g["db"]),
        "쌍바닥 중 두번째 바닥 거래량 감소%": round(float(g.loc[g["db"], "db_vol_lower"].mean()) * 100, 1) if g["db"].any() else None,
        "컵앤핸들 돌파 비율%": pc(g["ch"]),
        "컵앤핸들 돌파까지 일수 중앙값": f(g["ch_days"].astype(float)),
        "컵 깊이% 중앙값": f(g["ch_depth"].astype(float)),
        "첫 52주 신고가까지 일수 중앙값": f(g["fh_days"].astype(float)),
        "첫 신고가 시점 저점대비 상승% 중앙값": f(g["fh_gain"].astype(float)),
        "첫 신고가 시점 RS 중앙값": f(g["rs_fh"].astype(float)),
        "첫 신고가 시점 RS 80↑ 비율%": pc(g["rs_fh"].astype(float) >= 80),
        "첫 신고가 시점 분기 영업이익 증가율 중앙값": f(g["og_fh"].astype(float)),
        "첫 신고가 시점 영업이익 +30%↑ 비율%": pc(g["og_fh"].astype(float) >= 30),
        "저점 시가총액(억) 중앙값": f(g["mcap_low"]),
        "저점 시총 3천억 미만 비율%": pc(g["mcap_low"] < 3000),
    }


summary = {k: summ(df[df["grp"] == k]) for k in ("10배+", "3~10배", "3배 미만")}
for k, v in summary.items():
    print(f"== {k}"); [print(f"  {a}: {b}") for a, b in v.items()]
ten = df[df["grp"] == "10배+"].sort_values("mult", ascending=False)
clean = lambda x: None if (isinstance(x, float) and not np.isfinite(x)) else (round(x, 2) if isinstance(x, float) else x)
json.dump({"summary": summary, "ten": [{k: clean(v) for k, v in r.items()} for r in ten.to_dict("records")]},
          open(os.path.join(ROOT, "site", "data", "patterns.json"), "w", encoding="utf-8"), ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(f"패턴 분석 {len(df)}종목 (10배 {len(ten)})")
