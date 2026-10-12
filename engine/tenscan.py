"""텐배거 검색기: 2025~26년 10배 종목의 공통점으로 '초기 단계' 종목을 찾고, 같은 신호의 과거 성과를 함께 저장

신호(이벤트) = 최근 250거래일 최저가(L) 이후 '처음으로' 종가 52주 신고가를 낸 날
  - 저점 후 20거래일 이상 지남, 저점 전 52주 고점 대비 -25% 이상 빠졌던 종목 (깊은 하락 뒤 회복)
  - 같은 종목은 120거래일에 한 번만
앱에서 거르는 조건 (10배 종목 공통점, 첫 신고가 시점 값)
  - 저점 전 하락폭   10배 종목 중앙값 -55% (나머지 -45%)
  - RS             10배 종목 중앙값 92, 91%가 80 이상 (나머지 30%)
  - 분기 영업이익    10배 종목 중앙값 +100% (나머지 +23%)
  - 저점 시가총액    10배 종목 중앙값 1,428억 (현재 주식 수 × 저점 가격으로 근사)
저장: site/data/ten.json
  recent = 최근 30거래일 안의 신호 (오늘 볼 후보), ev = 전체 신호 + 이후 250거래일 성과 (앱 백테스트용)
"""
import os, json, time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

W = 250            # 52주 = 250거래일
MIN_AGO = 20       # 저점 후 최소 거래일
MIN_DD = -25.0     # 저장할 최소 하락폭 (앱에서 더 깊게 거름)
GAP = 120          # 같은 종목 신호 최소 간격
FWD = 250          # 성과 추적 기간 (약 1년)
RECENT = 30        # '최근 신호' 범위 (거래일)
MIN_PRICE, MIN_AMT = 1000, 3     # 원, 20일 평균 거래대금(억)


def _r(x, n=1):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else round(x, n)


def _rs_panel(hist):
    closes = {}
    for c, h in hist.items():
        if len(h) > 260:
            s = h["Close"].astype(float).copy()
            s.index = pd.to_datetime(s.index).normalize()
            closes[c] = s[~s.index.duplicated(keep="last")]
    CL = pd.DataFrame(closes).sort_index().ffill(limit=5)
    roc = lambda n: CL / CL.shift(n) - 1
    raw = 0.4 * roc(63) + 0.2 * roc(126) + 0.2 * roc(189) + 0.2 * roc(252)
    return (raw.rank(axis=1, pct=True) * 98 + 1).round()


def _fund_at(fs, d):
    """d 시점에 공시돼 있던 최신 분기 실적 (200일 넘으면 없음)"""
    if fs is None or not len(fs):
        return {}
    i = fs.index.searchsorted(d, side="right") - 1
    if i < 0 or (d - fs.index[i]).days > 200:
        return {}
    row = fs.iloc[i]
    return {k: row.get(k, np.nan) for k in ("og", "ogp", "opm_d", "sg", "cg")}


def events_for(code, h, rs_col, fs, shares):
    h = h[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    h = h[~h.index.duplicated(keep="last")].sort_index()
    n = len(h)
    if n < W + MIN_AGO + 5:
        return []
    lo, hi, cl, vol = (h[c].to_numpy() for c in ("Low", "High", "Close", "Volume"))
    lo = np.where(lo > 0, lo, cl)
    idx = h.index
    ph = pd.Series(hi).shift(1).rolling(W, min_periods=120).max().to_numpy()   # 전날까지 52주 고가
    newhi = np.nan_to_num(cl >= ph, nan=False) & np.isfinite(ph)
    cs = np.cumsum(newhi)
    amt20 = pd.Series(cl * vol / 1e8).rolling(20).mean().to_numpy()
    Lwin = np.full(n, -1)
    Lwin[W - 1:] = np.arange(n - W + 1) + sliding_window_view(lo, W).argmin(axis=1)   # 최근 250일 최저가 위치
    out, last_t = [], -10 ** 9
    for t in np.where(newhi)[0]:
        L = Lwin[t]
        if L < 0 or t - L < MIN_AGO or t - last_t < GAP:
            continue
        if cs[t - 1] - cs[L] > 0:                     # 저점 후 이미 신고가가 있었음 → 첫 신고가 아님
            continue
        if not np.isfinite(ph[L]) or cl[t] < MIN_PRICE or not (amt20[t] >= MIN_AMT):
            continue
        pre_dd = (lo[L] / ph[L] - 1) * 100
        if pre_dd > MIN_DD:
            continue
        last_t = t
        d = idx[t]
        rs = rs_col.get(d, np.nan) if rs_col is not None else np.nan
        f = _fund_at(fs, d)
        e = {"date": f"{d:%Y-%m-%d}", "code": code, "close": int(round(cl[t])),
             "ago": int(t - L), "gain": _r((cl[t] / lo[L] - 1) * 100, 0), "dd": _r(pre_dd, 0),
             "rs": None if not np.isfinite(rs) else int(rs),
             "og": _r(f.get("og"), 0), "ogp": _r(f.get("ogp"), 0), "opm_d": _r(f.get("opm_d"), 1), "sg": _r(f.get("sg"), 0),
             "mcl": _r(lo[L] * shares / 1e8, 0) if np.isfinite(shares) else None,      # 저점 시총(억)
             "mc": _r(cl[t] * shares / 1e8, 0) if np.isfinite(shares) else None,       # 신호일 시총(억)
             "amt": _r(amt20[t], 0)}
        # 이후 성과: 신호일 종가 매수 기준
        fh, fl, fc = hi[t + 1:t + 1 + FWD], lo[t + 1:t + 1 + FWD], cl[t + 1:t + 1 + FWD]
        e["nf"] = int(len(fc))                                                    # 추적된 거래일 수 (250이면 1년 완료)
        if len(fc):
            e["mx"] = _r(fh.max() / cl[t], 2)                                     # 1년 안 최고가 배수
            e["mxd"] = int(fh.argmax() + 1)                                       # 최고가까지 거래일
            e["dn"] = _r((fl.min() / cl[t] - 1) * 100, 0)                         # 1년 안 최대 하락
            for k in (60, 120, 250):
                e[f"r{k}"] = _r((fc[min(k, len(fc)) - 1] / cl[t] - 1) * 100, 1)
            e["now"] = _r((fc[-1] / cl[t] - 1) * 100, 1)                          # 마지막 날까지 수익률
        out.append(e)
    return out


def run(hist, meta, fund, out_dir):
    t0 = time.time()
    try:
        RS = _rs_panel(hist)
    except Exception as e:
        print(f"텐배거: RS 계산 실패 {e}")
        RS = None
    ev = []
    for code, h in hist.items():
        try:
            h2 = h.copy()
            h2.index = pd.to_datetime(h2.index).normalize()
            sh = meta["shares"].get(code, np.nan) if "shares" in meta else np.nan
            try:
                sh = float(sh)
            except (TypeError, ValueError):
                sh = np.nan
            rs_col = RS[code] if RS is not None and code in RS else None
            ev += events_for(code, h2, rs_col, fund.get(code) if fund else None, sh)
        except Exception as e:
            print(f"텐배거 {code}: {type(e).__name__} {e}")
    for e in ev:
        e["name"] = str(meta["name"].get(e["code"], ""))
        e["mkt"] = "Q" if meta["market"].get(e["code"]) == "KOSDAQ" else "K"
    ev.sort(key=lambda e: (e["date"], e["code"]))
    days = sorted({d for h in list(hist.values())[:300] for d in pd.to_datetime(h.index).normalize()})
    last = days[-1] if days else pd.Timestamp.now()
    cut = days[-RECENT] if len(days) >= RECENT else days[0]
    recent = [e for e in ev if pd.Timestamp(e["date"]) >= cut]
    # 최근 신호 종목의 현재 상태
    for e in recent:
        h = hist[e["code"]]
        c = float(h["Close"].iloc[-1])
        e["cur"] = int(round(c))
        e["since"] = _r((c / e["close"] - 1) * 100, 1)
        hi52 = float(h["High"].iloc[-W:].max())
        e["off"] = _r((c / hi52 - 1) * 100, 1)
        if RS is not None and e["code"] in RS:
            v = RS[e["code"]].iloc[-1]
            e["rs_now"] = None if not np.isfinite(v) else int(v)
    res = {"date": f"{last:%Y-%m-%d}", "fwd": FWD, "recent_days": RECENT, "fund": bool(fund),
           "note": "신호=최근 250일 저점 후 첫 종가 52주 신고가. 성과는 신호일 종가 매수 기준, 비용 차감 전. 현재 상장 종목만",
           "recent": recent, "ev": ev}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ten.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, separators=(",", ":"))
    _report(ev)
    print(f"텐배거 검색: 신호 {len(ev)}건, 최근 {RECENT}거래일 {len(recent)}건 ({time.time() - t0:.0f}초)")


def _report(ev):
    """run.log 점검용: 조건을 하나씩 더할 때 1년 성과"""
    df = pd.DataFrame(ev)
    if df.empty or "mx" not in df:
        return
    full = df[df["nf"] >= FWD]
    steps = [("첫 신고가 전부(하락 -25%↑)", lambda d: d),
             ("+ 하락 -40%↑", lambda d: d[d["dd"] <= -40]),
             ("+ RS 90↑", lambda d: d[(d["dd"] <= -40) & (d["rs"] >= 90)]),
             ("+ 영업익 +30%↑", lambda d: d[(d["dd"] <= -40) & (d["rs"] >= 90) & (d["og"] >= 30)]),
             ("+ 저점시총 5천억↓", lambda d: d[(d["dd"] <= -40) & (d["rs"] >= 90) & (d["og"] >= 30) & (d["mcl"] <= 5000)]),
             ("RS90+영업익30 (하락 무관)", lambda d: d[(d["rs"] >= 90) & (d["og"] >= 30)])]
    print(f"== 텐배거 신호 1년 성과 (1년 추적 완료 {len(full)}건)")
    for nm, fn in steps:
        g = fn(full)
        if not len(g):
            print(f"  {nm}: 0건"); continue
        print(f"  {nm}: {len(g)}건 | 1년 승률 {(g['r250'] > 0).mean() * 100:.0f}% 중앙값 {g['r250'].median():+.0f}% "
              f"평균 {g['r250'].mean():+.0f}% | 120일 중앙값 {g['r120'].median():+.0f}% | "
              f"최고 2배↑ {(g['mx'] >= 2).mean() * 100:.0f}% 3배↑ {(g['mx'] >= 3).mean() * 100:.0f}% "
              f"5배↑ {(g['mx'] >= 5).mean() * 100:.0f}% | 최대하락 중앙값 {g['dn'].median():.0f}%")
