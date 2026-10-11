"""윌리엄 오닐 CAN SLIM 조건과 점수.

과거 날짜에도 똑같이 적용해 백테스트에 씁니다.
  C  최근 분기 순이익 전년 동기 대비 증가율 (DART, 공시 다음 날부터)
  A  연간 순이익 3년 연평균 증가율, ROE
  N  52주 신고가 부근 + 베이스(횡보 구간) 돌파
  S  돌파일 거래량 (50일 평균 대비)
  L  상대강도 RS 점수 1~99 (IBD 방식: 최근 3·6·9·12개월 수익률 가중, 전 종목 순위)
  I  기관 매집 근사: 최근 50일 상승일 거래량 ÷ 하락일 거래량 (가격·거래량 기반)
  M  시장 방향: 지수와 50·200일선 (앱에서 하락장 쉬기 옵션)
추가 (몇 배 상승주의 공통점):
  영업이익 가속  분기 영업이익 전년 대비 증가율이 직전 분기보다 높아짐 + 영업이익률 변화
  품목 수출     관세청 품목별 수출 3개월 합계의 전년 대비 증가율 (engine/export_map.json 에 적은 종목만)
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

FUND_COLS = ["cg", "cgp", "sg", "og", "ogp", "opm", "opm_d", "ag", "roe"]
EXP_COLS = ["xg", "xgp"]


def _days_since_max(H, win):
    """전날까지 win일 중 최고가가 며칠 전에 나왔는지"""
    a = H.shift(1).to_numpy()
    out = np.full(len(a), np.nan)
    if len(a) > win:
        w = sliding_window_view(a, win)                 # w[i] = a[i:i+win]
        ok = ~np.isnan(w).any(axis=1)
        am = np.argmax(np.where(np.isnan(w), -np.inf, w), axis=1)
        idx = np.arange(win - 1, len(a))
        out[idx[ok]] = (win - 1 - am[ok]) + 1          # 오늘 기준 며칠 전
    return pd.Series(out, index=H.index)


def _attach(f, src, cols, label_cols, max_age):
    """공표일 기준 시계열을 일봉 날짜에 붙임 (각 날짜에 직전 공표 한 줄 전체, max_age일 넘으면 무효)"""
    if src is not None and len(src):
        sd = src.copy()
        sd["_at"] = sd.index
        sd = sd.reindex(f.index, method="ffill")
        age = (f.index.to_series() - pd.to_datetime(sd["_at"])).dt.days
        stale = age.isna() | (age > max_age)
        for c in cols + label_cols:
            f[c] = sd[c].where(~stale) if c in sd else (np.nan if c in cols else None)
    else:
        for c in cols:
            f[c] = np.nan
        for c in label_cols:
            f[c] = None


def features(code, h, P, fund=None, exp=None):
    h = h[["Open", "High", "Low", "Close", "Volume"]].astype(float).copy()
    h.index = pd.to_datetime(h.index).normalize()
    h = h[~h.index.duplicated(keep="last")].sort_index()
    O, H, L, C, V = (h[c] for c in ("Open", "High", "Low", "Close", "Volume"))
    f = pd.DataFrame(index=h.index)
    f["code"] = code
    f["open"], f["high"], f["low"], f["close"], f["volume"] = O, H, L, C, V
    f["chg"] = (C / C.shift(1) - 1) * 100
    rng = (H - L).replace(0, np.nan)
    f["pos"] = ((C - L) / rng).fillna(1.0)
    amt = C * V / 1e8
    f["amt_eok"] = amt
    f["amt20"] = amt.rolling(20).mean()
    f["nbar"] = np.arange(1, len(h) + 1)
    ma50, ma200 = C.rolling(50).mean(), C.rolling(200).mean()
    f["ma50"], f["ma200"] = ma50, ma200
    f["ma50_up"] = ma50 >= ma50.shift(10)
    f["ma200_up"] = ma200 >= ma200.shift(20)
    f["hi52"] = H.rolling(250, min_periods=200).max()
    f["off_hi"] = (C / f["hi52"] - 1) * 100
    # 베이스: 직전 base_len일 고점(피벗)과 깊이
    bl = int(P["base_len"])
    piv = H.shift(1).rolling(bl).max()
    f["pivot"] = piv
    f["depth"] = (1 - L.shift(1).rolling(bl).min() / piv) * 100
    f["piv_age"] = _days_since_max(H, bl)
    f["prev_close"] = C.shift(1)
    vol50 = V.shift(1).rolling(50).mean().clip(lower=1)
    f["vol_ratio"] = V / vol50
    # L: RS 원점수 (전 종목 순위는 build에서)
    roc = lambda n: C / C.shift(n) - 1
    f["rs_raw"] = 0.4 * roc(63) + 0.2 * roc(126) + 0.2 * roc(189) + 0.2 * roc(252)
    # I: 50일 상승일/하락일 거래량 비율
    d = C.diff()
    upv = V.where(d > 0, 0).rolling(50).sum()
    dnv = V.where(d < 0, 0).rolling(50).sum().clip(lower=1)
    f["ud"] = upv / dnv
    # 눌림목용: 최근 60일 안에 거래량 실린 베이스 돌파가 있었는지
    brk = (C > piv) & (f["vol_ratio"] >= P["min_vol"])
    f["brk_recent"] = brk.shift(1).rolling(60, min_periods=1).max().fillna(0) > 0
    # 실적 (공시 다음 날부터, 200일 넘게 새 공시가 없으면 무효)
    # 각 날짜에 직전 공시 한 줄 전체를 씀 (빈 값을 예전 값으로 채우지 않음)
    _attach(f, fund, FUND_COLS, ["ql"], 200)
    # 품목 수출 (공표 다음 날부터, 70일 넘게 새 수치가 없으면 무효)
    _attach(f, exp, EXP_COLS, ["xm", "xl"], 70)
    return f


def add_rs(F):
    """날짜별 전 종목 RS 순위 → 1~99"""
    F["rs"] = (F.groupby(level=0)["rs_raw"].rank(pct=True) * 98 + 1).round()
    return F


def _score(x, kind):
    lin = lambda v, a, b: ((v - a) / (b - a)).clip(0, 1)
    s = (lin(x["rs"], 50, 99) * 20 +
         lin(x["cg"], 0, 100).fillna(0) * 20 +
         lin(x["ag"], 0, 50).fillna(0) * 15 +
         lin(x["off_hi"], -25, 0) * 15 +
         lin(x["ud"], 0.8, 1.8) * 15)
    if kind == "B":
        s = s + lin(x["vol_ratio"], 1, 3) * 15
    else:                                         # 눌림: 거래량이 마를수록 좋음
        s = s + lin(1.2 - x["vol_ratio"], 0, 0.8) * 15
    return s.round(1)


def op_accel(og, ogp, min_g=30):
    """영업이익 가속: 이번 분기 증가율이 min_g% 이상이고 직전 분기 증가율보다 높음
    (직전 분기가 적자여서 증가율이 없으면 이번 분기 흑자 전환·증가만으로 인정)"""
    if not np.isfinite(og) or og < min_g:
        return False
    return (not np.isfinite(ogp)) or og > ogp


def _tags(x, kind):
    out = []
    for r in x.itertuples():
        t = []
        if r.high >= r.hi52 - 1e-9: t.append("52주신고가")
        if r.rs >= 90: t.append("RS90+")
        if np.isfinite(r.cg) and np.isfinite(r.cgp) and r.cg > r.cgp: t.append("이익가속")
        if np.isfinite(r.cg) and r.cg >= 100: t.append("이익2배+")
        if op_accel(r.og, r.ogp): t.append("영업익가속")
        if np.isfinite(r.opm_d) and r.opm_d > 0: t.append("마진개선")
        if np.isfinite(r.xg) and r.xg >= 20:
            t.append(f"수출{r.xg:+.0f}%" + ("↑" if np.isfinite(r.xgp) and r.xg > r.xgp else ""))
        if np.isfinite(r.roe) and r.roe >= 17: t.append("ROE17+")
        if r.ud >= 1.5: t.append("매집")
        if kind == "B":
            t.append(f"베이스{r.depth:.0f}%·{int(r.piv_age)}일")
            t.append(f"거래량x{r.vol_ratio:.1f}")
        else:
            t.append("50일선지지")
        out.append(" ".join(t))
    return out


def pick_day(day, P):
    """하루치 전 종목 지표에서 후보 선정 (돌파형 B + 50일선 눌림형 P). 실적 조건은 앱에서 거름"""
    out = []
    base = day[(day["close"] >= P["min_price"]) & (day["amt20"] >= P["min_amount_eok"]) &
               (day["nbar"] >= 253) & (day["rs"] >= P["min_rs"]) &
               (day["close"] > day["ma50"]) & (day["ma50"] > day["ma200"]) & day["ma200_up"]]

    # ---------- 돌파형: 베이스 피벗을 거래량 실어 처음 돌파 ----------
    b = base[(base["close"] > base["pivot"]) & (base["prev_close"] <= base["pivot"]) &
             (base["close"] <= base["pivot"] * (1 + P["max_ext"] / 100)) &
             base["depth"].between(P["min_depth"], P["max_depth"]) & (base["piv_age"] >= P["min_base_days"]) &
             (base["vol_ratio"] >= P["min_vol"]) & (base["chg"] > 0) & (base["pos"] >= 0.5) &
             (base["off_hi"] >= -P["max_off_hi"])]
    if len(b):
        b = b.assign(score=_score(b, "B"), type="B", tags=_tags(b, "B"))
        b = b.sort_values("score", ascending=False).head(P["store_n"])
        out.append(b.assign(rank=np.arange(1, len(b) + 1)))

    # ---------- 눌림형: 돌파 후 50일선까지 조용히 눌렸다 지지 ----------
    p = base[(base["rs"] >= P["pb_min_rs"]) & base["brk_recent"] & base["ma50_up"] &
             (base["low"] <= base["ma50"] * (1 + P["pb_touch"] / 100)) & (base["close"] >= base["ma50"]) &
             (base["vol_ratio"] <= P["pb_max_vol"]) & (base["chg"] >= 0) & (base["off_hi"] >= -P["pb_max_off_hi"])]
    if len(p):
        p = p.assign(score=_score(p, "P"), type="P", tags=_tags(p, "P"))
        p = p.sort_values("score", ascending=False).head(P["store_n"])
        out.append(p.assign(rank=np.arange(1, len(p) + 1)))
    return pd.concat(out) if out else pd.DataFrame()


def market_trend(close):
    """지수 추세: U 상승(50일선 위, 50일선 상승) / C 조정 / D 하락(200일선 아래)"""
    ma50, ma200 = close.rolling(50).mean(), close.rolling(200).mean()
    up = (close > ma50) & (ma50 >= ma50.shift(5))
    dn = close < ma200
    return pd.Series(np.where(up, "U", np.where(dn, "D", "C")), index=close.index)
