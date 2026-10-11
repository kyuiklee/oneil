"""매일 자동 실행: 시세·실적 수집 → 오늘 CAN SLIM 후보 + 과거 후보(백테스트) 계산 → 앱용 JSON 저장"""
import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screen import features, add_rs, pick_day, market_trend
import data as D
import fund as FD
import export as EX

ROOT = D.ROOT
OUT = os.path.join(ROOT, "site", "data")
STOPS = (5, 7, 8, 10)          # 앱에서 고를 수 있는 손절 % (매수가 대비)
TGTS = (15, 20, 25, 30)        # 앱에서 고를 수 있는 익절 %


def kst_now():
    return pd.Timestamp.now(tz="Asia/Seoul")


def r2(x, n=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else round(x, n)


def i10(x):
    """수익률 %를 0.1% 단위 정수로 (용량 절약)"""
    return int(round(float(x) * 10))


def exit_info(entry, fut, levels, is_stop):
    """레벨별로 처음 걸린 날(1부터)과 그때 수익률(0.1% 정수). 시가부터 넘으면 시가에 체결"""
    out = []
    O, Hh, Lw = fut["Open"].to_numpy(), fut["High"].to_numpy(), fut["Low"].to_numpy()
    for lv in levels:
        px = entry * (1 - lv / 100) if is_stop else entry * (1 + lv / 100)
        day, ret = 0, 0
        for j in range(len(O)):
            if is_stop and O[j] <= px:
                day, ret = j + 1, O[j] / entry - 1; break
            if is_stop and Lw[j] <= px:
                day, ret = j + 1, -lv / 100; break
            if not is_stop and O[j] >= px:
                day, ret = j + 1, O[j] / entry - 1; break
            if not is_stop and Hh[j] >= px:
                day, ret = j + 1, lv / 100; break
        out += [day, i10(ret * 100)]
    return out


def main():
    t0 = time.time()
    P = json.load(open(os.path.join(ROOT, "engine", "config.json"), encoding="utf-8"))
    now = kst_now()
    print(f"실행 {now:%Y-%m-%d %H:%M} KST")

    univ = D.load_universe()
    meta = univ.set_index("code")
    hist = D.load_histories(list(univ["code"]))
    if len(hist) < 1000:
        raise SystemExit(f"일봉 수집 종목이 너무 적습니다 ({len(hist)}) — 데이터 소스 문제")
    fund = FD.load(list(hist))
    has_fund = bool(fund)
    xp = EX.load(list(hist))
    has_exp = bool(xp)

    keep_days = int(P.get("backtest_days", 500)) + 5
    feats = []
    for c, h in hist.items():
        if len(h) < 60:
            continue
        f = features(c, h, P, fund.get(c) if fund else None, xp.get(c) if xp else None)
        feats.append(f.iloc[-keep_days:])
    F = pd.concat(feats)
    F = F[F.index >= F.index.max() - pd.Timedelta(days=int(keep_days * 1.5))]
    counts = F.groupby(level=0).size()
    days = counts[counts >= len(hist) * 0.5].index
    F = add_rs(F[F.index.isin(days)].copy())
    print(f"지표 계산 완료: {len(days)}거래일, 최종일 {days.max().date()} ({time.time() - t0:.0f}초)")

    picks = []
    for d, day in F.groupby(level=0):
        pk = pick_day(day, P)
        if len(pk):
            picks.append(pk.assign(date=d))
    T = pd.concat(picks) if picks else pd.DataFrame(columns=["date", "type", "rank", "code"])
    T["name"] = T["code"].map(meta["name"])
    T["mkt"] = np.where(T["code"].map(meta["market"]) == "KOSDAQ", "Q", "K")

    # ----- 시장 방향 (M) -----
    idx = D.index_daily()
    trend = {k: market_trend(s) for k, s in idx.items()}
    mk_of = {"K": "KOSPI", "Q": "KOSDAQ"}
    T["m"] = [trend.get(mk_of[m], pd.Series(dtype=object)).get(d, "C") for d, m in zip(T["date"], T["mkt"])]

    trade_day = days.max()
    is_today = trade_day.date() == now.date()
    hm = now.hour * 100 + now.minute
    status = "final" if (not is_today or hm >= 1530) else "preview"
    market = {}
    for k, s in idx.items():
        s = s[s.index <= trade_day]
        if len(s) >= 2:
            market[k] = {"chg": r2((s.iloc[-1] / s.iloc[-2] - 1) * 100),
                         "m": str(trend[k].get(trade_day, "C")),
                         "ma50": r2((s.iloc[-1] / s.rolling(50).mean().iloc[-1] - 1) * 100, 1)}

    def card(r):
        return {"code": r.code, "name": r.name, "mkt": r.mkt, "type": r.type, "rank": int(r.rank),
                "score": r2(r.score, 1), "close": int(r.close), "chg": r2(r.chg), "amt": int(round(r.amt20)),
                "rs": int(r.rs), "cg": r2(r.cg, 0), "cgp": r2(r.cgp, 0), "sg": r2(r.sg, 0), "ag": r2(r.ag, 0),
                "og": r2(r.og, 0), "ogp": r2(r.ogp, 0), "opm": r2(r.opm, 1), "opm_d": r2(r.opm_d, 1),
                "xg": r2(r.xg, 0), "xgp": r2(r.xgp, 0),
                "xl": r.xl if isinstance(r.xl, str) else None, "xm": r.xm if isinstance(r.xm, str) else None,
                "roe": r2(r.roe, 1), "ql": r.ql if isinstance(r.ql, str) else None,
                "vr": r2(r.vol_ratio, 1), "depth": r2(r.depth, 0), "pivot": int(round(r.pivot)),
                "off_hi": r2(r.off_hi, 1), "ud": r2(r.ud, 2), "m": r.m, "tags": r.tags}

    today = {
        "trade_date": f"{trade_day:%Y-%m-%d}",
        "generated_at": f"{now:%Y-%m-%d %H:%M}",
        "status": status,
        "fund": has_fund,
        "exp": has_exp,
        "market": market,
        "config": {k: P[k] for k in ("top_n", "min_rs", "min_amount_eok", "stop_pct", "target_pct", "fee_pct")},
        "cands": [card(r) for r in T[T["date"] == trade_day].sort_values(["type", "rank"]).itertuples()],
    }

    # ----- 과거 후보 + 매수 후 경로 (백테스트·기록·내 전략 공용) -----
    HOLD = int(P["hold_max"])
    hn = {}
    for c in T["code"].unique():
        h = hist[c][["Open", "High", "Low", "Close"]].astype(float).copy()
        h.index = pd.to_datetime(h.index).normalize()
        hn[c] = h[~h.index.duplicated(keep="last")].sort_index()
    rows = []
    for r in T[T["date"] < trade_day].sort_values(["date", "type", "rank"]).itertuples():
        h = hn[r.code]
        pos = h.index.get_indexer([r.date])[0]
        if pos < 0:
            continue
        fut = h.iloc[pos + 1: pos + 1 + HOLD]
        if not len(fut):
            continue
        c = [i10((x / r.close - 1) * 100) for x in fut["Close"]]
        rows.append([f"{r.date:%Y-%m-%d}", r.type, int(r.rank), r.code, r.name, r2(r.score, 1), int(r.close), r2(r.chg),
                     r.mkt, int(r.rs), r2(r.cg, 0), r2(r.ag, 0), r2(r.sg, 0), r2(r.roe, 1), r.m, c,
                     exit_info(r.close, fut, STOPS, True), exit_info(r.close, fut, TGTS, False),
                     r2(r.og, 0), r2(r.ogp, 0), r2(r.opm_d, 1), r2(r.xg, 0)])
    trades = {
        "fields": ["date", "type", "rank", "code", "name", "score", "close", "chg", "mkt", "rs", "cg", "ag", "sg", "roe", "m",
                   "c", "xs", "xt", "og", "ogp", "opm_d", "xg"],
        "stops": STOPS, "tgts": TGTS, "hold": HOLD, "fund": has_fund, "exp": has_exp,
        "note": "매수=추천일 종가. c=1~60일째 종가 수익률(0.1% 단위), xs/xt=손절·익절 레벨별 [처음 걸린 날, 수익률(0.1%)] (0=안 걸림). 비용 차감 전",
        "rows": rows,
    }
    bench = {k.lower(): [[f"{d:%Y-%m-%d}", r2(v)] for d, v in s.items() if d >= days.min()] for k, s in idx.items()}

    os.makedirs(OUT, exist_ok=True)
    for name, obj in (("today.json", today), ("trades.json", trades), ("bench.json", bench)):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    nb = sum(1 for x in today["cands"] if x["type"] == "B")
    print(f"저장: 오늘 돌파 {nb} / 눌림 {len(today['cands']) - nb}, 과거 후보 {len(rows)}건, "
          f"실적 {'있음' if has_fund else '없음'}, 수출 {'있음' if has_exp else '없음'}, 상태 {status} ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
