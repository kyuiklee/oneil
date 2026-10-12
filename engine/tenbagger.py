"""일회성 분석: 2025-01-01 이후 저점 대비 10배 이상 오른 적이 있는 종목 (현재 상장 종목만)
build.py가 남긴 일봉 캐시(cache/hist.pkl.gz)와 종목 목록(engine/tickers.csv)을 씀 → site/data/tenbagger.json"""
import os, gzip, json, pickle
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START = pd.Timestamp("2025-01-01")
MULT = 10.0

hist = pickle.load(gzip.open(os.path.join(ROOT, "cache", "hist.pkl.gz"), "rb"))
meta = pd.read_csv(os.path.join(ROOT, "engine", "tickers.csv"), dtype={"code": str}).set_index("code")
out = []
for code, h in hist.items():
    h = h[h.index >= START]
    if len(h) < 20:
        continue
    lo, hi, cl = h["Low"].astype(float).to_numpy(), h["High"].astype(float).to_numpy(), h["Close"].astype(float).to_numpy()
    lo = np.where(lo > 0, lo, cl)
    run_min = np.minimum.accumulate(lo)                     # 그날까지의 최저가
    ratio = hi / run_min                                      # 그날 고가 ÷ 그 전까지 최저가
    j = int(np.nanargmax(ratio))
    if not np.isfinite(ratio[j]) or ratio[j] < MULT:
        continue
    i = int(np.argmin(lo[: j + 1]))                           # 저점 날짜
    cur = cl[-1]
    out.append({"code": code, "name": str(meta["name"].get(code, "")), "mkt": str(meta["market"].get(code, "")),
                "low_date": f"{h.index[i]:%Y-%m-%d}", "low": round(lo[i]), "peak_date": f"{h.index[j]:%Y-%m-%d}", "peak": round(hi[j]),
                "mult_peak": round(float(ratio[j]), 1), "cur_date": f"{h.index[-1]:%Y-%m-%d}", "cur": round(cur),
                "mult_now": round(float(cur / lo[i]), 1), "from_peak": round(float((cur / hi[j] - 1) * 100), 1)})
out.sort(key=lambda r: -r["mult_peak"])
os.makedirs(os.path.join(ROOT, "site", "data"), exist_ok=True)
json.dump({"start": f"{START:%Y-%m-%d}", "mult": MULT, "n_scanned": len(hist), "rows": out},
          open(os.path.join(ROOT, "site", "data", "tenbagger.json"), "w", encoding="utf-8"), ensure_ascii=False)
print(f"10배 종목 {len(out)}개 / {len(hist)}종목")
