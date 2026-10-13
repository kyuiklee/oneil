"""일회성 (머지 금지): 반도체 소부장 가성비 스크린 — 시총 대비 이익, 이익 회복, 직전 사이클 최대 이익 대비 위치, 주가 위치"""
import os, sys, gzip, json, pickle, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import fund as FD

ROOT = D.ROOT
hist = pickle.load(gzip.open(os.path.join(ROOT, "cache", "hist.pkl.gz"), "rb"))
raw = pickle.load(gzip.open(os.path.join(ROOT, "cache", "fund.pkl.gz"), "rb"))["raw"]
import FinanceDataReader as fdr
ind = pd.DataFrame(columns=["ind"])
for src in ("KRX-DESC", "KOSDAQ-DESC"):
    try:
        d = fdr.StockListing(src).rename(columns={"Code": "code", "Industry": "ind"})
        d["code"] = d["code"].astype(str).str.zfill(6); ind = d.set_index("code")[["ind"]]; break
    except Exception as e:
        print("업종 목록 실패", src, e)
lst = fdr.StockListing("KRX").rename(columns={"Code": "code", "Stocks": "shares", "Marcap": "mcap"})
lst["code"] = lst["code"].astype(str).str.zfill(6); lst = lst.set_index("code")
print("KRX 열", list(lst.columns)[:12])

EXTRA = """솔브레인 동진쎄미켐 원익QnC 하나머티리얼즈 티씨케이 월덱스 케이엔제이 에스앤에스텍 덕산테코피아 이엔에프테크놀로지 램테크놀러지 미코 코미코
영창케미칼 한솔케미칼 피에스케이홀딩스 ISC 리노공업 티에스이 마이크로컨텍솔 오킨스전자 샘씨엔에스 디아이 유니테스트 와이씨 네오셈 엑시콘 하나마이크론
SFA반도체 두산테스나 엘비세미콘 테스 유진테크 원익IPS 주성엔지니어링 피에스케이 이오테크닉스 한미반도체 제우스 케이씨텍 에스티아이 씨앤지하이테크 저스템
에프에스티 오로스테크놀로지 넥스틴 파크시스템스 인텍플러스 HPSP 아이에스티이 기가비스 브이엠 지오엘리먼트 프로텍 대덕전자 심텍 코리아써키트 티엘비
아비코전자 윈팩 테크윙 원익머트리얼즈 후성 SK스퀘어 레이크머티리얼즈 디엔에프 이수스페셜티케미컬 비씨엔씨 케이씨 유니셈 에스에이엠티 가온칩스 에이디테크놀로지
코세스 GST 성도이엔지 신성이엔지 원익홀딩스 에이피티씨 엘오티베큠 한양이엔지 오션브릿지 시지트로닉스 큐알티 아이티엠반도체 동운아나텍 퀄리타스반도체
제너셈 마이크로투나노 티에프이 에이엘티 펨트론 ICTK 덕산하이메탈 SKC 에스씨디 한솔아이원스 디바이스이엔지 상아프론테크 와이아이케이 미래반도체 네패스
엠케이전자 케이엠 넥스틴 다원시스 아이엠티 쎄믹스 HB테크놀러지 이녹스첨단소재 피델릭스 제주반도체 KEC 매커스 어보브반도체 텔레칩스 칩스앤미디어""".split()
names = set(EXTRA)
def is_semi(code):
    if code in ind.index:
        s = str(ind.at[code, "ind"])
        if "반도체" in s: return True
    nm = str(lst["Name"].get(code, "")) if "Name" in lst else ""
    return nm in names

def quarters(code):
    """분기별 매출·영업이익·순이익 (fund.series와 같은 방식)"""
    Q, A = {}, {}
    for y in sorted({y for y, _ in raw}):
        cum = {}
        for qn, rp in enumerate(FD.REPRTS, 1):
            rec = raw.get((y, rp), {}).get(code)
            if rec is None: continue
            vals = {}
            for k in ("ni", "rev", "op"):
                amt, add = rec.get(k, np.nan), rec.get(k + "_cum", np.nan); prev = cum.get((qn - 1, k), np.nan)
                if qn == 1: c, qv = amt, amt
                elif qn < 4:
                    c = add if np.isfinite(add) else prev + amt
                    qv = add - prev if (np.isfinite(add) and amt == add) else amt
                else: c, qv = amt, amt - cum.get((3, k), np.nan)
                cum[(qn, k)] = c; vals[k] = qv
            Q[y * 4 + qn - 1] = vals
            if qn == 4: A[y] = rec.get("eq", np.nan)
    return Q, A

CL = pd.DataFrame({c: h["Close"].astype(float) for c, h in hist.items() if len(h) > 260})
CL.index = pd.to_datetime(CL.index).normalize(); CL = CL[~CL.index.duplicated()].sort_index().ffill(limit=5)
roc = lambda n: CL / CL.shift(n) - 1
RS = (0.4 * roc(63) + 0.2 * roc(126) + 0.2 * roc(189) + 0.2 * roc(252)).rank(axis=1, pct=True).iloc[-1] * 98 + 1

rows = []
for code, h in hist.items():
    if not is_semi(code): continue
    h = h.astype(float); h.index = pd.to_datetime(h.index); h = h[~h.index.duplicated()].sort_index()
    if len(h) < 260: continue
    cl = h["Close"].iloc[-1]; sh = lst["shares"].get(code, np.nan)
    if not np.isfinite(sh): continue
    mcap = cl * sh / 1e8
    amt20 = float((h["Close"] * h["Volume"]).iloc[-20:].mean() / 1e8)
    Q, A = quarters(code)
    if len(Q) < 8: continue
    qs = sorted(Q); last = qs[-1]
    ttm = lambda k, end: np.nansum([Q.get(q, {}).get(k, np.nan) for q in range(end - 3, end + 1)]) if all(np.isfinite(Q.get(q, {}).get(k, np.nan)) for q in range(end - 3, end + 1)) else np.nan
    op4, ni4, rev4 = ttm("op", last), ttm("ni", last), ttm("rev", last)
    peak = max([ttm("op", q) for q in qs if np.isfinite(ttm("op", q))] or [np.nan])
    opq = [Q[q].get("op", np.nan) for q in qs[-5:]]
    og = (opq[-1] / opq[0] - 1) * 100 if len(opq) == 5 and opq[0] > 0 else (100.0 if len(opq) == 5 and opq[-1] > 0 >= opq[0] else np.nan)
    eq = A.get(max(A)) if A else np.nan
    hi52, lo52 = h["High"].iloc[-250:].max(), h["Low"].iloc[-250:].min()
    rows.append(dict(code=code, name=str(lst["Name"].get(code, "")), ind=str(ind["ind"].get(code, ""))[:14], mcap=round(mcap), amt=round(amt20, 1),
                     op4=round(op4 / 1e8) if np.isfinite(op4) else None, ni4=round(ni4 / 1e8) if np.isfinite(ni4) else None,
                     opm=round(op4 / rev4 * 100, 1) if np.isfinite(op4) and rev4 else None,
                     per=round(mcap / (ni4 / 1e8), 1) if np.isfinite(ni4) and ni4 > 0 else None,
                     por=round(mcap / (op4 / 1e8), 1) if np.isfinite(op4) and op4 > 0 else None,
                     pbr=round(mcap / (eq / 1e8), 2) if np.isfinite(eq) and eq > 0 else None,
                     peak_op=round(peak / 1e8) if np.isfinite(peak) else None,
                     vs_peak=round(op4 / peak * 100) if np.isfinite(op4) and np.isfinite(peak) and peak > 0 else None,
                     og=round(og) if np.isfinite(og) else None,
                     opq=[round(x / 1e8) if np.isfinite(x) else None for x in opq],
                     off_hi=round((cl / hi52 - 1) * 100), from_lo=round((cl / lo52 - 1) * 100),
                     r1y=round((cl / h["Close"].iloc[-250] - 1) * 100), rs=int(RS.get(code, np.nan)) if np.isfinite(RS.get(code, np.nan)) else None))
df = pd.DataFrame(rows)
print(f"반도체 소부장 {len(df)}종목")
os.makedirs(os.path.join(ROOT, "site", "data"), exist_ok=True)
df.to_json(os.path.join(ROOT, "site", "data", "semi.json"), orient="records", force_ascii=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_colwidth", 40)
cols = ["name", "mcap", "op4", "opm", "per", "por", "pbr", "peak_op", "vs_peak", "og", "opq", "off_hi", "from_lo", "r1y", "rs", "amt"]
ok = df[(df.op4.fillna(0) > 0) & (df.amt >= 3) & (df.mcap >= 300)]
print("== 전체 (흑자, 거래대금 3억↑, 시총 300억↑) PER 순")
print(ok.sort_values("per", na_position="last")[cols].to_string(index=False))
