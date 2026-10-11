"""관세청 품목(HS코드)별 월간 수출 → 종목별 '수출 증가율' 지표

- 공공데이터포털 '관세청_품목별 수출입실적' API (인증키: 저장소 Secrets의 CUSTOMS_API_KEY)
- 어떤 종목이 어떤 품목을 수출하는지는 engine/export_map.json 에 직접 적어 둠 (자동으로 알 수 없음)
- 지표: 최근 3개월 수출액 합계의 전년 동기 대비 증가율(xg)과 한 달 전 같은 값(xgp) → 둘을 비교해 수출 가속 판단
- 월간 확정치는 다음 달 15일 무렵 공표되므로, 그달 수치는 다음 달 16일부터 쓴다 (미래 정보가 섞이지 않게)
"""
import os, time, gzip, json, pickle
import xml.etree.ElementTree as ET
from urllib.parse import unquote
import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache", "export.pkl.gz")
MAP = os.path.join(ROOT, "engine", "export_map.json")
API = "https://apis.data.go.kr/1220000/Itemtrade/getItemtradeList"
YEARS_BACK = 4          # 백테스트 기간(약 2년) + 전년 비교에 필요한 만큼
PUBLISH_DAY = 16        # 그달 수치를 쓰기 시작하는 다음 달 날짜


def _key():
    k = os.environ.get("CUSTOMS_API_KEY", "").strip()
    # 포털의 '인코딩' 키를 넣어도 동작하게: requests가 다시 인코딩하므로 한 번 풀어 둠
    return unquote(k) if "%" in k else k


def _fetch(key, hs, ym0, ym1):
    """한 품목의 ym0~ym1(최대 12개월) 월별 수출액 {YYYYMM: 달러}. 실패하면 None"""
    params = {"serviceKey": key, "strtYymm": ym0, "endYymm": ym1, "hsSgn": hs}
    for attempt in range(3):
        try:
            r = requests.get(API, params=params, timeout=30)
            root = ET.fromstring(r.content)
            code = (root.findtext(".//resultCode") or "").strip()
            if code not in ("00", "0", "000"):
                msg = root.findtext(".//resultMsg") or root.findtext(".//returnAuthMsg") or r.text[:120]
                print(f"  관세청 {hs} {ym0}-{ym1}: {code} {msg}")
                return None
            out = {}
            for it in root.iter("item"):
                ym = (it.findtext("year") or "").strip().replace(".", "").replace("-", "")
                if len(ym) != 6 or not ym.isdigit():          # '총계' 같은 합계 줄은 건너뜀
                    continue
                try:
                    v = float((it.findtext("expDlr") or "0").replace(",", ""))
                except ValueError:
                    continue
                out[ym] = out.get(ym, 0.0) + v               # 하위 HS코드로 나뉘어 오면 합산
            return out
        except Exception as e:
            if attempt == 2:
                print(f"  관세청 {hs} {ym0}-{ym1} 실패: {e}")
            time.sleep(2 + attempt * 2)
    return None


def _chunks(today):
    """(시작, 끝) 연월 묶음: 1년 단위, 지난달까지"""
    last = (today.replace(day=1) - pd.Timedelta(days=1))
    out = []
    for y in range(today.year - YEARS_BACK, today.year + 1):
        a, b = pd.Timestamp(y, 1, 1), min(pd.Timestamp(y, 12, 1), last.replace(day=1))
        if a <= b:
            out.append((f"{a:%Y%m}", f"{b:%Y%m}"))
    return out


def load_monthly(hs_list, key, refresh_hours=20):
    """품목별 월간 수출 {hs: pd.Series(월초 날짜 → 달러)}. 지난 연도는 캐시, 올해·작년은 하루 한 번 새로"""
    cache = {"raw": {}, "at": {}}
    if os.path.exists(CACHE):
        try:
            with gzip.open(CACHE, "rb") as f:
                cache = pickle.load(f)
        except Exception as e:
            print(f"수출 캐시 읽기 실패: {e}")
    now = time.time()
    today = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
    n_req = 0
    for hs in hs_list:
        for ym0, ym1 in _chunks(today):
            k = (hs, ym0)
            recent = int(ym0[:4]) >= today.year - 1
            if k in cache["raw"] and cache["raw"][k].get("_end") == ym1 and \
                    not (recent and now - cache["at"].get(k, 0) > refresh_hours * 3600):
                continue
            d = _fetch(key, hs, ym0, ym1)
            n_req += 1
            if d is None:
                continue
            d["_end"] = ym1
            cache["raw"][k], cache["at"][k] = d, now
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with gzip.open(CACHE, "wb") as f:
        pickle.dump(cache, f)
    out = {}
    for hs in hs_list:
        m = {}
        for (h, _), d in cache["raw"].items():
            if h == hs:
                m.update({ym: v for ym, v in d.items() if not ym.startswith("_")})
        if m:
            s = pd.Series(m)
            s.index = pd.to_datetime(s.index, format="%Y%m")
            out[hs] = s.sort_index()
    print(f"관세청 수출: 요청 {n_req}회, 품목 {len(out)}/{len(hs_list)}개")
    return out


def indicators(s):
    """월간 수출 → 공표일(다음 달 16일) 기준 지표 시계열 [xg, xgp, xm]"""
    s = s.asfreq("MS")                                       # 빠진 달은 NaN (합계 왜곡 방지)
    r3 = s.rolling(3).sum()                                  # 빠진 달이 있으면 NaN
    xg = (r3 / r3.shift(12) - 1) * 100
    df = pd.DataFrame({"xg": xg, "xgp": xg.shift(1),
                       "xm": [f"{d:%y.%m}" for d in s.index]}, index=s.index)
    df = df[np.isfinite(df["xg"])]
    df.index = df.index + pd.offsets.MonthBegin(1) + pd.Timedelta(days=PUBLISH_DAY - 1)
    return df


def load(codes):
    """{종목코드: 지표 시계열(xg, xgp, xm, xl)}. 키나 매핑이 없으면 None"""
    key = _key()
    if not key:
        print("CUSTOMS_API_KEY 없음 — 수출 지표 없이 실행")
        return None
    try:
        mp = json.load(open(MAP, encoding="utf-8"))
    except Exception as e:
        print(f"수출 매핑 파일 읽기 실패: {e}")
        return None
    items = {hs: v for hs, v in mp.items() if not hs.startswith("_")}
    try:
        monthly = load_monthly(list(items), key)
    except Exception as e:
        print(f"관세청 수출 실패: {e}")
        return None
    ind = {hs: indicators(s) for hs, s in monthly.items()}
    codes = set(codes)
    out = {}
    for hs, v in items.items():
        if hs not in ind or not len(ind[hs]):
            continue
        df = ind[hs].assign(xl=f"{v.get('name', hs)}({hs})")
        for c in v.get("codes", []):
            c = str(c).zfill(6)
            if c in codes and c not in out:                  # 한 종목이 여러 품목에 있으면 먼저 적은 품목을 씀
                out[c] = df
    print(f"수출 지표 연결 {len(out)}종목")
    return out or None
