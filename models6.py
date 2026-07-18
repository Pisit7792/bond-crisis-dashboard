"""
models6.py — 6 โมเดลทำกำไร (deterministic, โปร่งใส, ทดสอบได้)

หลักการซื่อสัตย์:
1. คะแนนทุกตัวคำนวณจากสูตรที่เปิดเผย (percentile ของตัวชี้วัดจริง) ไม่มี AI แต่งเลข
2. ตาราง 'ได้ประโยชน์/เสียประโยชน์' คือ "แนวโน้มตามประวัติศาสตร์" ไม่ใช่กฎตายตัว —
   ความสัมพันธ์พังได้ (เช่น หุ้น-บอนด์ปี 2022, ทอง+BTC ตอนแบงก์รัน 2023)
   ทุกโมเดลจึงมีช่อง note เตือนความไม่เสถียรกำกับ
3. คะแนน ≠ ความน่าจะเป็นกำไร — เป็นมาตรวัดว่า 'สภาพแวดล้อมแบบนั้นกำลังเด่นแค่ไหน'
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import engine as E


def _pct_rank(series: pd.Series, invert: bool = False, min_n: int = 60) -> float:
    """เปอร์เซ็นไทล์ (0-100) ของค่าล่าสุดเทียบประวัติตัวเอง; invert=True กลับด้าน."""
    s = pd.Series(series).dropna()
    if len(s) < min_n:
        return float("nan")
    p = float((s <= s.iloc[-1]).mean() * 100.0)
    return 100.0 - p if invert else p


def _chg(series: pd.Series, periods: int) -> pd.Series:
    return pd.Series(series).dropna().diff(periods)


def _pchg(series: pd.Series, periods: int) -> pd.Series:
    return pd.Series(series).dropna().pct_change(periods) * 100.0


# ---------------------------------------------------------------------------
# นิยาม 6 โมเดล: components = (ชื่อ, ฟังก์ชันจาก data->Series, invert)
# data คือ dict ของ pd.Series (คีย์ = FRED id หรือ "MOVE")
# ---------------------------------------------------------------------------

MODEL_DEFS: dict[str, dict] = {
    "inflation_oil": {
        "th": "เงินเฟ้อ-น้ำมัน",
        "desc": "สภาพแวดล้อมเงินเฟ้อจากต้นทุนพลังงานเร่งตัว",
        "components": [
            ("น้ำมัน %chg 3 เดือน", lambda d: _pchg(d.get("DCOILWTICO"), 63), False),
            ("เงินเฟ้อคาดหวัง 5 ปี (T5YIE)", lambda d: d.get("T5YIE"), False),
            ("CPI YoY", lambda d: _pchg(d.get("CPIAUCSL"), 12), False),
        ],
    },
    "yield_shock": {
        "th": "Yield ช็อก",
        "desc": "ดอกเบี้ยระยะยาวพุ่งเร็ว/ผันผวนสูง (bond sell-off)",
        "components": [
            ("|Δ10Y 20 วัน| (แรงของการพุ่ง)",
             lambda d: _chg(d.get("DGS10"), 20).abs(), False),
            ("MOVE (ความผันผวนพันธบัตร)", lambda d: d.get("MOVE"), False),
            ("ความเร็วการชัน 10Y-2Y",
             lambda d: _chg(d.get("T10Y2Y"), 20).abs(), False),
        ],
    },
    "recovery": {
        "th": "ฟื้นตัว",
        "desc": "risk-on: ความเครียดคลาย เครดิตแคบลง แนวโน้มสินทรัพย์เสี่ยงดีขึ้น",
        "components": [
            ("Stress index กำลังลด (Δ3 เดือน)",
             lambda d: _chg(d.get("STLFSI4"), 13), True),
            ("HY spread กำลังแคบ (Δ1 เดือน)",
             lambda d: _chg(d.get("BAMLH0A0HYM2"), 21), True),
            ("ระดับ HY spread ต่ำ", lambda d: d.get("BAMLH0A0HYM2"), True),
        ],
    },
    "fed_pivot": {
        "th": "Fed เปลี่ยนท่าที",
        "desc": "ตลาดเริ่ม price การผ่อนคลาย: 2Y ร่วงเร็ว, cut ถูกคาดการณ์",
        "components": [
            ("2Y กำลังร่วง (Δ20 วัน)", lambda d: _chg(d.get("DGS2"), 20), True),
            ("2Y ต่ำกว่าดอกเบี้ยนโยบาย (EFFR-2Y)",
             lambda d: (d.get("EFFR") - d.get("DGS2")).dropna()
             if d.get("EFFR") is not None and d.get("DGS2") is not None else None,
             False),
            ("Curve ชันขึ้นจากก้นบึ้ง (Δ(10-2) 20 วัน)",
             lambda d: _chg(d.get("T10Y2Y"), 20), False),
        ],
    },
    "credit_crisis": {
        "th": "วิกฤตสินเชื่อ",
        "desc": "ตลาดเครดิตตึง: spread กว้างและถ่างเร็ว",
        "components": [
            ("ระดับ HY spread", lambda d: d.get("BAMLH0A0HYM2"), False),
            ("HY ถ่างเร็ว (Δ1 เดือน)",
             lambda d: _chg(d.get("BAMLH0A0HYM2"), 21), False),
            ("Stress index", lambda d: d.get("STLFSI4"), False),
        ],
    },
    "bank_run": {
        "th": "แบงก์รัน",
        "desc": "เงินฝากไหลออก + แบงก์พึ่งหน้าต่างกู้ Fed + repo ตึง",
        "components": [
            ("เงินฝากธนาคาร %chg 2 สัปดาห์",
             lambda d: _pchg(d.get("DPSACBW027SBOG"), 2), True),
            ("ยอดกู้จาก Fed (BORROW)", lambda d: d.get("BORROW"), False),
            ("SOFR-EFFR (repo ตึง)",
             lambda d: (d.get("SOFR") - d.get("EFFR")).dropna()
             if d.get("SOFR") is not None and d.get("EFFR") is not None else None,
             False),
        ],
    },
}

# ---------------------------------------------------------------------------
# สินทรัพย์ได้/เสียประโยชน์ต่อโมเดล (แนวโน้มประวัติศาสตร์ — ไม่ใช่กฎตายตัว)
# ชื่อใช้ display name ให้ตรงกับ data_sources.YF_ASSETS
# ---------------------------------------------------------------------------

ASSET_IMPACT: dict[str, dict] = {
    "inflation_oil": {
        "benefit": ["USOIL", "XAUUSD", "AUDUSD"],
        "lose": ["NAS100", "US500", "TLT (20y+ UST ETF)"],
        "note": "หุ้น growth/บอนด์มักเสียเมื่อเงินเฟ้อเร่ง แต่ปี 2022 ทองก็ร่วงตาม "
                "real yield — ความสัมพันธ์ไม่เสถียร",
    },
    "yield_shock": {
        "benefit": ["DXY", "USDJPY"],
        "lose": ["TLT (20y+ UST ETF)", "NAS100", "XAUUSD", "BTC"],
        "note": "yield พุ่ง → ดอลลาร์มักแข็ง สินทรัพย์ duration ยาว/ไม่มี yield "
                "มักเสีย — แต่ตอน 'ขายทุกอย่าง' (มี.ค. 2020) ดอลลาร์เท่านั้นที่รอด",
    },
    "recovery": {
        "benefit": ["NAS100", "US500", "US30", "ETH", "SOL", "EURUSD",
                    "AUDUSD", "GBPUSD", "USOIL"],
        "lose": ["DXY"],
        "note": "โหมด risk-on กว้าง — แต่ breadth ที่เห็นคืออดีตถึงปัจจุบัน "
                "ไม่ใช่การันตีการฟื้นต่อ",
    },
    "fed_pivot": {
        "benefit": ["TLT (20y+ UST ETF)", "XAUUSD", "NAS100", "BTC", "ETH"],
        "lose": ["DXY"],
        "note": "ระวังกับดัก: pivot เพราะ 'เศรษฐกิจพัง' หุ้นมักลงก่อนขึ้น "
                "(2001, 2007 Fed ลดดอกแล้วหุ้นยังลงต่อ)",
    },
    "credit_crisis": {
        "benefit": ["TLT (20y+ UST ETF)", "DXY"],
        "lose": ["US500", "US30", "NAS100", "BTC", "ETH", "SOL", "USOIL"],
        "note": "flight-to-quality เข้าบอนด์รัฐ 'โดยปกติ' — ยกเว้นวิกฤตที่บอนด์"
                "คือปัญหาเอง (Gilt 2022) ซึ่งบอนด์ร่วงด้วย",
    },
    "bank_run": {
        "benefit": ["XAUUSD", "TLT (20y+ UST ETF)", "BTC"],
        "lose": ["US500", "US30"],
        "note": "อิงเหตุการณ์ มี.ค. 2023 (ทอง/บอนด์/BTC ขึ้น หุ้นแบงก์ร่วง) — "
                "หลักฐานจากวิกฤตเดียว จึงเป็นข้อสันนิษฐานที่อ่อนที่สุดในตารางนี้",
    },
}


def score_models(data: dict) -> dict[str, dict]:
    """คำนวณคะแนน 0-100 ของทั้ง 6 โมเดล + รายละเอียด component.

    คืน {key: {"th", "score", "components": {ชื่อ: percentile}, "missing": [...]}}
    คะแนน = ค่าเฉลี่ย percentile ของ components ที่มีข้อมูล (equal weights)
    """
    out = {}
    for key, spec in MODEL_DEFS.items():
        comps, missing = {}, []
        for name, fn, invert in spec["components"]:
            try:
                s = fn(data)
            except Exception:
                s = None
            v = _pct_rank(s, invert) if s is not None else float("nan")
            if math.isnan(v):
                missing.append(name)
            else:
                comps[name] = round(v, 1)
        score = float(np.mean(list(comps.values()))) if comps else float("nan")
        out[key] = {"th": spec["th"], "desc": spec["desc"],
                    "score": round(score, 1) if not math.isnan(score) else float("nan"),
                    "components": comps, "missing": missing}
    return out


def score_history(data: dict, key: str, freq: str = "W-FRI",
                  min_window: int = 60) -> pd.Series:
    """ประวัติคะแนนโมเดล (expanding percentile — ไม่ใช้ข้อมูลอนาคต) รายสัปดาห์.
    ใช้ดู 'โมเดลขยับกี่จุด' สำหรับ trigger ห้องประชุม."""
    spec = MODEL_DEFS[key]
    cols = {}
    for name, fn, invert in spec["components"]:
        try:
            s = fn(data)
        except Exception:
            s = None
        if s is None or len(pd.Series(s).dropna()) < min_window:
            continue
        s = pd.Series(s).dropna().resample(freq).last().dropna()
        r = s.expanding(min_periods=min_window // 5).apply(
            lambda w: (w <= w[-1]).mean() * 100.0, raw=True)
        cols[name] = (100.0 - r) if invert else r
    if not cols:
        return pd.Series(dtype=float)
    return pd.DataFrame(cols).mean(axis=1).dropna()


def model_delta(hist: pd.Series, lookback: int = 1) -> float:
    """การเปลี่ยนแปลงคะแนนล่าสุดเทียบ lookback งวดก่อน (จุด)."""
    h = hist.dropna()
    if len(h) <= lookback:
        return float("nan")
    return float(h.iloc[-1] - h.iloc[-1 - lookback])
