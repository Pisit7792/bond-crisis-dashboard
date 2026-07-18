"""
signals.py — สัญญาณเทรดแบบ rule-based (โปร่งใส ทดสอบได้)

กติกาที่ตั้งไว้ล่วงหน้า (pre-committed, ตามสเปกผู้ใช้ + ชั้นความซื่อสัตย์):
1. เกิดสัญญาณเมื่อ 'โมเดลถึง 40 จุด' → ดูสินทรัพย์ในตารางได้/เสียประโยชน์
2. ทิศทางต้องผ่าน trend filter: LONG เฉพาะราคาเหนือ 200DMA,
   SHORT เฉพาะราคาใต้ 200DMA (ไม่สวนแนวโน้มใหญ่)
3. SL = 2×ATR, TP = 4×ATR → R:R 1:2 เสมอ — เป็น 'ธรรมเนียมการวางระดับตาม
   ความผันผวน' ไม่ใช่คำพยากรณ์ว่าราคาจะไปถึง
4. สินทรัพย์เดียวโดนหลายโมเดล: ทางเดียวกัน → เก็บตัวคะแนนสูงสุด /
   ขัดแย้งกัน (LONG ปะทะ SHORT) → ตัดทิ้งและรายงานว่าขัดแย้ง (ไม่เลือกข้างเอง)
5. เทคนิคอลที่กำกับ (RSI, โมเมนตัม, ระยะจาก 200DMA) เป็น 'คำอธิบายสภาพราคา'
   ไม่ใช่การพยากรณ์ และไม่มีการอ้าง win rate ใดๆ จนกว่า journal จะครบ 100 เทรด

หมายเหตุ ATR: คำนวณจากราคาปิด (ค่าเฉลี่ย |Δclose| 14 วัน) เพราะข้อมูลที่ดึงมี
เฉพาะราคาปิด — เป็นค่าประมาณของ ATR จริง (ที่ใช้ high/low) และติดป้ายไว้เช่นนั้น
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

SIGNAL_THRESHOLD = 40.0
SL_ATR = 2.0
TP_ATR = 4.0

DISCLAIMER = ("สัญญาณเป็นผลของกฎที่ประกาศไว้ ไม่ใช่คำแนะนำการลงทุน — "
              "ยังไม่มีหลักฐานประสิทธิภาพของกฎชุดนี้ (journal ต้องครบ ≥100 เทรดก่อน) "
              "และหลักฐานวิชาการชี้ว่า <3-5% ของรายย่อยทำกำไรจากสัญญาณได้สม่ำเสมอ")


def rsi(close: pd.Series, n: int = 14) -> float:
    c = pd.Series(close).dropna()
    if len(c) < n + 1:
        return float("nan")
    d = c.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return float(out.iloc[-1])


def atr_approx(close: pd.Series, n: int = 14) -> float:
    """ATR โดยประมาณจากราคาปิด: ค่าเฉลี่ย |การเปลี่ยนแปลงรายวัน| n วัน."""
    c = pd.Series(close).dropna()
    if len(c) < n + 1:
        return float("nan")
    return float(c.diff().abs().rolling(n).mean().iloc[-1])


def build_signals(model_scores: dict[str, dict], asset_impact: dict[str, dict],
                  prices: pd.DataFrame, ticker_names: dict[str, str],
                  threshold: float = SIGNAL_THRESHOLD) -> dict:
    """สร้างสัญญาณจากโมเดล >= threshold.

    คืน {"signals": [dict...], "conflicts": [dict...], "skipped": [dict...]}
    - signals: ผ่านทุกกติกา (พร้อม entry/SL/TP/เทคนิคอล)
    - conflicts: สินทรัพย์ที่โมเดลให้ทิศตรงข้ามกัน → ไม่ออกสัญญาณ
    - skipped: เข้าเกณฑ์โมเดลแต่ไม่ผ่าน trend filter (บอกเหตุผลตรงๆ)
    """
    name_to_ticker = {v: k for k, v in ticker_names.items()}
    proposals: dict[str, list[dict]] = {}

    for mkey, m in model_scores.items():
        sc = m.get("score")
        if sc is None or (isinstance(sc, float) and math.isnan(sc)) or sc < threshold:
            continue
        imp = asset_impact.get(mkey, {})
        for side, names in (("LONG", imp.get("benefit", [])),
                            ("SHORT", imp.get("lose", []))):
            for disp in names:
                tkr = name_to_ticker.get(disp)
                if tkr is None or tkr not in prices.columns:
                    continue
                proposals.setdefault(disp, []).append(
                    {"model": m["th"], "model_key": mkey, "score": sc,
                     "side": side, "ticker": tkr})

    signals, conflicts, skipped = [], [], []
    for disp, props in proposals.items():
        sides = {p["side"] for p in props}
        if len(sides) > 1:
            conflicts.append({
                "asset": disp,
                "รายละเอียด": " ปะทะ ".join(
                    f"{p['model']} ({p['side']} {p['score']:.0f})" for p in props),
                "การตัดสิน": "ไม่ออกสัญญาณ — โมเดลขัดแย้งกัน (ระบบไม่เลือกข้างเอง)"})
            continue
        best = max(props, key=lambda p: p["score"])
        c = prices[best["ticker"]].dropna()
        if len(c) < 260:
            skipped.append({"asset": disp, "เหตุผล": "ประวัติราคาสั้นเกิน"})
            continue
        last = float(c.iloc[-1])
        ma200 = float(c.rolling(200).mean().iloc[-1])
        trend_up = last >= ma200
        if (best["side"] == "LONG" and not trend_up) or \
           (best["side"] == "SHORT" and trend_up):
            skipped.append({
                "asset": disp,
                "เหตุผล": (f"{best['model']} เสนอ {best['side']} แต่ราคา"
                            f"{'ใต้' if best['side']=='LONG' else 'เหนือ'} 200DMA "
                            "— ไม่สวนแนวโน้มใหญ่ (trend filter)")})
            continue
        a = atr_approx(c)
        if math.isnan(a) or a <= 0:
            skipped.append({"asset": disp, "เหตุผล": "คำนวณ ATR ไม่ได้"})
            continue
        sl = last - SL_ATR * a if best["side"] == "LONG" else last + SL_ATR * a
        tp = last + TP_ATR * a if best["side"] == "LONG" else last - TP_ATR * a
        mom = float(c.pct_change(252).iloc[-1] * 100) if len(c) > 252 else float("nan")
        signals.append({
            "asset": disp, "ticker": best["ticker"], "side": best["side"],
            "model": best["model"], "strength": round(best["score"], 0),
            "entry": round(last, 4), "sl": round(sl, 4), "tp": round(tp, 4),
            "rr": "1:2",
            "atr14≈": round(a, 4),
            "rsi14": round(rsi(c), 0),
            "mom12m%": round(mom, 1) if not math.isnan(mom) else None,
            "dist_200dma%": round((last / ma200 - 1) * 100, 1),
            "trend": "ขาขึ้น (เหนือ 200DMA)" if trend_up else "ขาลง (ใต้ 200DMA)",
        })

    signals.sort(key=lambda s: -s["strength"])
    return {"signals": signals, "conflicts": conflicts, "skipped": skipped}


def journal_csv(signals: list[dict], as_of: str) -> str:
    """สร้าง CSV สำหรับบันทึกสัญญาณ (signal journal) — เพื่อสะสมหลักฐานให้ครบ
    100 เทรดก่อนตัดสินระบบ คอลัมน์ pnl เว้นว่างให้กรอกเมื่อปิดออเดอร์"""
    rows = ["date,asset,side,model,strength,entry,sl,tp,pnl,note"]
    for s in signals:
        rows.append(f"{as_of},{s['asset']},{s['side']},{s['model']},"
                    f"{s['strength']:.0f},{s['entry']},{s['sl']},{s['tp']},,")
    return "\n".join(rows) + "\n"
