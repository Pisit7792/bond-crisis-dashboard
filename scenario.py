"""
scenario.py — จำลองสถานการณ์ (what-if) ต่อคะแนน 6 โมเดล

ความซื่อสัตย์ของโมดูลนี้ (สำคัญที่สุด):
- ค่าความไว (sensitivity) ทั้งหมดเป็น "สมมติฐานการออกแบบ" ที่ตั้งจากทิศทาง
  ความสัมพันธ์ในประวัติศาสตร์ — *ไม่ใช่* ค่าสัมประสิทธิ์ที่ประมาณจากข้อมูลจริง
- เมทริกซ์ทั้งหมดเปิดดูได้บนหน้าจอ (ไม่มีกล่องดำ) และผลลัพธ์เป็น
  "ค่าประมาณทิศทาง ไม่ใช่ผลคำนวณเต็มรูปแบบ"
- ทิศทางอิงเหตุการณ์จริง เช่น เงินฝากไหลออก→แบงก์รัน (SVB 2023),
  ประมูล bid-to-cover อ่อน→yield ตึง, SOFR-EFFR ถ่าง→repo ตึง (ก.ย. 2019)
"""
from __future__ import annotations

import pandas as pd

# (key, ป้ายไทย, หน่วย, min, max, step, default)
SLIDERS: list[tuple] = [
    ("fed_bps", "Fed ขึ้น/ลดดอกเบี้ย", "bps", -100, 100, 5, 0),
    ("oil_pct", "ราคาน้ำมันเปลี่ยน", "%", -30, 30, 1, 0),
    ("gold_pct", "ราคาทองคำเปลี่ยน", "%", -20, 20, 1, 0),
    ("vix_pts", "VIX เปลี่ยน", "pts", -15, 40, 1, 0),
    ("hy_bps", "HY Spread เปลี่ยน", "bps", -150, 400, 10, 0),
    ("cpi_pt", "เงินเฟ้อ CPI เปลี่ยน", "pt", -1.0, 2.0, 0.1, 0.0),
    ("depo_pct", "เงินฝากแบงก์ (2 สัปดาห์)", "%", -10, 3, 0.5, 0.0),
    ("dw_bn", "Fed Discount Window พุ่ง", "$B", 0, 300, 10, 0),
    ("repo_bps", "SOFR-EFFR spread (repo ตึง)", "bps", 0, 100, 5, 0),
    ("debt_pt", "หนี้สหรัฐต่อ GDP เพิ่ม", "pt", 0, 20, 1, 0),
    ("btc_x", "ประมูล 10Y Bid-to-Cover", "x", 1.5, 3.5, 0.1, 2.5),
]

SLIDER_BASE = {k: d for k, _, _, _, _, _, d in SLIDERS}

# จุดที่เปลี่ยนต่อ 1 หน่วยของ slider (บวก = ดันคะแนนโมเดลขึ้น)
# แถว = โมเดล, คอลัมน์ = slider; ตัวเลขคือ "สมมติฐานการออกแบบ" เปิดแก้ได้
SENSITIVITY: dict[str, dict[str, float]] = {
    "inflation_oil": {"oil_pct": 0.6, "cpi_pt": 15.0, "gold_pct": 0.15,
                      "fed_bps": -0.02},
    "yield_shock":   {"fed_bps": 0.08, "vix_pts": 0.30, "hy_bps": 0.02,
                      "cpi_pt": 8.0, "debt_pt": 0.5, "btc_x": -15.0},
    "recovery":      {"oil_pct": -0.25, "vix_pts": -0.50, "hy_bps": -0.03,
                      "fed_bps": -0.03, "depo_pct": 0.8, "cpi_pt": -5.0},
    "fed_pivot":     {"fed_bps": -0.15, "vix_pts": 0.20, "cpi_pt": -10.0,
                      "hy_bps": 0.01, "depo_pct": -0.5},
    "credit_crisis": {"hy_bps": 0.08, "vix_pts": 0.40, "repo_bps": 0.15,
                      "oil_pct": 0.10, "fed_bps": 0.02},
    "bank_run":      {"depo_pct": -3.5, "dw_bn": 0.12, "repo_bps": 0.25,
                      "vix_pts": 0.15, "gold_pct": 0.10},
}


def apply_scenario(base_scores: dict[str, float],
                   values: dict[str, float]) -> dict[str, dict]:
    """คำนวณคะแนนใหม่ต่อโมเดลจากค่า slider.

    delta ของ slider = ค่า - ค่าตั้งต้น (ส่วนใหญ่ 0; bid-to-cover ตั้งต้น 2.5)
    คืน {model: {"base","delta","new"}} โดย new ถูก cap 0-100
    """
    out = {}
    for m, sens in SENSITIVITY.items():
        base = base_scores.get(m)
        base = 0.0 if base is None or pd.isna(base) else float(base)
        delta = 0.0
        for k, per_unit in sens.items():
            delta += per_unit * (float(values.get(k, SLIDER_BASE[k])) - SLIDER_BASE[k])
        new = min(100.0, max(0.0, base + delta))
        out[m] = {"base": round(base, 1), "delta": round(new - base, 1),
                  "new": round(new, 1)}
    return out


def sensitivity_table() -> pd.DataFrame:
    """เมทริกซ์ความไวแบบตาราง (จุดต่อ 1 หน่วย) — โชว์บนหน้าจอเพื่อความโปร่งใส"""
    labels = {k: f"{th} ({u})" for k, th, u, *_ in SLIDERS}
    df = pd.DataFrame(SENSITIVITY).T.fillna(0.0)
    df.columns = [labels.get(c, c) for c in df.columns]
    return df
