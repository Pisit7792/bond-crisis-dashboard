"""
data_sources.py — ชั้นดึงข้อมูล (แยกจากการคำนวณ)

แหล่งข้อมูลจริง (ฟรี):
- FRED REST API (ต้องมี API key ฟรี: https://fred.stlouisfed.org/docs/api/api_key.html)
- yfinance สำหรับ ^MOVE และราคาสินทรัพย์

โหมด DEMO: สร้างข้อมูลสังเคราะห์ที่ 'ติดป้ายชัดเจนว่าไม่ใช่ข้อมูลจริง'
มีไว้เพื่อดู UI/ทดสอบตรรกะเท่านั้น — ห้ามใช้ตัดสินใจลงทุน
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# series ที่ใช้ (ทั้งหมดเป็นข้อมูลสาธารณะของ Fed/BofA ผ่าน FRED)
FRED_SERIES = {
    # yield curve
    "DGS3MO": "US Treasury 3M",
    "DGS2": "US Treasury 2Y",
    "DGS10": "US Treasury 10Y",
    "DGS30": "US Treasury 30Y",
    "T10Y3M": "Spread 10Y-3M",
    "T10Y2Y": "Spread 10Y-2Y",
    # stress / credit / vol
    "STLFSI4": "St. Louis Fed Financial Stress Index",
    "NFCI": "Chicago Fed National Financial Conditions Index",
    "BAMLH0A0HYM2": "ICE BofA US High Yield OAS",
    "BAMLC0A0CM": "ICE BofA US Corporate (IG) OAS",
    "VIXCLS": "CBOE VIX",
}

CURVE_SNAPSHOT = ["DGS3MO", "DGS2", "DGS10", "DGS30"]
CURVE_TENORS_Y = {"DGS3MO": 0.25, "DGS2": 2, "DGS10": 10, "DGS30": 30}

# สินทรัพย์ในตาราง Trend State (ตามที่ผู้ใช้ติดตาม + พันธบัตร)
YF_ASSETS = {
    "^NDX": "NAS100", "^DJI": "US30", "ETH-USD": "ETH", "SOL-USD": "SOL",
    "CL=F": "USOIL", "EURUSD=X": "EURUSD", "TLT": "TLT (20y+ UST ETF)",
}
YF_MOVE = "^MOVE"


def fetch_fred_series(series_id: str, api_key: str, start: str = "1990-01-01",
                      timeout: int = 20) -> pd.Series:
    """ดึงหนึ่ง series จาก FRED REST API -> pd.Series(index=date, float)."""
    params = {
        "series_id": series_id, "api_key": api_key, "file_type": "json",
        "observation_start": start,
    }
    r = requests.get(FRED_BASE, params=params, timeout=timeout)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    if not obs:
        return pd.Series(dtype=float, name=series_id)
    df = pd.DataFrame(obs)
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values,
                  index=pd.to_datetime(df["date"]), name=series_id)
    return s.dropna()


def fetch_all_fred(api_key: str, start: str = "1990-01-01") -> dict[str, pd.Series]:
    """ดึงทุก series ใน FRED_SERIES; series ที่พังจะข้ามพร้อมเก็บ error."""
    out, errors = {}, {}
    for sid in FRED_SERIES:
        try:
            out[sid] = fetch_fred_series(sid, api_key, start)
        except Exception as e:  # แสดง error ตรงๆ ใน UI ไม่กลบ
            errors[sid] = str(e)
    out["_errors"] = errors
    return out


def fetch_yf_history(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    """ดึงราคาปิดจาก yfinance -> DataFrame คอลัมน์ = ticker.
    import ภายในฟังก์ชันเพื่อให้แอปรันได้แม้ไม่ได้ติดตั้ง yfinance (โหมด demo)."""
    import yfinance as yf  # optional dependency
    data = yf.download(tickers, period=period, interval="1d",
                       auto_adjust=True, progress=False)
    close = data["Close"] if "Close" in data else data
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return close.dropna(how="all")


# ---------------------------------------------------------------------------
# DEMO DATA — สังเคราะห์ ติดป้ายชัดเจน
# ---------------------------------------------------------------------------

def demo_bundle(seed: int = 11) -> dict:
    """สร้างชุดข้อมูลสังเคราะห์ครบทุก series ที่แอปใช้ + ธง is_demo=True.

    ออกแบบให้มี 'ช่วงเครียด' หนึ่งช่วงเพื่อให้เห็นพฤติกรรมของทุกโมดูล
    (curve inversion, stress พุ่ง, HY ถ่าง, MOVE พุ่ง, สินทรัพย์เสี่ยงหลุด 200DMA)
    ตัวเลขไม่มีความหมายทางเศรษฐกิจ — เพื่อทดสอบ UI เท่านั้น
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=252 * 8)
    n = len(days)
    stress_start, stress_end = int(n * 0.62), int(n * 0.70)

    def rw(level, vol, drift=0.0):
        x = level + np.cumsum(rng.normal(drift, vol, n))
        return pd.Series(x, index=days)

    # yields: ปกติ curve ชัน; ช่วงเครียดให้ 3M พุ่งจน invert
    y3m = rw(2.0, 0.015).clip(0.05, None)
    y10 = rw(3.2, 0.02).clip(0.3, None)
    bump = np.zeros(n); bump[stress_start:stress_end] = np.linspace(0, 1.6, stress_end - stress_start)
    decay = np.zeros(n); decay[stress_end:] = np.linspace(1.6, 0.4, n - stress_end)
    y3m = (y3m + bump + decay).clip(0.05, None)
    y2 = (y3m * 0.5 + y10 * 0.5) + rng.normal(0, 0.03, n)
    y30 = y10 + 0.4 + rng.normal(0, 0.03, n)
    t10y3m = y10 - y3m
    t10y2y = y10 - y2

    def stress_like(base, calm_vol, spike):
        x = rw(base, calm_vol)
        add = np.zeros(n); add[stress_start:stress_end] = np.linspace(0, spike, stress_end - stress_start)
        add[stress_end:] = np.linspace(spike, spike * 0.2, n - stress_end)
        return (x + add)

    stlfsi = stress_like(-0.4, 0.01, 3.0)
    nfci = stress_like(-0.5, 0.008, 1.5)
    hy = stress_like(3.5, 0.02, 4.5).clip(2.0, None)
    ig = stress_like(1.2, 0.008, 1.5).clip(0.6, None)
    vix = stress_like(16, 0.15, 30).clip(9, None)
    move = stress_like(90, 0.8, 110).clip(40, None)

    fred = {"DGS3MO": y3m, "DGS2": pd.Series(y2, index=days), "DGS10": y10,
            "DGS30": pd.Series(y30, index=days), "T10Y3M": t10y3m,
            "T10Y2Y": t10y2y, "STLFSI4": stlfsi.iloc[::5],  # weekly-ish
            "NFCI": nfci.iloc[::5], "BAMLH0A0HYM2": hy, "BAMLC0A0CM": ig,
            "VIXCLS": vix, "_errors": {}}

    # asset prices: เทรนด์ขึ้น แล้ว drawdown ช่วงเครียด
    prices = {}
    for tkr in list(YF_ASSETS) :
        base = rng.uniform(50, 20000)
        ret = rng.normal(0.0004, 0.012, n)
        ret[stress_start:stress_end] -= 0.004
        prices[tkr] = pd.Series(base * np.exp(np.cumsum(ret)), index=days)
    px = pd.DataFrame(prices)

    return {"fred": fred, "prices": px, "move": move, "is_demo": True}
