"""
engine.py — Bond Crisis Dashboard: computation core
ฟังก์ชันคำนวณล้วน (pure functions) แยกจาก UI/ข้อมูล เพื่อให้ทดสอบได้

หลักการอ้างอิง (ดูรายละเอียด/ข้อจำกัดใน README):
- Recession probit: Estrella & Mishkin (1996, NY Fed); Estrella & Trubin (2006)
- Regime: Hamilton (1989) — 2-state Gaussian HMM (Baum-Welch)
- Forecast combination: Bates & Granger (1969); equal weights per Clemen (1989)
- Trade stats: Wilson (1927) CI; Bailey & Lopez de Prado (2012) Probabilistic Sharpe
หมายเหตุความซื่อสัตย์: ทุกคะแนนในไฟล์นี้เป็น "เครื่องมือวินิจฉัย" (diagnostic)
ไม่ใช่เครื่องทำนายกำไร — วรรณกรรม EWS ชี้ false alarm สูง (P(crisis|alarm)~50%)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# 1) Yield curve -> recession probability (probit)
# ---------------------------------------------------------------------------
# สัมประสิทธิ์ fit กับตารางความน่าจะเป็นที่ NY Fed เผยแพร่ (Estrella & Mishkin 1996)
# ตาราง: spread 1.21%->5%, 0.76->10, 0.46->15, 0.22->20, 0.02->25, -0.17->30,
#        -0.50->40, -0.82->50, -1.13->60, -1.46->70, -1.85->80, -2.40->90
# max abs error ของ fit = 0.13 percentage point (ตรวจสอบแล้ว)
PROBIT_A = -0.6624
PROBIT_B = -0.8115


def recession_probability(spread_10y_3m: float) -> float:
    """P(recession ภายใน 12 เดือน) จาก spread 10y-3m (หน่วย %).

    ข้อจำกัดที่ต้องแสดงคู่กันเสมอ: lead time ผันแปร 6-24 เดือน,
    มี false signals ในประวัติศาสตร์, และทำนาย recession ไม่ใช่ bond
    liquidity crisis (Taper 2013 / มี.ค. 2020 / Gilt 2022 ไม่ได้ถูกเตือนด้วยตัวนี้)
    """
    if spread_10y_3m is None or (isinstance(spread_10y_3m, float) and math.isnan(spread_10y_3m)):
        return float("nan")
    return float(norm.cdf(PROBIT_A + PROBIT_B * spread_10y_3m))


# ---------------------------------------------------------------------------
# 2) Percentile / z-score helpers (ใช้ทำ sub-score 0-100)
# ---------------------------------------------------------------------------

def percentile_of_latest(series: pd.Series) -> float:
    """เปอร์เซ็นไทล์ (0-100) ของค่าล่าสุดเทียบประวัติทั้งหมดของ series เอง."""
    s = pd.Series(series).dropna()
    if len(s) < 30:
        return float("nan")
    latest = s.iloc[-1]
    return float((s <= latest).mean() * 100.0)


def zscore_of_latest(series: pd.Series) -> float:
    s = pd.Series(series).dropna()
    if len(s) < 30 or s.std(ddof=0) == 0:
        return float("nan")
    return float((s.iloc[-1] - s.mean()) / s.std(ddof=0))


def percentile_history(series: pd.Series, min_window: int = 260) -> pd.Series:
    """เปอร์เซ็นไทล์แบบ expanding (ไม่ใช้ข้อมูลอนาคต) — ป้องกัน look-ahead bias."""
    s = pd.Series(series).dropna()
    out = s.expanding(min_periods=min_window).apply(
        lambda w: (w <= w[-1]).mean() * 100.0, raw=True
    )
    return out


# ---------------------------------------------------------------------------
# 3) Composite crisis score (5 sub-models, equal weights)
# ---------------------------------------------------------------------------

SUBMODEL_LABELS = {
    "curve": "1. Yield Curve (probit recession prob.) — leading, lead time 6-24 เดือน",
    "stress": "2. Financial Stress Index percentile — coincident",
    "credit": "3. HY Credit Spread percentile — coincident/นำเล็กน้อย",
    "vol": "4. Rates/Equity Volatility percentile (MOVE/VIX) — coincident",
    "breadth": "5. Risk-off Breadth (% สินทรัพย์เสี่ยงใต้ 200DMA) — trend context",
}


def composite_crisis_score(subscores: dict[str, float]) -> tuple[float, dict[str, float]]:
    """รวม 5 sub-score (0-100) ด้วย equal weights.

    ทำไม equal weights: Bates-Granger (1969) เริ่มวรรณกรรม forecast combination;
    Clemen (1989) และ 'forecast combination puzzle' ชี้ว่า simple average
    มักชนะ optimal weights เพราะ error ในการประมาณ covariance สูง.
    คืน (score, subscores_ที่ใช้จริง) — ข้าม NaN และเฉลี่ยเฉพาะตัวที่มีข้อมูล
    """
    used = {k: v for k, v in subscores.items() if v is not None and not math.isnan(v)}
    if not used:
        return float("nan"), {}
    score = float(np.mean(list(used.values())))
    return score, used


def submodel_correlation(sub_hist: pd.DataFrame) -> pd.DataFrame:
    """สหสัมพันธ์ระหว่างประวัติ sub-score — ถ้า >0.8 คู่ใด แปลว่า ensemble
    ไม่ได้ diversity จริง (การรวมไม่เพิ่มคุณค่า) ตามเงื่อนไขของ forecast
    combination ที่ต้องการ errors ไม่สหสัมพันธ์กัน."""
    return sub_hist.dropna().corr()


# ---------------------------------------------------------------------------
# 4) Regime detection — 2-state Gaussian HMM (Hamilton 1989 แบบย่อ)
# ---------------------------------------------------------------------------

@dataclass
class HMMResult:
    mu: np.ndarray            # ค่าเฉลี่ยแต่ละ regime
    sigma: np.ndarray         # ส่วนเบี่ยงเบนมาตรฐานแต่ละ regime
    trans: np.ndarray         # transition matrix 2x2
    smoothed: np.ndarray      # P(regime=k | ข้อมูลทั้งหมด) — in-sample
    filtered: np.ndarray      # P(regime=k | ข้อมูลถึงเวลา t) — real-time
    loglik: float
    converged: bool
    high_vol_state: int       # index ของ regime ความผันผวนสูง


def fit_hmm_2state(x: np.ndarray, max_iter: int = 200, tol: float = 1e-6,
                   seed: int = 7) -> HMMResult:
    """Baum-Welch EM สำหรับ 2-state Gaussian HMM.

    เหตุผลที่ implement เอง: ไม่พึ่ง dependency หนัก และควบคุม failure mode ได้.
    คำเตือนตามวรรณกรรม: regime 'มองเห็นง่ายหลังเกิด (smoothed) แต่ยากใน
    real time (filtered)' — UI ต้องแสดง filtered คู่ smoothed เสมอ
    เพื่อให้เห็น detection lag ตามจริง ไม่หลอกตัวเอง.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 60:
        raise ValueError("ต้องการข้อมูลอย่างน้อย 60 จุดสำหรับ HMM")

    rng = np.random.default_rng(seed)
    # init: แบ่งด้วย median ของ |x - mean| เพื่อเดา low/high vol
    mu = np.array([np.mean(x), np.mean(x)])
    s = np.std(x)
    sigma = np.array([0.5 * s + 1e-8, 1.5 * s + 1e-8])
    trans = np.array([[0.95, 0.05], [0.05, 0.95]])
    pi = np.array([0.5, 0.5])

    def emissions(mu, sigma):
        # (n,2) ความหนาแน่นปกติ, กัน underflow ด้วย floor
        e = np.column_stack([
            norm.pdf(x, mu[k], max(sigma[k], 1e-8)) for k in range(2)
        ])
        return np.clip(e, 1e-300, None)

    prev_ll = -np.inf
    converged = False
    for _ in range(max_iter):
        e = emissions(mu, sigma)
        # forward (scaled)
        alpha = np.zeros((n, 2)); c = np.zeros(n)
        alpha[0] = pi * e[0]; c[0] = alpha[0].sum(); alpha[0] /= c[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ trans) * e[t]
            c[t] = alpha[t].sum(); alpha[t] /= c[t]
        # backward (scaled)
        beta = np.zeros((n, 2)); beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = (trans @ (e[t + 1] * beta[t + 1])) / c[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        # xi: expected transitions
        xi_num = np.zeros((2, 2))
        for t in range(n - 1):
            m = (alpha[t][:, None] * trans) * (e[t + 1] * beta[t + 1])[None, :]
            xi_num += m / m.sum()
        # M-step
        pi = gamma[0]
        trans = xi_num / xi_num.sum(axis=1, keepdims=True)
        for k in range(2):
            w = gamma[:, k]
            mu[k] = np.sum(w * x) / w.sum()
            sigma[k] = math.sqrt(max(np.sum(w * (x - mu[k]) ** 2) / w.sum(), 1e-10))
        ll = float(np.sum(np.log(c)))
        if abs(ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll

    # filtered probabilities (real-time view)
    e = emissions(mu, sigma)
    filt = np.zeros((n, 2))
    filt[0] = pi * e[0]; filt[0] /= filt[0].sum()
    for t in range(1, n):
        filt[t] = (filt[t - 1] @ trans) * e[t]
        filt[t] /= filt[t].sum()

    high = int(np.argmax(sigma))
    return HMMResult(mu=mu, sigma=sigma, trans=trans, smoothed=gamma,
                     filtered=filt, loglik=prev_ll, converged=converged,
                     high_vol_state=high)


# ---------------------------------------------------------------------------
# 5) Trade statistics — ตอบคำถาม "6-10 เทรดบอกอะไรได้บ้าง" อย่างซื่อสัตย์
# ---------------------------------------------------------------------------

def wilson_ci(wins: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval สำหรับ win rate — ดีกว่า normal approx ที่ n เล็ก."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(1 - (1 - conf) / 2)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def required_n(moe: float = 0.05, conf: float = 0.95, p: float = 0.5) -> int:
    """จำนวนเทรดขั้นต่ำให้ margin of error ของ win rate <= moe."""
    z = norm.ppf(1 - (1 - conf) / 2)
    return int(math.ceil((z * z * p * (1 - p)) / (moe * moe)))


def profit_factor(pnl: pd.Series) -> float:
    """PF = gross profit / gross loss. ไวต่อ outlier มาก — เชื่อถือได้ที่
    ~100-200+ เทรด. PF<1 = ขาดทุนจริงในช่วงที่วัด (แต่ n เล็กยังสรุป edge ไม่ได้)."""
    p = pd.Series(pnl).dropna()
    gp = p[p > 0].sum(); gl = -p[p < 0].sum()
    if gl == 0:
        return float("inf") if gp > 0 else float("nan")
    return float(gp / gl)


def expectancy(pnl: pd.Series) -> float:
    p = pd.Series(pnl).dropna()
    return float(p.mean()) if len(p) else float("nan")


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR (Bailey & Lopez de Prado 2012): P(true SR > benchmark) ปรับ skew/kurtosis.
    หมายเหตุ: Deflated SR (2014) ต้องรู้ 'จำนวน trials ที่ทดสอบทั้งหมด' —
    ถ้าผู้ใช้ลองหลายกลยุทธ์แล้วเลือกตัวดีสุด PSR นี้ยัง 'มองโลกดีเกินจริง'."""
    r = pd.Series(returns).dropna().astype(float)
    n = len(r)
    if n < 10 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    g3 = float(pd.Series(r).skew())
    g4 = float(pd.Series(r).kurt()) + 3.0  # pandas ให้ excess kurtosis
    denom = math.sqrt(max(1 - g3 * sr + ((g4 - 1) / 4.0) * sr * sr, 1e-12))
    stat = (sr - sr_benchmark) * math.sqrt(n - 1) / denom
    return float(norm.cdf(stat))


def trade_log_report(df: pd.DataFrame) -> dict:
    """สรุปสถิติจาก trade log (ต้องมีคอลัมน์ 'pnl').

    คืน dict พร้อม 'verdict' ภาษาไทยที่ตรงไปตรงมาเรื่องขนาดตัวอย่าง:
    n<30: noise ล้วน / 30-99: เริ่มเห็นเค้าลาง / >=100: เริ่มมีน้ำหนักทางสถิติ
    """
    pnl = pd.to_numeric(df["pnl"], errors="coerce").dropna()
    n = int(len(pnl))
    wins = int((pnl > 0).sum())
    wr = wins / n if n else float("nan")
    lo, hi = wilson_ci(wins, n)
    pf = profit_factor(pnl)
    ex = expectancy(pnl)
    psr = probabilistic_sharpe_ratio(pnl)
    if n == 0:
        verdict = "ไม่มีข้อมูล"
    elif n < 30:
        verdict = (f"n={n} เล็กเกินกว่าจะสรุปอะไรได้ — ช่วงความเชื่อมั่น win rate "
                   f"กว้างถึง {lo:.0%}-{hi:.0%} (แยก edge จากเหรียญโยนไม่ได้) "
                   f"ต้องการอย่างน้อย ~{required_n():d} เทรด")
    elif n < 100:
        verdict = (f"n={n} เริ่มเห็นเค้าลางแต่ยังไม่พอ — CI win rate {lo:.0%}-{hi:.0%}; "
                   f"PF ที่ n ระดับนี้ยังไวต่อ outlier มาก")
    else:
        verdict = (f"n={n} เริ่มมีน้ำหนักทางสถิติ — CI win rate {lo:.0%}-{hi:.0%}. "
                   "อย่าลืม: ถ้าเลือกกลยุทธ์นี้จากการลองหลายตัว ค่าที่เห็น inflated "
                   "(ต้อง deflate ตามจำนวน trials — Bailey & Lopez de Prado 2014)")
    return {"n": n, "wins": wins, "win_rate": wr, "ci_low": lo, "ci_high": hi,
            "profit_factor": pf, "expectancy": ex, "psr": psr, "verdict": verdict}


# ---------------------------------------------------------------------------
# 6) Scenario math — duration/convexity approximation
# ---------------------------------------------------------------------------

def bond_price_impact(duration: float, convexity: float, dy_bp: float) -> float:
    """ผลกระทบราคาโดยประมาณ (%) จาก yield shift dy_bp (basis points).
    dP/P ~ -D*dy + 0.5*C*dy^2 — เป็น local approximation; shift ใหญ่/curve
    ไม่ขนานจะคลาดเคลื่อน. ใช้เพื่อ scenario range ไม่ใช่ point forecast."""
    dy = dy_bp / 10000.0
    return float((-duration * dy + 0.5 * convexity * dy * dy) * 100.0)


# กรณีศึกษาจริง (อ้างอิงรายงานวิจัยรอบก่อน) — ใช้เป็น reference scenarios
HISTORICAL_EPISODES = [
    {"name": "Taper Tantrum 2013", "move": "US 10y +~100bp ใน ~4 เดือน (2.03%→2.96%)",
     "precursor": "carry-trade positioning แออัด (Fed FEDS Note 2023)",
     "lesson": "สัญญาณนำคือ positioning ไม่ใช่ราคา"},
    {"name": "COVID Dash-for-Cash มี.ค. 2020",
     "move": "UST market depth -93% จากค่าเฉลี่ย ก.พ.; Treasuries ร่วงพร้อมหุ้น",
     "precursor": "basis-trade leverage >$1tn; hedge funds ขาย >$200bn",
     "lesson": "flight-to-safety ล้มเหลวได้เมื่อ leverage ต้อง unwind"},
    {"name": "UK Gilt/LDI ก.ย. 2022", "move": "30y gilt +>100bp ใน 4 วัน (2-5x สถิติเดิม)",
     "precursor": "LDI leverage/duration mismatch; ขายบังคับ ~GBP25bn",
     "lesson": "margin spiral ใน NBFI — ดู leverage ก่อนราคา"},
    {"name": "SVB มี.ค. 2023", "move": "แบงก์ล้มใน ~48 ชม.; ถอน $42bn/วัน",
     "precursor": "HTM unrealized loss ~$15bn ~ equity; uninsured deposits ~94%",
     "lesson": "duration mismatch อยู่ใน footnotes ก่อนวิกฤต"},
]


# ---------------------------------------------------------------------------
# 7) Tiered risk warnings — จำกัดจำนวนตามงาน alert fatigue (Ancker 2017)
# ---------------------------------------------------------------------------

def build_warnings(composite: float, spread_10y3m: float, stress_pct: float,
                   hy_z: float, move_pct: float,
                   inverted_days: int = 0) -> list[dict]:
    """คืนรายการเตือนแบบ tiered สูงสุด 3 รายการ (เรียงตามความรุนแรง).
    ทุกเตือนมี false-alarm framing เพราะ EWS literature:
    P(crisis|alarm) ~ 50% (Bussiere-Fratzscher, ECB WP 145)."""
    W = []
    def _n(x):
        return x is not None and not (isinstance(x, float) and math.isnan(x))
    if _n(composite) and composite >= 70:
        W.append({"tier": 1, "msg": f"Composite score {composite:.0f}/100 อยู่โซนสูง — "
                  "ตรวจ leverage/สภาพคล่องพอร์ต; คาด false alarm ได้ ~ครึ่งหนึ่งของการเตือน"})
    if _n(spread_10y3m) and spread_10y3m < 0 and _n(stress_pct) and stress_pct >= 90:
        W.append({"tier": 1, "msg": "Curve inverted พร้อม stress percentile >=90 พร้อมกัน — "
                  "สภาวะที่เคยเกิดร่วมช่วงก่อน/ระหว่างวิกฤตในอดีต (ไม่ใช่คำทำนาย)"})
    if _n(spread_10y3m) and spread_10y3m < 0 and inverted_days >= 20:
        W.append({"tier": 2, "msg": f"10y-3m inverted ต่อเนื่อง {inverted_days} วันทำการ — "
                  "โมเดล probit ชี้ความเสี่ยง recession 12 เดือนสูงขึ้น (lead 6-24 เดือน, มี false signals)"})
    if _n(hy_z) and hy_z >= 2:
        W.append({"tier": 2, "msg": f"HY spread z-score {hy_z:.1f} (>=2) — credit stress ผิดปกติ"})
    if _n(move_pct) and move_pct >= 90:
        W.append({"tier": 3, "msg": f"MOVE percentile {move_pct:.0f} — ความผันผวนพันธบัตรโซนสูงสุดในประวัติ"})
    W.sort(key=lambda w: w["tier"])
    return W[:3]  # cap ที่ 3: งาน clinical DSS พบ override 49-96% เมื่อ alert ล้น
