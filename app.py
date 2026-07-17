"""
Bond Crisis Dashboard v2 — Decision Intelligence สำหรับเฝ้าระวังความเครียดตลาดพันธบัตร

จุดยืนของระบบ (ตามหลักฐานงานวิจัย): "เครื่องมือวินิจฉัยความเสี่ยง (risk diagnostic)"
ไม่ใช่ "เครื่องผลิตสัญญาณกำไร" — ทุกโมดูลแสดงข้อจำกัด/false alarm rate คู่กับตัวเลขเสมอ

รัน: streamlit run app.py   (ดู README.md สำหรับการติดตั้งและ FRED API key ฟรี)
"""
from __future__ import annotations

import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine as E
import data_sources as D
import explain as X

st.set_page_config(page_title="Bond Crisis Dashboard", page_icon="🛡️",
                   layout="wide")

DISCLAIMER = ("ข้อมูลเพื่อการศึกษา/วินิจฉัยความเสี่ยงเท่านั้น ไม่ใช่คำแนะนำการลงทุน — "
              "วรรณกรรม Early Warning Systems ชี้ว่า P(วิกฤต|สัญญาณเตือน) ≈ 50% "
              "(Bussière & Fratzscher, ECB WP 145) และระบบสัญญาณเทรดรายย่อย <3-5% "
              "เท่านั้นที่ทำกำไรสม่ำเสมอ (Barber, Lee, Liu, Odean & Zhang 2020)")


# ---------------------------------------------------------------------------
# Sidebar: แหล่งข้อมูล
# ---------------------------------------------------------------------------
st.sidebar.title("🛡️ Bond Crisis Dashboard")
st.sidebar.caption("v2 — ทุกตัวเลขมีคำอธิบายภาษาคน ดูแท็บ คู่มืออ่านค่า")
mode = st.sidebar.radio("แหล่งข้อมูล", ["Live (FRED + yfinance)", "Demo (ข้อมูลสังเคราะห์)"],
                        help="Live ต้องมี FRED API key (ฟรี) และอินเทอร์เน็ต")
fred_key = ""
if mode.startswith("Live"):
    fred_key = st.sidebar.text_input("FRED API key", type="password",
                                     help="ขอฟรีที่ fred.stlouisfed.org/docs/api/api_key.html")
st.sidebar.caption(DISCLAIMER)


@st.cache_data(ttl=3600, show_spinner="กำลังดึงข้อมูล FRED...")
def load_fred(key: str) -> dict:
    return D.fetch_all_fred(key)


@st.cache_data(ttl=3600, show_spinner="กำลังดึงราคาสินทรัพย์ (yfinance)...")
def load_prices() -> tuple[pd.DataFrame, pd.Series]:
    tickers = list(D.YF_ASSETS) + [D.YF_MOVE]
    px = D.fetch_yf_history(tickers)
    move = px[D.YF_MOVE].dropna() if D.YF_MOVE in px else pd.Series(dtype=float)
    assets = px[[c for c in px.columns if c != D.YF_MOVE]]
    return assets, move


is_demo = mode.startswith("Demo")
if is_demo:
    bundle = D.demo_bundle()
    fred, prices, move = bundle["fred"], bundle["prices"], bundle["move"]
    st.warning("⚠️ **DEMO DATA (สังเคราะห์)** — ตัวเลขทั้งหมดสร้างขึ้นเพื่อทดสอบหน้าจอ "
               "ไม่มีความหมายทางเศรษฐกิจ ห้ามใช้ตัดสินใจใดๆ", icon="⚠️")
else:
    if not fred_key:
        st.info("ใส่ FRED API key ในแถบซ้ายเพื่อดูข้อมูลจริง หรือสลับเป็นโหมด Demo "
                "เพื่อดูหน้าจอก่อน")
        st.stop()
    fred = load_fred(fred_key)
    if fred.get("_errors"):
        st.error("ดึงบาง series ไม่สำเร็จ (แสดงตรงๆ ไม่กลบ): "
                 + json.dumps(fred["_errors"], ensure_ascii=False))
    try:
        prices, move = load_prices()
    except Exception as e:
        st.warning(f"yfinance ใช้ไม่ได้ ({e}) — sub-model MOVE จะ fallback เป็น VIX "
                   "และ Trend State จะว่าง")
        prices, move = pd.DataFrame(), pd.Series(dtype=float)


def S(sid: str) -> pd.Series:
    """helper: ดึง series จาก dict fred (คืน series ว่างถ้าไม่มี)"""
    s = fred.get(sid)
    return s if isinstance(s, pd.Series) else pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# คำนวณ 5 sub-models -> composite
# ---------------------------------------------------------------------------
t10y3m = S("T10Y3M")
spread_latest = float(t10y3m.iloc[-1]) if len(t10y3m) else float("nan")
rec_prob = E.recession_probability(spread_latest)

vol_series = move if len(move) > 100 else S("VIXCLS")
vol_name = "MOVE" if len(move) > 100 else "VIX (fallback)"

breadth = float("nan")
if len(prices) > 250:
    ma200 = prices.rolling(200).mean()
    below = (prices.iloc[-1] < ma200.iloc[-1])
    breadth = float(below.mean() * 100)

subs = {
    "curve": rec_prob * 100 if not math.isnan(rec_prob) else float("nan"),
    "stress": E.percentile_of_latest(S("STLFSI4")),
    "credit": E.percentile_of_latest(S("BAMLH0A0HYM2")),
    "vol": E.percentile_of_latest(vol_series),
    "breadth": breadth,
}
composite, used = E.composite_crisis_score(subs)

# นับวัน invert ต่อเนื่องล่าสุด
inv_days = 0
for v in reversed(t10y3m.dropna().values):
    if v < 0:
        inv_days += 1
    else:
        break

warnings = E.build_warnings(composite, spread_latest, subs["stress"],
                            E.zscore_of_latest(S("BAMLH0A0HYM2")),
                            subs["vol"], inv_days)

TABS = st.tabs(["ภาพรวม", "Yield Curve", "Macro Monitor", "Regime",
                "Trend State", "Scenarios", "Trade Log & สถิติ",
                "Sentiment", "News", "Bond AI", "คู่มืออ่านค่า"])

# ---------------------------------------------------------------------------
# TAB 1: ภาพรวม
# ---------------------------------------------------------------------------
with TABS[0]:
    st.markdown("#### 📋 บทสรุปภาษาคน")
    st.info(X.plain_summary(composite, subs, spread_latest, rec_prob,
                            inv_days, len(warnings), is_demo, vol_name))
    st.caption("บทสรุปนี้สร้างจากกฎที่ตรวจสอบได้ (rule-based) ไม่ใช่ AI แต่งเอง "
               "— ข้อความคงที่ อ้างอิงเฉพาะตัวเลขที่คำนวณจริงบนหน้านี้")
    st.divider()
    left, right = st.columns([1, 2])
    with left:
        st.metric("Composite Crisis Score", f"{composite:.0f}/100"
                  if not math.isnan(composite) else "n/a")
        st.caption("Diagnostic ไม่ใช่คำทำนาย — เฉลี่ยเท่ากัน 5 โมเดล "
                   "(equal weights ตาม Bates-Granger 1969 / Clemen 1989) "
                   f"ใช้จริง {len(used)}/5 โมเดล")
        if warnings:
            for w in warnings:
                icon = {1: "🟥", 2: "🟧", 3: "🟨"}[w["tier"]]
                st.write(f"{icon} Tier {w['tier']}: {w['msg']}")
            st.caption("จำกัดแสดงสูงสุด 3 เตือน — ป้องกัน alert fatigue "
                       "(override rate 49-96% เมื่อ alert ล้น; Ancker et al. 2017)")
        else:
            st.success("ไม่มีสัญญาณเตือนเข้าเกณฑ์ ณ ตอนนี้ (ไม่ได้แปลว่าปลอดภัย — "
                       "วิกฤต 2020/2022 มาจาก leverage ที่ตัวชี้วัดราคาไม่เห็นล่วงหน้า)")
    with right:
        rows = []
        for k, label in E.SUBMODEL_LABELS.items():
            v = subs.get(k)
            rows.append({"โมเดล": label,
                         "คะแนน (0-100)": "n/a" if v is None or math.isnan(v) else f"{v:.0f}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(f"Volatility ใช้ {vol_name} | ข้อมูลถึง: "
                   f"{t10y3m.index[-1].date() if len(t10y3m) else 'n/a'}")

    # ตรวจ diversity ของ ensemble: correlation ระหว่างประวัติ sub-score
    with st.expander("ตรวจสุขภาพ ensemble (correlation ระหว่าง 5 โมเดล)"):
        try:
            hist = {}
            if len(t10y3m) > 300:
                hist["curve"] = t10y3m.apply(E.recession_probability) * 100
            for key_, sid in [("stress", "STLFSI4"), ("credit", "BAMLH0A0HYM2")]:
                s = S(sid)
                if len(s) > 300:
                    hist[key_] = E.percentile_history(s)
            if len(vol_series) > 300:
                hist["vol"] = E.percentile_history(vol_series)
            if hist:
                H = pd.DataFrame(hist).resample("ME").last().dropna()
                corr = E.submodel_correlation(H)
                st.dataframe(corr.round(2), use_container_width=True)
                hot = [(a, b, corr.loc[a, b]) for a in corr.index for b in corr.columns
                       if a < b and corr.loc[a, b] > 0.8]
                if hot:
                    st.warning("คู่ที่ correlation > 0.8: "
                               + ", ".join(f"{a}-{b} ({c:.2f})" for a, b, c in hot)
                               + " — ensemble ไม่ได้ diversity จริง การรวมไม่เพิ่มคุณค่า "
                                 "(เกณฑ์จากรายงานวิจัย); พิจารณาลด/เปลี่ยนโมเดล")
                else:
                    st.success("ไม่มีคู่ใด correlation > 0.8 — diversity ใช้ได้")
        except Exception as e:
            st.info(f"คำนวณ correlation ไม่ได้: {e}")

# ---------------------------------------------------------------------------
# TAB 2: Yield Curve
# ---------------------------------------------------------------------------
with TABS[1]:
    c1, c2, c3 = st.columns(3)
    c1.metric("10Y-3M spread", f"{spread_latest:+.2f}%" if not math.isnan(spread_latest) else "n/a",
              delta="INVERTED" if spread_latest < 0 else "ปกติ",
              delta_color="inverse" if spread_latest < 0 else "normal")
    t10y2y = S("T10Y2Y")
    sp2 = float(t10y2y.iloc[-1]) if len(t10y2y) else float("nan")
    c2.metric("10Y-2Y spread", f"{sp2:+.2f}%" if not math.isnan(sp2) else "n/a")
    c3.metric("P(recession ใน 12 เดือน)",
              f"{rec_prob:.0%}" if not math.isnan(rec_prob) else "n/a")
    st.caption("Probit จาก Estrella & Mishkin (1996, NY Fed); สัมประสิทธิ์ fit กับตาราง "
               "ที่ NY Fed เผยแพร่ (คลาดเคลื่อนสูงสุด 0.13 จุด). **ข้อจำกัดที่ต้องรู้:** "
               "lead time ผันแปร 6-24 เดือน ใช้จับจังหวะเทรดไม่ได้, เคยมี false signals, "
               "และทำนาย recession ไม่ใช่วิกฤตสภาพคล่องพันธบัตร (Taper 2013 / มี.ค. 2020 / "
               "Gilt 2022 ไม่ได้ถูกเตือนล่วงหน้าด้วยตัวนี้)")
    with st.expander("💡 อธิบายแบบง่าย: กราฟนี้บอกอะไร และไม่บอกอะไร"):
        st.markdown(X.curve_explainer(rec_prob, spread_latest, inv_days))

    snap_x, snap_y = [], []
    for sid, tenor in D.CURVE_TENORS_Y.items():
        s = S(sid)
        if len(s):
            snap_x.append(tenor); snap_y.append(float(s.iloc[-1]))
    colA, colB = st.columns(2)
    if snap_x:
        f = go.Figure(go.Scatter(x=snap_x, y=snap_y, mode="lines+markers"))
        f.update_layout(title="Curve snapshot (ล่าสุด)", xaxis_title="อายุ (ปี)",
                        yaxis_title="Yield %", height=340,
                        margin=dict(l=10, r=10, t=40, b=10))
        colA.plotly_chart(f, use_container_width=True)
    if len(t10y3m):
        f2 = go.Figure()
        f2.add_scatter(x=t10y3m.index, y=t10y3m.values, name="10Y-3M")
        if len(t10y2y):
            f2.add_scatter(x=t10y2y.index, y=t10y2y.values, name="10Y-2Y")
        f2.add_hline(y=0, line_dash="dot")
        f2.update_layout(title="Term spreads (ประวัติ)", height=340,
                         margin=dict(l=10, r=10, t=40, b=10))
        colB.plotly_chart(f2, use_container_width=True)
    if inv_days:
        st.info(f"10Y-3M inverted ต่อเนื่อง {inv_days} วันทำการล่าสุด")

# ---------------------------------------------------------------------------
# TAB 3: Macro Monitor
# ---------------------------------------------------------------------------
with TABS[2]:
    st.caption("**สำคัญ:** ดัชนีความเครียดสร้างจากราคาตลาด จึงเป็น *coincident* "
               "(พร้อมเหตุการณ์) ไม่ใช่ *leading* — ใช้เป็นเทอร์โมมิเตอร์สภาวะปัจจุบัน "
               "ไม่ใช่เครื่องทำนาย (Kliesen & Smith 2010)")
    rows = []
    for sid in ["STLFSI4", "NFCI", "BAMLH0A0HYM2", "BAMLC0A0CM", "VIXCLS"]:
        s = S(sid)
        if not len(s):
            continue
        rows.append({"ตัวชี้วัด": D.FRED_SERIES[sid], "ล่าสุด": round(float(s.iloc[-1]), 2),
                     "z-score": round(E.zscore_of_latest(s), 2),
                     "percentile": round(E.percentile_of_latest(s), 0),
                     "ชนิด": "coincident"})
    if len(move) > 100:
        rows.append({"ตัวชี้วัด": "MOVE (Treasury implied vol)",
                     "ล่าสุด": round(float(move.iloc[-1]), 1),
                     "z-score": round(E.zscore_of_latest(move), 2),
                     "percentile": round(E.percentile_of_latest(move), 0),
                     "ชนิด": "coincident"})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    with st.expander("💡 อธิบายแบบง่าย: percentile / z-score อ่านยังไง"):
        st.markdown(X.macro_explainer(subs["stress"],
                                      E.zscore_of_latest(S("STLFSI4"))))
    pick = st.selectbox("ดูกราฟ", ["STLFSI4", "NFCI", "BAMLH0A0HYM2", "VIXCLS"]
                        + (["MOVE"] if len(move) > 100 else []))
    s = move if pick == "MOVE" else S(pick)
    if len(s):
        f = go.Figure(go.Scatter(x=s.index, y=s.values))
        f.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                        title=pick)
        st.plotly_chart(f, use_container_width=True)
    st.caption("บทเรียนกรณีศึกษา 2013/2020/2022/2023: สิ่งที่ 'นำ' วิกฤตคือ "
               "positioning/leverage (carry crowding, basis trade, LDI, duration "
               "mismatch) ไม่ใช่ตัวชี้วัดราคา — ควรตามข้อมูล funding/repo/positioning "
               "เพิ่มใน v2")

# ---------------------------------------------------------------------------
# TAB 4: Regime Detection
# ---------------------------------------------------------------------------
with TABS[3]:
    st.caption("2-state Gaussian HMM (Hamilton 1989) บนการเปลี่ยนแปลงรายสัปดาห์ของ "
               "US 10Y yield — **คำเตือนตามวรรณกรรม:** regime มองเห็นง่ายย้อนหลัง "
               "(smoothed) แต่ยากใน real time (filtered) และการยืนยัน regime มัก "
               "'ช้ากว่าเหตุการณ์' — ใช้เป็นบริบท ไม่ใช่ trigger เทรด")
    y10 = S("DGS10")
    if len(y10) > 400:
        dy = y10.resample("W-FRI").last().diff().dropna()
        try:
            res = E.fit_hmm_2state(dy.values)
            hv = res.high_vol_state
            p_now = float(res.filtered[-1, hv])
            c1, c2, c3 = st.columns(3)
            c1.metric("P(High-vol regime) — real-time (filtered)", f"{p_now:.0%}")
            c2.metric("σ regime สงบ (bp/สัปดาห์)", f"{res.sigma[1-hv]*100:.0f}")
            c3.metric("σ regime ผันผวน (bp/สัปดาห์)", f"{res.sigma[hv]*100:.0f}")
            f = go.Figure()
            f.add_scatter(x=dy.index, y=res.smoothed[:, hv], name="smoothed (ย้อนหลัง)",
                          line=dict(width=1))
            f.add_scatter(x=dy.index, y=res.filtered[:, hv], name="filtered (real-time)",
                          line=dict(width=1, dash="dot"))
            f.update_layout(title="P(High-volatility regime)", height=340,
                            yaxis_range=[0, 1], margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(f, use_container_width=True)
            st.caption("ช่องว่างระหว่างเส้น smoothed กับ filtered = detection lag ตามจริง "
                       f"| EM converged: {res.converged}")
            with st.expander("💡 อธิบายแบบง่าย: regime และสองเส้นนี้คืออะไร"):
                st.markdown(X.regime_explainer(p_now, res.sigma[1-hv]*100,
                                               res.sigma[hv]*100))
        except Exception as e:
            st.error(f"HMM ไม่ converge/ข้อมูลไม่พอ: {e}")
    else:
        st.info("ต้องการประวัติ DGS10 มากกว่านี้")

# ---------------------------------------------------------------------------
# TAB 5: Trend State (เดิมคือ 'Signal Trading' — เปลี่ยนชื่อให้ตรงหลักฐาน)
# ---------------------------------------------------------------------------
with TABS[4]:
    st.caption("**นี่คือ 'สถานะแนวโน้ม' เพื่อเป็นบริบท ไม่ใช่สัญญาณซื้อขาย** — "
               "หลักฐาน: day traders ขาดทุนเฉลี่ย 23.9bp/วันหลังต้นทุน และ <3% "
               "ทำกำไรได้สม่ำเสมอ (Barber, Lee, Liu, Odean & Zhang 2020); "
               "Sharpe 1.28 ของ TSMOM คือพอร์ต 58 ตลาด long-short ไม่ใช่การถือ "
               "long ไม่กี่ตัว (Moskowitz-Ooi-Pedersen 2012; ถูกโต้โดย Huang et al. 2020)")
    if len(prices) > 260:
        ma200 = prices.rolling(200).mean()
        mom12 = prices.pct_change(252)
        vol20 = prices.pct_change().rolling(20).std() * math.sqrt(252) * 100
        rows = []
        for tkr in prices.columns:
            last = prices[tkr].dropna()
            if len(last) < 260:
                continue
            rows.append({
                "สินทรัพย์": D.YF_ASSETS.get(tkr, tkr),
                "ราคา": round(float(last.iloc[-1]), 2),
                "เทียบ 200DMA": "เหนือ" if last.iloc[-1] >= ma200[tkr].iloc[-1] else "ใต้",
                "โมเมนตัม 12 เดือน": f"{mom12[tkr].iloc[-1]:+.1%}"
                    if not math.isnan(mom12[tkr].iloc[-1]) else "n/a",
                "Realized vol 20d (ปี)": f"{vol20[tkr].iloc[-1]:.0f}%"
                    if not math.isnan(vol20[tkr].iloc[-1]) else "n/a",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if not math.isnan(breadth):
            st.metric("Risk-off breadth (% ใต้ 200DMA)", f"{breadth:.0f}%")
        longs = sum(1 for r in rows if r["เทียบ 200DMA"] == "เหนือ")
        if rows and longs == len(rows):
            st.warning("ทุกสินทรัพย์อยู่ฝั่งเดียวกัน — ถ้าถือ long ทั้งหมดพร้อมกัน "
                       "นั่นคือ concentration risk ไม่ใช่ diversification แบบใน "
                       "งานวิจัย TSMOM")
    else:
        st.info("ไม่มีข้อมูลราคา (yfinance) — โหมด Live ต้องติดตั้ง yfinance")
    with st.expander("💡 อธิบายแบบง่าย: อ่านตารางนี้ยังไง"):
        st.markdown(X.trend_explainer(breadth))

# ---------------------------------------------------------------------------
# TAB 6: Scenarios
# ---------------------------------------------------------------------------
with TABS[5]:
    st.caption("Scenario range แทน point forecast — เพราะ track record ของ point "
               "forecast ทางเศรษฐกิจแย่มาก (SPF อธิบาย variability ~0% ที่ 4 ไตรมาส; "
               "St. Louis Fed 2023) ธนาคารกลางจึงใช้ severe-but-plausible scenarios")
    c1, c2 = st.columns(2)
    dur = c1.number_input("Duration ของพอร์ตพันธบัตร (ปี)", 0.0, 30.0, 7.0, 0.5)
    conv = c2.number_input("Convexity", 0.0, 500.0, 60.0, 5.0)
    shifts = [-200, -100, -50, -25, 0, 25, 50, 100, 200]
    tbl = pd.DataFrame({
        "Yield shift (bp)": shifts,
        "ผลกระทบราคาโดยประมาณ (%)": [round(E.bond_price_impact(dur, conv, s), 2)
                                      for s in shifts],
    })
    st.dataframe(tbl, hide_index=True, use_container_width=True)
    st.caption("สูตร dP/P ≈ -D·Δy + ½·C·Δy² เป็น local approximation — "
               "shift ใหญ่/curve ไม่ขนานจะคลาดเคลื่อน")
    st.subheader("Reference scenarios จากวิกฤตจริง")
    st.dataframe(pd.DataFrame(E.HISTORICAL_EPISODES).rename(columns={
        "name": "เหตุการณ์", "move": "ขนาดการเคลื่อนไหว",
        "precursor": "สิ่งที่นำหน้า (ของจริง)", "lesson": "บทเรียน"}),
        hide_index=True, use_container_width=True)
    with st.expander("💡 อธิบายแบบง่าย: duration และ convexity คืออะไร"):
        st.markdown(X.scenario_explainer(dur, E.bond_price_impact(dur, conv, 100)))

# ---------------------------------------------------------------------------
# TAB 7: Trade Log & สถิติ (Backtest Research v1)
# ---------------------------------------------------------------------------
with TABS[6]:
    st.caption("อัปโหลด trade log (CSV มีคอลัมน์ `pnl`) เพื่อดูว่าตัวเลขของคุณ "
               "**บอกอะไรได้จริงทางสถิติ** — โมดูลนี้มีไว้กันการหลอกตัวเองจาก "
               "sample เล็ก (เช่น PF 0.65 จาก ~10 เทรด = ยังสรุป edge ไม่ได้ "
               "แต่ก็คือขาดทุนจริงในช่วงที่วัด)")
    up = st.file_uploader("trade log CSV (คอลัมน์บังคับ: pnl)", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            if "pnl" not in df.columns:
                st.error("ไม่พบคอลัมน์ 'pnl'")
            else:
                r = E.trade_log_report(df)
                c = st.columns(5)
                c[0].metric("จำนวนเทรด", r["n"])
                c[1].metric("Win rate", f"{r['win_rate']:.0%}",
                            f"CI95: {r['ci_low']:.0%}-{r['ci_high']:.0%}")
                pf_txt = "∞" if math.isinf(r["profit_factor"]) else f"{r['profit_factor']:.2f}"
                c[2].metric("Profit factor", pf_txt)
                c[3].metric("Expectancy/เทรด", f"{r['expectancy']:.2f}")
                c[4].metric("PSR (P(SR>0))",
                            f"{r['psr']:.0%}" if not math.isnan(r["psr"]) else "n/a")
                st.info(r["verdict"])
                st.markdown("**💡 แปลผลแบบภาษาคน**")
                st.markdown(X.interpret_trade_log(r))
                st.caption("PSR = Probabilistic Sharpe Ratio (Bailey & López de Prado "
                           "2012). ถ้ากลยุทธ์นี้ถูก 'เลือก' จากการลองหลายตัว ต้องใช้ "
                           "Deflated SR (2014) ซึ่งต้องรู้จำนวน trials — ค่าที่เห็น "
                           "จึงมองโลกดีเกินจริงเสมอในกรณีนั้น | เกณฑ์จากรายงานวิจัย: "
                           "ถ้าครบ 100+ เทรดแล้ว PF ยัง <1.0 → ไม่มี edge ควรหยุด/redesign")
        except Exception as e:
            st.error(f"อ่านไฟล์ไม่ได้: {e}")
    with st.expander("เครื่องคิดเลข: ต้องเทรดกี่ครั้งถึงเชื่อ win rate ได้"):
        moe = st.slider("ยอมรับความคลาดเคลื่อน (± จุด)", 1, 15, 5) / 100
        st.write(f"ต้องการประมาณ **{E.required_n(moe):,} เทรด** "
                 f"(95% confidence, กรณีแย่สุด p=0.5)")

# ---------------------------------------------------------------------------
# TAB 8: Retail Sentiment (AAII — อัปโหลดไฟล์)
# ---------------------------------------------------------------------------
with TABS[7]:
    st.caption("AAII ระบุเองว่า survey **'does not predict future market direction'** — "
               "งานวิจัย/practitioner ใช้เฉพาะ *extreme readings* เป็นบริบท contrarian "
               "ระยะ 6-12 เดือน (และ extreme คงอยู่ต่อได้นาน — เข้าเร็วไปก็ขาดทุน). "
               "ดาวน์โหลดข้อมูลจาก aaii.com แล้วอัปโหลดเป็น CSV: คอลัมน์ "
               "`date,bullish,bearish` (สัดส่วน 0-1 หรือ %)")
    upA = st.file_uploader("AAII CSV", type=["csv"], key="aaii")
    if upA is not None:
        try:
            a = pd.read_csv(upA)
            a.columns = [c.strip().lower() for c in a.columns]
            a["date"] = pd.to_datetime(a["date"])
            for col in ("bullish", "bearish"):
                if a[col].max() > 1.5:
                    a[col] = a[col] / 100.0
            a["spread"] = (a["bullish"] - a["bearish"]) * 100
            a = a.sort_values("date")
            f = go.Figure(go.Scatter(x=a["date"], y=a["spread"], name="Bull-Bear (pp)"))
            f.add_hline(y=30, line_dash="dot"); f.add_hline(y=-30, line_dash="dot")
            f.update_layout(title="AAII Bull-Bear spread (เส้นประ = โซน extreme ±30pp)",
                            height=340, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(f, use_container_width=True)
            last = a.iloc[-1]
            if last["spread"] >= 30:
                st.warning(f"Extreme bullish ({last['spread']:.0f}pp) — บริบท contrarian "
                           "เชิงลบระยะ 6-12 เดือนตามสถิติในอดีต (ไม่ deterministic)")
            elif last["spread"] <= -30:
                st.info(f"Extreme bearish ({last['spread']:.0f}pp) — บริบท contrarian "
                        "เชิงบวกระยะ 6-12 เดือนตามสถิติในอดีต (ไม่ deterministic)")
            else:
                st.write(f"ล่าสุด {last['spread']:.0f}pp — ไม่ extreme, ไม่มีนัยใช้งาน")
        except Exception as e:
            st.error(f"อ่านไฟล์ไม่ได้ (ต้องมี date,bullish,bearish): {e}")

# ---------------------------------------------------------------------------
# TAB 9: News (headlines เท่านั้น — v1 ไม่ทำ sentiment score)
# ---------------------------------------------------------------------------
with TABS[8]:
    st.caption("v1 แสดง headlines จากแหล่งทางการเท่านั้น **โดยตั้งใจไม่ให้คะแนน "
               "sentiment** — dictionary methods เพดาน F1 ~65-70% บนข้อความการเงิน "
               "และผลวิจัยที่อ้าง Sharpe สูงมักมี look-ahead bias (ข่าวถูก price in เร็ว)")
    default_feeds = "https://www.federalreserve.gov/feeds/press_all.xml"
    feeds = st.text_area("RSS feeds (บรรทัดละ 1 URL)", default_feeds, height=80)
    if st.button("ดึงหัวข้อข่าว"):
        try:
            import feedparser  # optional
            for url in [u.strip() for u in feeds.splitlines() if u.strip()]:
                d = feedparser.parse(url)
                st.subheader(d.feed.get("title", url))
                for e in d.entries[:8]:
                    st.markdown(f"- [{e.get('title','(no title)')}]({e.get('link','#')}) "
                                f"— {e.get('published','')}")
        except ImportError:
            st.error("ต้องติดตั้ง feedparser ก่อน: pip install feedparser")
        except Exception as e:
            st.error(f"ดึง feed ไม่ได้: {e}")

# ---------------------------------------------------------------------------
# TAB 10: Bond AI (decision support — ไม่ใช่ที่ปรึกษาการลงทุน)
# ---------------------------------------------------------------------------
with TABS[9]:
    st.caption("**Decision support ไม่ใช่ investment advice** — LLM มี hallucination "
               "ในงานการเงิน (Kang & Liu 2023) และคำนวณเลขไม่น่าเชื่อถือ (FAITH 2025: "
               "accuracy ตกใกล้ 0% ในการคำนวณ multivariate) — ระบบนี้จึง **ส่งตัวเลข "
               "ที่ dashboard คำนวณแล้วให้ AI อธิบายเท่านั้น และห้าม AI คำนวณเลขใหม่**")
    api_key = st.text_input("Anthropic API key (ไม่บังคับ)", type="password")
    context = {
        "as_of": str(datetime.now().date()),
        "data_mode": "DEMO (synthetic — ห้ามตีความ)" if is_demo else "LIVE",
        "composite_score_0_100": None if math.isnan(composite) else round(composite, 1),
        "submodels": {k: (None if v is None or math.isnan(v) else round(v, 1))
                      for k, v in subs.items()},
        "spread_10y3m_pct": None if math.isnan(spread_latest) else round(spread_latest, 2),
        "recession_prob_12m": None if math.isnan(rec_prob) else round(rec_prob, 3),
        "inverted_days": inv_days,
        "warnings": [w["msg"] for w in warnings],
    }
    with st.expander("ข้อมูลที่ส่งให้ AI (โปร่งใส)"):
        st.code(json.dumps(context, ensure_ascii=False, indent=2))
    SYSTEM = ("คุณคือ Bond AI ผู้ช่วยอธิบายผลของ Bond Crisis Dashboard "
              "กติกาเคร่งครัด: (1) ใช้เฉพาะตัวเลขใน context ที่ให้ ห้ามคำนวณ/ประมาณ"
              "ตัวเลขใหม่เอง ถ้าไม่มีข้อมูลให้บอกว่าไม่มี (2) เป็น decision support "
              "ห้ามให้คำแนะนำซื้อ/ขาย/ถือ หรือคำแนะนำการลงทุนเฉพาะบุคคล "
              "(3) พูดถึงข้อจำกัดของตัวชี้วัดเสมอเมื่อเกี่ยวข้อง (false alarms, lead time, "
              "coincident vs leading) (4) ตอบภาษาไทย ตรงไปตรงมา ไม่ขายฝัน "
              f"(5) ถ้า data_mode เป็น DEMO ให้ย้ำทุกคำตอบว่าเป็นข้อมูลสังเคราะห์\n\n"
              f"context: {json.dumps(context, ensure_ascii=False)}")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for m in st.session_state.chat:
        st.chat_message(m["role"]).write(m["content"])
    q = st.chat_input("ถามเกี่ยวกับตัวเลขบน dashboard...")
    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        st.chat_message("user").write(q)
        if not api_key:
            msg = "ยังไม่ได้ใส่ Anthropic API key — ใส่ key ด้านบนเพื่อใช้งาน Bond AI"
            st.chat_message("assistant").write(msg)
            st.session_state.chat.append({"role": "assistant", "content": msg})
        else:
            try:
                import anthropic  # optional
                client = anthropic.Anthropic(api_key=api_key)
                resp = client.messages.create(
                    model="claude-sonnet-4-6", max_tokens=1000, system=SYSTEM,
                    messages=[{"role": m["role"], "content": m["content"]}
                              for m in st.session_state.chat],
                )
                ans = "".join(b.text for b in resp.content if b.type == "text")
                st.chat_message("assistant").write(ans)
                st.session_state.chat.append({"role": "assistant", "content": ans})
            except ImportError:
                st.error("ต้องติดตั้งก่อน: pip install anthropic")
            except Exception as e:
                st.error(f"เรียก API ไม่สำเร็จ: {e}")

st.divider()
st.caption(DISCLAIMER + " | โหมดข้อมูล: " + ("DEMO (สังเคราะห์)" if is_demo else "LIVE"))

# ---------------------------------------------------------------------------
# TAB 11: คู่มืออ่านค่า (อภิธานศัพท์ภาษาคน)
# ---------------------------------------------------------------------------
with TABS[10]:
    st.subheader("คู่มืออ่านค่า — ทุกตัวเลขบนแดชบอร์ด อธิบายแบบภาษาคน")
    st.caption("ทุกคำมี 3 ส่วนเสมอ: คืออะไร / วิธีอ่าน / **สิ่งที่มัน *ไม่ได้* บอก** "
               "— ส่วนที่สามสำคัญที่สุด เพราะการตีความเกินจริงคือความเสี่ยงหลัก"
               "ของแดชบอร์ดทุกอัน")
    for term, d in X.GLOSSARY.items():
        with st.expander(term):
            st.markdown(f"**คืออะไร:** {d['what']}")
            st.markdown(f"**วิธีอ่าน:** {d['read']}")
            st.markdown(f"**สิ่งที่มัน *ไม่ได้* บอก:** {d['not']}")
