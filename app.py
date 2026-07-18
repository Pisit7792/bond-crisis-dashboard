"""
Bond Crisis Dashboard v3 — Decision Intelligence เฝ้าระวังความเครียดตลาดพันธบัตร

v3 เพิ่มตามสเปกผู้ใช้: 6 โมเดลทำกำไร (สินทรัพย์ได้/เสียประโยชน์), จำลองสถานการณ์,
วิกฤตแบงก์รัน, ข่าว→โมเดล, ห้องประชุม AI, สัญญาณเทรด (โมเดล ≥40 + เทคนิคอลกำกับ)

จุดยืนเดิมไม่เปลี่ยน: risk diagnostic ไม่ใช่เครื่องผลิตกำไร — ทุกฟีเจอร์มีชั้น
ความซื่อสัตย์ (ข้อจำกัด, ขนาดตัวอย่าง, ความไม่เสถียรของความสัมพันธ์) กำกับเสมอ
รัน: streamlit run app.py
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
import models6 as M
import scenario as SC
import signals as SG
import meeting as MT

st.set_page_config(page_title="Bond Crisis Dashboard v3", page_icon="🛡️",
                   layout="wide")

DISCLAIMER = ("ข้อมูลเพื่อการศึกษา/วินิจฉัยความเสี่ยงเท่านั้น ไม่ใช่คำแนะนำการลงทุน — "
              "P(วิกฤต|สัญญาณเตือน) ≈ 50% ในวรรณกรรม EWS และ <3-5% ของรายย่อย"
              "ทำกำไรจากสัญญาณได้สม่ำเสมอ (Barber et al. 2020)")

PAGES = ["ภาพรวม", "โมเดลทำกำไร (6)", "สัญญาณเทรด", "จำลองสถานการณ์",
         "วิกฤตแบงก์รัน", "ข่าวสาร", "ห้องประชุม AI", "ข้อมูลมหภาค",
         "Trend สินทรัพย์", "Trade Log & สถิติ", "Sentiment", "คู่มืออ่านค่า"]

# ---------------------------------------------------------------------------
# Sidebar: นำทาง + แหล่งข้อมูล
# ---------------------------------------------------------------------------
st.sidebar.title("🛡️ Bond Crisis v3")
page = st.sidebar.radio("เมนู", PAGES, label_visibility="collapsed")
st.sidebar.divider()
mode = st.sidebar.radio("แหล่งข้อมูล", ["Live (FRED + yfinance)", "Demo (สังเคราะห์)"])
fred_key = ""
if mode.startswith("Live"):
    fred_key = st.sidebar.text_input("FRED API key", type="password",
                                     help="ฟรีที่ fred.stlouisfed.org/docs/api/api_key.html")
st.sidebar.caption(DISCLAIMER)


@st.cache_data(ttl=3600, show_spinner="กำลังดึงข้อมูล FRED...")
def load_fred(key: str) -> dict:
    return D.fetch_all_fred(key)


@st.cache_data(ttl=3600, show_spinner="กำลังดึงราคาสินทรัพย์...")
def load_prices() -> tuple[pd.DataFrame, pd.Series]:
    tickers = list(D.YF_ASSETS) + [D.YF_MOVE]
    px = D.fetch_yf_history(tickers)
    move = px[D.YF_MOVE].dropna() if D.YF_MOVE in px else pd.Series(dtype=float)
    assets = px[[c for c in px.columns if c != D.YF_MOVE]]
    return assets, move


is_demo = mode.startswith("Demo")
if is_demo:
    b = D.demo_bundle()
    fred, prices, move = b["fred"], b["prices"], b["move"]
    st.warning("⚠️ **DEMO DATA (สังเคราะห์)** — ห้ามใช้ตัดสินใจใดๆ", icon="⚠️")
else:
    if not fred_key:
        st.info("ใส่ FRED API key ในแถบซ้าย หรือสลับเป็นโหมด Demo เพื่อดูหน้าจอ")
        st.stop()
    fred = load_fred(fred_key)
    if fred.get("_errors"):
        st.error("ดึงบาง series ไม่สำเร็จ: "
                 + json.dumps(fred["_errors"], ensure_ascii=False))
    try:
        prices, move = load_prices()
    except Exception as e:
        st.warning(f"yfinance ใช้ไม่ได้ ({e}) — สัญญาณเทรด/Trend จะว่าง, "
                   "MOVE จะ fallback เป็น VIX")
        prices, move = pd.DataFrame(), pd.Series(dtype=float)


def S(sid: str) -> pd.Series:
    s = fred.get(sid)
    return s if isinstance(s, pd.Series) else pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# คำนวณแกนกลาง: Crisis composite (เตือนภัย) + 6 โมเดลทำกำไร + สัญญาณ
# ---------------------------------------------------------------------------
t10y3m = S("T10Y3M")
spread_latest = float(t10y3m.iloc[-1]) if len(t10y3m) else float("nan")
rec_prob = E.recession_probability(spread_latest)
vol_series = move if len(move) > 100 else S("VIXCLS")
vol_name = "MOVE" if len(move) > 100 else "VIX (fallback)"
breadth = float("nan")
if len(prices) > 250:
    breadth = float((prices.iloc[-1] < prices.rolling(200).mean().iloc[-1]).mean() * 100)

subs = {"curve": rec_prob * 100 if not math.isnan(rec_prob) else float("nan"),
        "stress": E.percentile_of_latest(S("STLFSI4")),
        "credit": E.percentile_of_latest(S("BAMLH0A0HYM2")),
        "vol": E.percentile_of_latest(vol_series),
        "breadth": breadth}
composite, used = E.composite_crisis_score(subs)
inv_days = 0
for v in reversed(t10y3m.dropna().values):
    if v < 0:
        inv_days += 1
    else:
        break
warnings = E.build_warnings(composite, spread_latest, subs["stress"],
                            E.zscore_of_latest(S("BAMLH0A0HYM2")),
                            subs["vol"], inv_days)

MDATA = dict(fred)
MDATA["MOVE"] = move
mscores = M.score_models(MDATA)


@st.cache_data(ttl=3600, show_spinner="คำนวณประวัติคะแนนโมเดล...")
def model_deltas(cache_key: str) -> dict:
    out = {}
    for k in M.MODEL_DEFS:
        try:
            out[k] = round(M.model_delta(M.score_history(MDATA, k)), 1)
        except Exception:
            out[k] = float("nan")
    return out


cache_key = ("demo" if is_demo else "live") + str(
    t10y3m.index[-1].date() if len(t10y3m) else "")
deltas = model_deltas(cache_key)
sig = SG.build_signals(mscores, M.ASSET_IMPACT, prices, D.YF_ASSETS) \
    if len(prices) else {"signals": [], "conflicts": [], "skipped": []}

MODEL_ORDER = sorted(M.MODEL_DEFS, key=lambda k: -(mscores[k]["score"]
                     if mscores[k]["score"] == mscores[k]["score"] else -1))


def model_bar_chart():
    keys = MODEL_ORDER[::-1]
    vals = [mscores[k]["score"] for k in keys]
    ths = [mscores[k]["th"] for k in keys]
    f = go.Figure(go.Bar(x=vals, y=ths, orientation="h",
                         text=[f"{v:.1f}" for v in vals], textposition="outside"))
    f.update_layout(height=300, xaxis_range=[0, 100],
                    margin=dict(l=10, r=40, t=10, b=10))
    return f


# ---------------------------------------------------------------------------
# หน้า: ภาพรวม
# ---------------------------------------------------------------------------
def page_overview():
    st.markdown("#### 📋 บทสรุปภาษาคน")
    st.info(X.plain_summary(composite, subs, spread_latest, rec_prob,
                            inv_days, len(warnings), is_demo, vol_name))
    st.caption("บทสรุปเป็น rule-based (ไม่ใช่ AI แต่ง) — อ้างเฉพาะตัวเลขที่คำนวณจริง")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Crisis Score (ระบบเตือนภัย)",
                  f"{composite:.0f}/100" if composite == composite else "n/a")
        for w in warnings:
            st.write({1: "🟥", 2: "🟧", 3: "🟨"}[w["tier"]] + f" {w['msg']}")
        if not warnings:
            st.success("ไม่มีเตือนเข้าเกณฑ์ (ความเงียบ ≠ ปลอดภัย — วิกฤตจริงมาจาก "
                       "leverage ที่มองไม่เห็น)")
        st.metric("สัญญาณเทรดที่เข้าเกณฑ์", len(sig["signals"]),
                  f"ขัดแย้ง {len(sig['conflicts'])} | ไม่ผ่าน filter {len(sig['skipped'])}")
    with c2:
        st.markdown("**6 โมเดลทำกำไร (คะแนนสภาพแวดล้อม 0-100)**")
        st.plotly_chart(model_bar_chart(), use_container_width=True)
        st.caption("คะแนน = 'สภาพแวดล้อมแบบนั้นเด่นแค่ไหนเทียบอดีต' ไม่ใช่ความน่าจะเป็น"
                   "กำไร | Δ สัปดาห์: "
                   + ", ".join(f"{mscores[k]['th']} {deltas.get(k, float('nan')):+.1f}"
                               for k in MODEL_ORDER))


# ---------------------------------------------------------------------------
# หน้า: โมเดลทำกำไร (6)
# ---------------------------------------------------------------------------
def page_models():
    st.caption("คะแนนทุกตัวมาจากสูตรเปิดเผย (percentile ของตัวชี้วัดจริง) — "
               "ตาราง 'ได้/เสียประโยชน์' คือแนวโน้มตามประวัติศาสตร์ *ไม่ใช่กฎตายตัว* "
               "ความสัมพันธ์พังได้ (หุ้น-บอนด์ 2022, ทอง+BTC ตอนแบงก์รัน 2023)")
    for k in MODEL_ORDER:
        m = mscores[k]
        imp = M.ASSET_IMPACT[k]
        sc = m["score"]
        d = deltas.get(k, float("nan"))
        with st.expander(f"{m['th']} — {sc:.1f}/100"
                         + (f"  (Δสัปดาห์ {d:+.1f})" if d == d else ""),
                         expanded=(k == MODEL_ORDER[0])):
            st.write(m["desc"])
            cols = st.columns(3)
            cols[0].markdown("**ส่วนประกอบคะแนน (percentile)**")
            for name, v in m["components"].items():
                cols[0].write(f"- {name}: {v:.0f}")
            for miss in m["missing"]:
                cols[0].write(f"- {miss}: ⚠️ ไม่มีข้อมูล")
            cols[1].markdown("**ได้ประโยชน์ (แนวโน้มอดีต)**")
            for a in imp["benefit"]:
                cols[1].write(f"🟢 {a}")
            cols[2].markdown("**เสียประโยชน์ (แนวโน้มอดีต)**")
            for a in imp["lose"]:
                cols[2].write(f"🔴 {a}")
            st.warning(f"ข้อจำกัด: {imp['note']}")
    st.caption(f"เกณฑ์สัญญาณ: โมเดล ≥ {SG.SIGNAL_THRESHOLD:.0f} จุด → ดูหน้า 'สัญญาณเทรด' "
               f"| เกณฑ์เรียกประชุม: ขยับ ≥ {MT.DELTA_TRIGGER_PTS:.0f} จุด/สัปดาห์")


# ---------------------------------------------------------------------------
# หน้า: สัญญาณเทรด
# ---------------------------------------------------------------------------
def page_signals():
    st.error(SG.DISCLAIMER, icon="⚠️")
    st.caption(f"กติกาที่ประกาศล่วงหน้า: โมเดล ≥ {SG.SIGNAL_THRESHOLD:.0f} → สินทรัพย์ใน"
               "ตารางได้/เสียประโยชน์ | LONG เฉพาะเหนือ 200DMA, SHORT เฉพาะใต้ "
               f"(ไม่สวนแนวโน้ม) | SL {SG.SL_ATR:.0f}×ATR, TP {SG.TP_ATR:.0f}×ATR (R:R 1:2) "
               "| โมเดลขัดแย้ง → ไม่ออกสัญญาณ | ATR ประมาณจากราคาปิด")
    if not len(prices):
        st.info("ไม่มีข้อมูลราคา (yfinance) — โหมด Demo หรือเครื่องที่ต่อเน็ตเท่านั้น")
        return
    if sig["signals"]:
        df = pd.DataFrame(sig["signals"])[
            ["asset", "side", "model", "strength", "entry", "sl", "tp", "rr",
             "atr14≈", "rsi14", "mom12m%", "dist_200dma%", "trend"]]
        df.columns = ["สินทรัพย์", "ทิศ", "โมเดลที่มา", "ความแข็งแรง", "ราคาเข้า",
                      "SL", "TP", "R:R", "ATR14≈", "RSI14", "โมเมนตัม 12ด.%",
                      "ห่าง 200DMA %", "แนวโน้ม"]
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.caption("'ความแข็งแรง' = คะแนนโมเดลที่มา (สูตรเปิดเผย) | เทคนิคอล (RSI/"
                   "โมเมนตัม/ระยะจาก 200DMA) เป็น *คำอธิบายสภาพราคา* ไม่ใช่การพยากรณ์")
        st.download_button("⬇️ บันทึก Signal Journal (CSV)",
                           SG.journal_csv(sig["signals"], str(datetime.now().date())),
                           file_name="signal_journal.csv", mime="text/csv")
        st.info("วินัยหลักฐาน: กรอกผล pnl เมื่อปิดออเดอร์ แล้วอัปโหลดที่หน้า "
                "'Trade Log & สถิติ' — ห้ามตัดสินระบบก่อนครบ 100 เทรด "
                "(PF จาก 9 เทรดคือ noise ทั้งขาดีและขาร้าย)")
    else:
        st.write("ไม่มีสัญญาณเข้าเกณฑ์ตอนนี้ (ไม่มีโมเดลถึง "
                 f"{SG.SIGNAL_THRESHOLD:.0f} จุด หรือไม่ผ่าน trend filter)")
    if sig["conflicts"]:
        st.markdown("**โมเดลขัดแย้งกัน (ระบบไม่เลือกข้างเอง):**")
        st.dataframe(pd.DataFrame(sig["conflicts"]), hide_index=True,
                     use_container_width=True)
    if sig["skipped"]:
        with st.expander(f"เข้าเกณฑ์โมเดลแต่ถูกกรองออก ({len(sig['skipped'])}) — ดูเหตุผล"):
            st.dataframe(pd.DataFrame(sig["skipped"]), hide_index=True,
                         use_container_width=True)


# ---------------------------------------------------------------------------
# หน้า: จำลองสถานการณ์
# ---------------------------------------------------------------------------
def page_scenario():
    st.caption("การจำลองเป็นค่าประมาณ *ทิศทาง* จากเมทริกซ์ความไวที่เปิดเผย — "
               "ตัวเลขความไวเป็น 'สมมติฐานการออกแบบ' อิงทิศทางในประวัติศาสตร์ "
               "*ไม่ใช่* ค่าที่ประมาณจากข้อมูลจริง และไม่ใช่ผลคำนวณเต็มรูปแบบ")
    left, right = st.columns([1, 1])
    vals = {}
    with left:
        st.markdown("**ปรับสถานการณ์สมมติ**")
        if st.button("Reset ทุกตัว"):
            for k, *_rest in SC.SLIDERS:
                st.session_state.pop(f"sc_{k}", None)
            st.rerun()
        for k, th, u, lo, hi, step, dflt in SC.SLIDERS:
            vals[k] = st.slider(f"{th} ({u})", float(lo), float(hi),
                                float(st.session_state.get(f"sc_{k}", dflt)),
                                float(step), key=f"sc_{k}")
    base = {k: mscores[k]["score"] for k in M.MODEL_DEFS}
    res = SC.apply_scenario(base, vals)
    with right:
        st.markdown("**ผลกระทบต่อคะแนนโมเดล**")
        order = sorted(res, key=lambda k: -res[k]["new"])
        for i, k in enumerate(order, 1):
            r = res[k]
            arrow = "→"
            color = "🟢" if r["delta"] > 0.05 else ("🔴" if r["delta"] < -0.05 else "⚪")
            st.write(f"#{i} **{mscores[k]['th']}**  {r['base']:.1f} {arrow} "
                     f"**{r['new']:.1f}**  {color} {r['delta']:+.1f}")
            st.progress(min(1.0, r["new"] / 100.0))
        moved = [k for k in order if abs(res[k]["delta"]) > 0.05]
        if moved:
            top = moved[0]
            st.info(f"อ่านผล: สถานการณ์นี้กระทบ **{mscores[top]['th']}** มากสุด "
                    f"({res[top]['delta']:+.1f} จุด) → ดูสินทรัพย์ได้/เสียประโยชน์"
                    "ของโมเดลนั้นในหน้า 'โมเดลทำกำไร' — และจำไว้ว่านี่คือ what-if "
                    "ไม่ใช่คำพยากรณ์ว่าจะเกิด")
    with st.expander("🔍 เมทริกซ์ความไว (จุดต่อ 1 หน่วย) — โปร่งใส แก้ได้ในโค้ด scenario.py"):
        st.dataframe(SC.sensitivity_table(), use_container_width=True)
        st.caption("ทิศทางอ้างอิงเหตุการณ์จริง เช่น เงินฝากไหลออก→แบงก์รัน (2023), "
                   "repo ตึง (ก.ย. 2019), ประมูลอ่อน→yield ตึง — แต่ขนาดตัวเลขเป็น"
                   "สมมติฐาน ไม่ใช่ค่าประมาณทางสถิติ")


# ---------------------------------------------------------------------------
# หน้า: วิกฤตแบงก์รัน
# ---------------------------------------------------------------------------
def page_bankrun():
    m = mscores["bank_run"]
    st.metric("คะแนนโมเดลแบงก์รัน", f"{m['score']:.1f}/100"
              if m["score"] == m["score"] else "n/a",
              f"Δสัปดาห์ {deltas.get('bank_run', float('nan')):+.1f}")
    st.write("ส่วนประกอบ: " + " | ".join(f"{n}: {v:.0f}"
             for n, v in m["components"].items()))
    charts = [("DPSACBW027SBOG", "เงินฝากธนาคารพาณิชย์ (ระดับ)"),
              ("BORROW", "ยอดกู้จาก Fed (BORROW)"),
              ("RRPONTSYD", "Reverse Repo (ON RRP)")]
    cols = st.columns(3)
    for (sid, title), c in zip(charts, cols):
        s = S(sid)
        if len(s):
            f = go.Figure(go.Scatter(x=s.index, y=s.values))
            f.update_layout(title=title, height=260,
                            margin=dict(l=10, r=10, t=40, b=10))
            c.plotly_chart(f, use_container_width=True)
    sofr, effr = S("SOFR"), S("EFFR")
    if len(sofr) and len(effr):
        sp = ((sofr - effr) * 100).dropna()
        f = go.Figure(go.Scatter(x=sp.index, y=sp.values))
        f.add_hline(y=0, line_dash="dot")
        f.update_layout(title="SOFR - EFFR (bp) — repo ตึงเมื่อถ่างขึ้น", height=260,
                        margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(f, use_container_width=True)
    st.error("**สิ่งที่ระบบนี้มองไม่เห็น (เหตุผลที่แบงก์รันมัก 'จู่ๆ ก็มา'):** "
             "สัดส่วนเงินฝากไม่คุ้มครอง และผลขาดทุนซ่อนใน HTM เป็นข้อมูล *รายไตรมาส* "
             "จาก filings — SVB มี HTM loss ~$15bn ≈ ทุนทั้งก้อน และเงินฝากไม่คุ้มครอง "
             "~94% อยู่ใน footnotes ก่อนล้ม; ทั้งระบบแบงก์สหรัฐมี unrealized loss "
             "$620.4bn ณ สิ้นปี 2022 (FDIC) — dashboard รายวันจับสิ่งเหล่านี้ไม่ได้ "
             "จับได้แค่ 'อาการ' (เงินฝากไหล, กู้ Fed, repo ตึง) ซึ่งมาช้ากว่าเหตุ")
    st.dataframe(pd.DataFrame([e for e in E.HISTORICAL_EPISODES
                               if "SVB" in e["name"]]).rename(columns={
        "name": "เหตุการณ์", "move": "ขนาด", "precursor": "สิ่งที่นำหน้า",
        "lesson": "บทเรียน"}), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# หน้า: ข่าวสาร (rule-based mapping — ไม่อ้างความแม่น ML)
# ---------------------------------------------------------------------------
def page_news():
    st.caption("จับคู่ข่าว→โมเดลด้วย 'กฎคำสำคัญแบบเปิดเผย' (ดูรายการคำได้ข้างล่าง) — "
               "โปร่งใสแต่หยาบ: พาดหัวเชิงปฏิเสธ/เสียดสีอาจจับผิด และ v3 ยังคง"
               "*ไม่ให้คะแนน sentiment* เพราะเพดานความแม่นต่ำ (เหตุผลเดิมจาก v1)")
    default_feeds = "https://www.federalreserve.gov/feeds/press_all.xml"
    feeds = st.text_area("RSS feeds (บรรทัดละ 1 URL)", default_feeds, height=70)
    if st.button("ดึงและวิเคราะห์หัวข้อข่าว"):
        try:
            import feedparser
        except ImportError:
            st.error("ต้องติดตั้งก่อน: pip install feedparser")
            return
        max_sev, rows = 0, []
        for url in [u.strip() for u in feeds.splitlines() if u.strip()]:
            try:
                d = feedparser.parse(url)
                for e in d.entries[:12]:
                    title = e.get("title", "")
                    c = MT.classify_news(title)
                    max_sev = max(max_sev, c["severity"])
                    rows.append({"หัวข้อ": title,
                                 "โมเดลที่เกี่ยว": ", ".join(
                                     M.MODEL_DEFS[m]["th"] for m in c["models"]) or "—",
                                 "ความรุนแรง": c["severity"],
                                 "ลิงก์": e.get("link", "")})
            except Exception as ex:
                st.error(f"ดึง {url} ไม่ได้: {ex}")
        st.session_state["news_max_sev"] = max_sev
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)
            st.write(f"ความรุนแรงสูงสุดรอบนี้: **{max_sev}** "
                     f"(เกณฑ์เรียกประชุม ≥ {MT.NEWS_TRIGGER_SEVERITY})")
    with st.expander("🔍 กฎคำสำคัญที่ใช้ (โปร่งใส แก้ได้ใน meeting.py)"):
        st.write({M.MODEL_DEFS[k]["th"]: v for k, v in MT.MODEL_KEYWORDS.items()})
        st.write({f"ระดับ {lvl}": kws for lvl, kws in MT.SEVERITY_RULES})


# ---------------------------------------------------------------------------
# หน้า: ห้องประชุม AI
# ---------------------------------------------------------------------------
def page_meeting():
    st.caption("**ความจริงที่ต้องรู้ก่อนใช้:** 'AI หลายตัว' คือโมเดลภาษาตัวเดียว"
               "เล่นหลายบทบาท — ความเห็น *ไม่อิสระทางสถิติ* (ผิดก็มักผิดทางเดียวกัน) "
               "คุณค่าคือโครงสร้างบังคับให้มีข้อโต้แย้ง ไม่ใช่ฉันทามติผู้เชี่ยวชาญอิสระ "
               "| กติกาเหล็ก: AI ห้ามคิดเลข/ตั้งราคาเป้าเอง — ลงมติได้แค่ เห็นด้วย/"
               "คัดค้าน/งดออกเสียง ต่อสัญญาณที่ engine คำนวณแล้ว")
    ev_txt = st.text_input("เวลาประกาศตัวเลขสำคัญ (ISO เช่น 2026-07-20 19:30, "
                           "คั่นด้วย ; ) — ระบบไม่เดาปฏิทินเอง", "")
    events = []
    for tok in [t.strip() for t in ev_txt.split(";") if t.strip()]:
        try:
            events.append(datetime.fromisoformat(tok))
        except ValueError:
            st.warning(f"อ่านเวลาไม่ได้: {tok}")
    trig = MT.should_convene(deltas, st.session_state.get("news_max_sev", 0),
                             events, datetime.now())
    if trig["convene"]:
        st.warning("เงื่อนไขเปิดประชุมทำงาน:\n" + "\n".join(
            f"- {r}" for r in trig["reasons"]))
    else:
        st.success(f"ยังไม่เข้าเงื่อนไขอัตโนมัติ (โมเดลขยับ ≥{MT.DELTA_TRIGGER_PTS:.0f} "
                   f"จุด / ข่าว ≥{MT.NEWS_TRIGGER_SEVERITY} / ±{MT.EVENT_WINDOW_HOURS} ชม. "
                   "รอบตัวเลขสำคัญ) — เปิดเองได้ด้านล่าง")
    ids = [p["id"] for p in MT.PERSONAS if p["id"] != "chair"]
    labels = {p["id"]: p["th"] for p in MT.PERSONAS}
    panel = st.multiselect("เลือกผู้เข้าประชุม (ค่าตั้งต้น 5 — เพิ่มได้ถึง 11+ประธาน "
                           "แต่จำไว้: มากขึ้น = ค่า token มากขึ้น ไม่ใช่ความเห็น"
                           "อิสระมากขึ้น)", ids, default=MT.DEFAULT_PANEL,
                           format_func=lambda i: labels[i])
    api_key = st.text_input("Anthropic API key", type="password")
    context = {
        "as_of": str(datetime.now().date()),
        "data_mode": "DEMO (สังเคราะห์)" if is_demo else "LIVE",
        "crisis_score": None if composite != composite else round(composite, 1),
        "model_scores": {m["th"]: m["score"] for m in mscores.values()},
        "model_deltas_wk": {mscores[k]["th"]: deltas.get(k) for k in deltas},
        "warnings": [w["msg"] for w in warnings],
        "signals_pending": [
            {k: s[k] for k in ("asset", "side", "model", "strength", "entry",
                               "sl", "tp", "rsi14", "trend")}
            for s in sig["signals"]],
        "conflicts": [c["asset"] for c in sig["conflicts"]],
    }
    with st.expander("ข้อมูลที่ส่งให้ AI (โปร่งใส)"):
        st.code(json.dumps(context, ensure_ascii=False, indent=2))
    if st.button("🏛️ เปิดประชุม (3 API calls)"):
        if not api_key:
            st.error("ต้องใส่ Anthropic API key")
        elif not sig["signals"] and not warnings:
            st.info("ไม่มีสัญญาณ/เตือนให้ลงมติ — ประชุมไปก็ไม่มีวาระ")
        else:
            try:
                import anthropic
            except ImportError:
                st.error("ต้องติดตั้งก่อน: pip install anthropic")
                return
            client = anthropic.Anthropic(api_key=api_key)
            msgs = []
            try:
                with st.spinner("รอบ 1: แถลงมุมมอง..."):
                    msgs.append({"role": "user", "content": MT.build_round1_prompt(
                        panel, json.dumps(context, ensure_ascii=False))})
                    r1 = client.messages.create(model="claude-sonnet-4-6",
                                                max_tokens=1800, messages=msgs)
                    t1 = "".join(b.text for b in r1.content if b.type == "text")
                    msgs.append({"role": "assistant", "content": t1})
                with st.spinner("รอบ 2: โต้แย้ง..."):
                    msgs.append({"role": "user", "content": MT.build_round2_prompt()})
                    r2 = client.messages.create(model="claude-sonnet-4-6",
                                                max_tokens=1200, messages=msgs)
                    t2 = "".join(b.text for b in r2.content if b.type == "text")
                    msgs.append({"role": "assistant", "content": t2})
                with st.spinner("ประธานสรุปมติ..."):
                    msgs.append({"role": "user", "content": MT.build_chair_prompt()})
                    r3 = client.messages.create(model="claude-sonnet-4-6",
                                                max_tokens=1000, messages=msgs)
                    t3 = "".join(b.text for b in r3.content if b.type == "text")
                st.session_state["meeting"] = [("รอบ 1 — แถลงมุมมอง", t1),
                                               ("รอบ 2 — โต้แย้ง", t2),
                                               ("มติประธาน", t3)]
            except Exception as e:
                st.error(f"เรียก API ไม่สำเร็จ: {e}")
    for title, body in st.session_state.get("meeting", []):
        with st.expander(title, expanded=(title == "มติประธาน")):
            st.markdown(body)
    if st.session_state.get("meeting"):
        st.caption("มติที่ประชุม = ความเห็นเชิงคุณภาพจากโมเดลเดียวเล่นหลายบท "
                   "ต่อสัญญาณที่คำนวณด้วยกฎ — ไม่ใช่คำแนะนำการลงทุน และไม่เพิ่ม"
                   "ความน่าจะเป็นถูกของสัญญาณ")


# ---------------------------------------------------------------------------
# หน้า: ข้อมูลมหภาค (Curve / Stress / Regime — ย่อจาก v2)
# ---------------------------------------------------------------------------
def page_macro():
    tC, tS, tR = st.tabs(["Yield Curve", "Stress Monitor", "Regime"])
    with tC:
        c1, c2, c3 = st.columns(3)
        c1.metric("10Y-3M", f"{spread_latest:+.2f}%" if spread_latest == spread_latest
                  else "n/a", "INVERTED" if spread_latest < 0 else "ปกติ",
                  delta_color="inverse" if spread_latest < 0 else "normal")
        t2 = S("T10Y2Y")
        c2.metric("10Y-2Y", f"{float(t2.iloc[-1]):+.2f}%" if len(t2) else "n/a")
        c3.metric("P(recession 12 เดือน)",
                  f"{rec_prob:.0%}" if rec_prob == rec_prob else "n/a")
        if len(t10y3m):
            f = go.Figure()
            f.add_scatter(x=t10y3m.index, y=t10y3m.values, name="10Y-3M")
            if len(t2):
                f.add_scatter(x=t2.index, y=t2.values, name="10Y-2Y")
            f.add_hline(y=0, line_dash="dot")
            f.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(f, use_container_width=True)
        with st.expander("💡 อธิบายแบบง่าย"):
            st.markdown(X.curve_explainer(rec_prob, spread_latest, inv_days))
    with tS:
        rows = []
        for sid in ["STLFSI4", "NFCI", "BAMLH0A0HYM2", "BAMLC0A0CM", "VIXCLS",
                    "DCOILWTICO", "T5YIE"]:
            s = S(sid)
            if len(s):
                rows.append({"ตัวชี้วัด": D.FRED_SERIES[sid],
                             "ล่าสุด": round(float(s.iloc[-1]), 2),
                             "z": round(E.zscore_of_latest(s), 2),
                             "pct": round(E.percentile_of_latest(s), 0)})
        if len(move) > 100:
            rows.append({"ตัวชี้วัด": "MOVE", "ล่าสุด": round(float(move.iloc[-1]), 1),
                         "z": round(E.zscore_of_latest(move), 2),
                         "pct": round(E.percentile_of_latest(move), 0)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("ตัวชี้วัดเหล่านี้เป็น coincident (พร้อมเหตุการณ์) — "
                   "เทอร์โมมิเตอร์ ไม่ใช่เครื่องพยากรณ์")
        with st.expander("💡 percentile / z-score อ่านยังไง"):
            st.markdown(X.macro_explainer(subs["stress"],
                                          E.zscore_of_latest(S("STLFSI4"))))
    with tR:
        y10 = S("DGS10")
        if len(y10) > 400:
            dy = y10.resample("W-FRI").last().diff().dropna()
            try:
                res = E.fit_hmm_2state(dy.values)
                hv = res.high_vol_state
                p_now = float(res.filtered[-1, hv])
                st.metric("P(High-vol regime) — real-time", f"{p_now:.0%}")
                f = go.Figure()
                f.add_scatter(x=dy.index, y=res.smoothed[:, hv], name="smoothed (ย้อนหลัง)")
                f.add_scatter(x=dy.index, y=res.filtered[:, hv], name="filtered (real-time)",
                              line=dict(dash="dot"))
                f.update_layout(height=300, yaxis_range=[0, 1],
                                margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(f, use_container_width=True)
                with st.expander("💡 อธิบายแบบง่าย"):
                    st.markdown(X.regime_explainer(p_now, res.sigma[1 - hv] * 100,
                                                   res.sigma[hv] * 100))
            except Exception as e:
                st.error(f"HMM ไม่ converge: {e}")


# ---------------------------------------------------------------------------
# หน้า: Trend สินทรัพย์ / Trade Log / Sentiment / คู่มือ (พอร์ตจาก v2)
# ---------------------------------------------------------------------------
def page_trend():
    st.caption("บริบทแนวโน้ม — ไม่ใช่สัญญาณซื้อขาย (<3% ของ day traders กำไร"
               "สม่ำเสมอหลังต้นทุน; Barber et al. 2020)")
    if len(prices) > 260:
        ma200 = prices.rolling(200).mean()
        mom12 = prices.pct_change(252)
        vol20 = prices.pct_change().rolling(20).std() * math.sqrt(252) * 100
        rows = []
        for t in prices.columns:
            c = prices[t].dropna()
            if len(c) < 260:
                continue
            rows.append({"สินทรัพย์": D.YF_ASSETS.get(t, t),
                         "ราคา": round(float(c.iloc[-1]), 2),
                         "เทียบ 200DMA": "เหนือ" if c.iloc[-1] >= ma200[t].iloc[-1] else "ใต้",
                         "โมเมนตัม 12ด.": f"{mom12[t].iloc[-1]:+.1%}",
                         "Vol 20d (ปี)": f"{vol20[t].iloc[-1]:.0f}%"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if breadth == breadth:
            st.metric("Risk-off breadth (% ใต้ 200DMA)", f"{breadth:.0f}%")
        with st.expander("💡 อ่านตารางนี้ยังไง"):
            st.markdown(X.trend_explainer(breadth))
    else:
        st.info("ไม่มีข้อมูลราคา")


def page_tradelog():
    st.caption("อัปโหลด CSV (คอลัมน์ `pnl`) — โมดูลกันหลอกตัวเองจาก sample เล็ก "
               "รวมถึง journal จากหน้า 'สัญญาณเทรด' หลังกรอกผลจริง")
    up = st.file_uploader("trade log CSV", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            if "pnl" not in df.columns:
                st.error("ไม่พบคอลัมน์ 'pnl'")
                return
            r = E.trade_log_report(df)
            c = st.columns(5)
            c[0].metric("จำนวนเทรด", r["n"])
            c[1].metric("Win rate", f"{r['win_rate']:.0%}",
                        f"CI95: {r['ci_low']:.0%}-{r['ci_high']:.0%}")
            pf = "∞" if math.isinf(r["profit_factor"]) else f"{r['profit_factor']:.2f}"
            c[2].metric("Profit factor", pf)
            c[3].metric("Expectancy", f"{r['expectancy']:.2f}")
            c[4].metric("PSR", f"{r['psr']:.0%}" if r["psr"] == r["psr"] else "n/a")
            st.info(r["verdict"])
            st.markdown("**💡 แปลผลแบบภาษาคน**")
            st.markdown(X.interpret_trade_log(r))
        except Exception as e:
            st.error(f"อ่านไฟล์ไม่ได้: {e}")
    with st.expander("ต้องเทรดกี่ครั้งถึงเชื่อ win rate ได้"):
        moe = st.slider("ยอมรับคลาดเคลื่อน (± จุด)", 1, 15, 5) / 100
        st.write(f"ต้องการประมาณ **{E.required_n(moe):,} เทรด** (95% conf, p=0.5)")


def page_sentiment():
    st.caption("AAII ระบุเองว่า survey 'does not predict future market direction' — "
               "ใช้เฉพาะ extreme readings เป็นบริบท contrarian 6-12 เดือน "
               "อัปโหลด CSV: date,bullish,bearish")
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
            f = go.Figure(go.Scatter(x=a["date"], y=a["spread"]))
            f.add_hline(y=30, line_dash="dot"); f.add_hline(y=-30, line_dash="dot")
            f.update_layout(title="AAII Bull-Bear (โซน extreme ±30pp)", height=320,
                            margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(f, use_container_width=True)
            last = float(a["spread"].iloc[-1])
            if last >= 30:
                st.warning(f"Extreme bullish ({last:.0f}pp) — บริบท contrarian เชิงลบ "
                           "6-12 เดือนตามอดีต (ไม่ deterministic)")
            elif last <= -30:
                st.info(f"Extreme bearish ({last:.0f}pp) — บริบท contrarian เชิงบวก "
                        "6-12 เดือนตามอดีต (ไม่ deterministic)")
            else:
                st.write(f"ล่าสุด {last:.0f}pp — ไม่ extreme, ไม่มีนัยใช้งาน")
        except Exception as e:
            st.error(f"อ่านไฟล์ไม่ได้ (ต้องมี date,bullish,bearish): {e}")


def page_glossary():
    st.subheader("คู่มืออ่านค่า — อธิบายทุกตัวเลขแบบภาษาคน")
    st.caption("ทุกคำมี 3 ส่วน: คืออะไร / วิธีอ่าน / **สิ่งที่มัน *ไม่ได้* บอก**")
    for term, d in X.GLOSSARY.items():
        with st.expander(term):
            st.markdown(f"**คืออะไร:** {d['what']}")
            st.markdown(f"**วิธีอ่าน:** {d['read']}")
            st.markdown(f"**สิ่งที่มัน *ไม่ได้* บอก:** {d['not']}")


ROUTES = {"ภาพรวม": page_overview, "โมเดลทำกำไร (6)": page_models,
          "สัญญาณเทรด": page_signals, "จำลองสถานการณ์": page_scenario,
          "วิกฤตแบงก์รัน": page_bankrun, "ข่าวสาร": page_news,
          "ห้องประชุม AI": page_meeting, "ข้อมูลมหภาค": page_macro,
          "Trend สินทรัพย์": page_trend, "Trade Log & สถิติ": page_tradelog,
          "Sentiment": page_sentiment, "คู่มืออ่านค่า": page_glossary}

st.title(page)
ROUTES[page]()
st.divider()
st.caption(DISCLAIMER + " | โหมด: " + ("DEMO (สังเคราะห์)" if is_demo else "LIVE"))
