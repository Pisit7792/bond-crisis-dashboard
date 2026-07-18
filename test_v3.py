"""ทดสอบโมดูล v3 ทั้งหมดแบบ offline: python3 test_v3.py"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import data_sources as D
import models6 as M
import scenario as SC
import signals as SG
import meeting as MT

ok = fail = 0

def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1; print(f"PASS  {name} {detail}")
    else:
        fail += 1; print(f"FAIL  {name} {detail}")

b = D.demo_bundle()
data = dict(b["fred"]); data["MOVE"] = b["move"]
prices = b["prices"]

# 1) 6 โมเดล: ครบ 6, คะแนน 0-100, components แสดงได้
scores = M.score_models(data)
check("models_count", len(scores) == 6, f"n={len(scores)}")
allv = [m["score"] for m in scores.values()]
check("models_in_range", all(0 <= v <= 100 for v in allv if v == v), str(allv))
check("models_components_present",
      all(len(m["components"]) >= 2 for m in scores.values()),
      {k: len(m["components"]) for k, m in scores.items()})
# demo มีช่วงเครียด (HY ถ่าง เงินฝากไหลออก) → credit/bank_run ควรไม่เป็นศูนย์
check("bank_run_active_in_demo", scores["bank_run"]["score"] > 20,
      f"={scores['bank_run']['score']}")

# 2) ASSET_IMPACT: ทุกชื่อสินทรัพย์ต้องมีจริงใน YF_ASSETS + มี note ทุกโมเดล
names = set(D.YF_ASSETS.values())
bad = [(k, a) for k, imp in M.ASSET_IMPACT.items()
       for a in imp["benefit"] + imp["lose"] if a not in names]
check("impact_assets_exist", not bad, str(bad))
check("impact_notes", all(M.ASSET_IMPACT[k].get("note") for k in M.ASSET_IMPACT))

# 3) score_history + delta
h = M.score_history(data, "credit_crisis")
check("history_nonempty", len(h) > 30, f"len={len(h)}")
check("delta_finite", M.model_delta(h) == M.model_delta(h))  # not NaN

# 4) Scenario: reset → delta 0 ทุกโมเดล
base = {k: m["score"] for k, m in scores.items()}
r0 = SC.apply_scenario(base, dict(SC.SLIDER_BASE))
check("scenario_reset_zero", all(abs(v["delta"]) < 1e-9 for v in r0.values()))
# น้ำมัน +20% → inflation_oil ขึ้น, recovery ลง
r1 = SC.apply_scenario(base, {**SC.SLIDER_BASE, "oil_pct": 20})
check("oil_up_inflation_up", r1["inflation_oil"]["delta"] > 5,
      f"Δ={r1['inflation_oil']['delta']}")
check("oil_up_recovery_down", r1["recovery"]["delta"] < 0)
# เงินฝาก -5% → bank_run พุ่ง
r2 = SC.apply_scenario(base, {**SC.SLIDER_BASE, "depo_pct": -5})
check("deposit_out_bankrun_up", r2["bank_run"]["delta"] > 10,
      f"Δ={r2['bank_run']['delta']}")
# Fed ลด 50bp → fed_pivot ขึ้น
r3 = SC.apply_scenario(base, {**SC.SLIDER_BASE, "fed_bps": -50})
check("cut_pivot_up", r3["fed_pivot"]["delta"] > 5)
# ประมูลอ่อน (bid-to-cover 1.8) → yield_shock ขึ้น
r4 = SC.apply_scenario(base, {**SC.SLIDER_BASE, "btc_x": 1.8})
check("weak_auction_yieldshock_up", r4["yield_shock"]["delta"] > 5)
# cap 0-100
r5 = SC.apply_scenario({k: 95 for k in base},
                       {**SC.SLIDER_BASE, "hy_bps": 400, "vix_pts": 40})
check("scenario_capped", all(v["new"] <= 100 for v in r5.values()))
check("sensitivity_table_shape", SC.sensitivity_table().shape[0] == 6)

# 5) Signals — ฉาก A: โมเดลขัดแย้งกัน (credit SHORT ปะทะ recovery LONG)
idx = pd.bdate_range(end=pd.Timestamp.today(), periods=400)
L = len(idx)
up = pd.Series(np.linspace(80, 130, L) + np.random.default_rng(0).normal(0, 0.8, L), index=idx)
dn = pd.Series(np.linspace(130, 80, L) + np.random.default_rng(1).normal(0, 0.8, L), index=idx)
pxA = pd.DataFrame({"^GSPC": dn, "TLT": up, "BTC-USD": up})
scoresA = {"credit_crisis": {"th": "วิกฤตสินเชื่อ", "score": 55.0},
           "recovery": {"th": "ฟื้นตัว", "score": 40.0}}
resA = SG.build_signals(scoresA, M.ASSET_IMPACT, pxA, D.YF_ASSETS)
check("A_conflict_us500", any(c["asset"] == "US500" for c in resA["conflicts"]),
      f"conflicts={[c['asset'] for c in resA['conflicts']]}")
check("A_tlt_long_survives",
      any(s["asset"].startswith("TLT") and s["side"] == "LONG" for s in resA["signals"]))
check("A_no_us500_signal", not any(s["asset"] == "US500" for s in resA["signals"]))

# ฉาก B: โมเดลเดียวเด่น (credit 55, อื่นต่ำกว่าเกณฑ์) → สัญญาณสะอาด
scoresB = {"credit_crisis": {"th": "วิกฤตสินเชื่อ", "score": 55.0},
           "recovery": {"th": "ฟื้นตัว", "score": 30.0},
           "inflation_oil": {"th": "เงินเฟ้อ-น้ำมัน", "score": 10.0}}
resB = SG.build_signals(scoresB, M.ASSET_IMPACT, pxA, D.YF_ASSETS)
sigB = {s["asset"]: s for s in resB["signals"]}
check("B_short_us500", sigB.get("US500", {}).get("side") == "SHORT",
      f"signals={list(sigB)}")
check("B_long_tlt", sigB.get("TLT (20y+ UST ETF)", {}).get("side") == "LONG")
check("B_btc_skipped_by_trend",
      any(s["asset"] == "BTC" for s in resB["skipped"]),
      f"skipped={[s['asset'] for s in resB['skipped']]}")
check("B_below_threshold_no_gold", "XAUUSD" not in sigB)
s0 = sigB["US500"]
check("B_rr_1to2",
      abs(abs(s0["tp"] - s0["entry"]) - 2 * abs(s0["entry"] - s0["sl"])) < 1e-6)
check("B_short_levels", s0["sl"] > s0["entry"] > s0["tp"])
check("B_has_technicals",
      all(k in s0 for k in ("rsi14", "mom12m%", "dist_200dma%", "trend", "atr14≈")))
csv = SG.journal_csv(resB["signals"], "2026-07-18")
check("journal_csv", csv.startswith("date,asset") and csv.count("\n") == len(resB["signals"]) + 1)

# 6) Meeting: triggers 3 เงื่อนไข
now = datetime(2026, 7, 18, 12, 0)
r = MT.should_convene({"credit_crisis": 7.2}, 0, [], now)
check("trigger_delta", r["convene"] and "7.2" in r["reasons"][0])
r = MT.should_convene({"credit_crisis": 3.0}, 50, [], now)
check("trigger_news", r["convene"])
r = MT.should_convene({}, 0, [now + timedelta(hours=10)], now)
check("trigger_event_pre", r["convene"])
r = MT.should_convene({}, 0, [now - timedelta(hours=30)], now)
check("no_trigger_event_old", not r["convene"])
r = MT.should_convene({"a": 2.0}, 25, [], now)
check("no_trigger_quiet", not r["convene"])

# 7) News classification
c = MT.classify_news("Regional bank failure sparks deposit run fears")
check("news_bankrun_100", "bank_run" in c["models"] and c["severity"] == 100, str(c))
c = MT.classify_news("Oil surges 8% as OPEC cuts output")
check("news_oil_50", "inflation_oil" in c["models"] and c["severity"] == 50, str(c))
c = MT.classify_news("ยีลด์พันธบัตรพุ่งหลังประมูล bid-to-cover อ่อน")
check("news_thai_yield", "yield_shock" in c["models"], str(c))
c = MT.classify_news("Weather is nice today")
check("news_none", c["models"] == [] and c["severity"] == 0)

# 8) Prompts: มี iron rules + persona ครบ
p1 = MT.build_round1_prompt(MT.DEFAULT_PANEL, "{}")
check("prompt_iron_rules", "กติกาเหล็ก" in p1 and "ห้ามคำนวณ" in p1)
check("prompt_devil", "Devil" in p1)
check("personas_12", len(MT.PERSONAS) == 12)
check("chair_prompt_minority", "เสียงข้างน้อย" in MT.build_chair_prompt())

print(f"\n== {ok} passed, {fail} failed ==")
raise SystemExit(1 if fail else 0)
