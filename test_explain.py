"""ทดสอบโมดูลคำอธิบาย: python3 test_explain.py"""
import math

import explain as X

ok = fail = 0

def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1; print(f"PASS  {name} {detail}")
    else:
        fail += 1; print(f"FAIL  {name} {detail}")

NAN = float("nan")

# 1) plain_summary — เคสปกติ ต้องมีโซน, ตัวขับ, และประโยคความซื่อสัตย์
subs = {"curve": 40.0, "stress": 70.0, "credit": 55.0, "vol": 80.0, "breadth": 20.0}
s = X.plain_summary(53.0, subs, -0.4, 0.31, 25, 2, is_demo=False)
check("summary_nonempty", len(s) > 200)
check("summary_zone", "ปานกลาง" in s)
check("summary_top_driver", "ความผันผวน" in s)  # vol สูงสุด
check("summary_honesty_thermometer", "เทอร์โมมิเตอร์" in s and "หมอดู" in s)
check("summary_false_alarm", "ผิดราวครึ่งหนึ่ง" in s)
check("summary_inverted_mentioned", "inverted" in s or "กลับหัว" in s)
check("summary_recession_freq", "31 ใน 100" in s)
check("summary_zone_disclaimer", "ไม่ใช่มาตรฐานสากล" in s)

# 2) plain_summary — demo ต้องติดป้าย, NaN ต้องไม่พัง
s_demo = X.plain_summary(53.0, subs, NAN, NAN, 0, 0, is_demo=True)
check("summary_demo_flag", "DEMO" in s_demo)
s_nan = X.plain_summary(NAN, {k: NAN for k in subs}, NAN, NAN, 0, 0, False)
check("summary_all_nan_safe", "ไม่พอ" in s_nan or "ไม่มีข้อมูล" in s_nan)

# 3) zones ครอบคลุมทุกช่วง
for v, zname in [(10, "สงบ"), (45, "ปานกลาง"), (70, "ตึงตัว"), (90, "ตึงตัวมาก")]:
    check(f"zone_{v}", X.composite_zone(v)[0] == zname)
check("zone_nan", X.composite_zone(NAN)[0] == "ไม่มีข้อมูล")

# 4) interpret_percentile / zscore — ทุกช่วง + NaN + ประโยคกันตีความเกิน
for v in [10, 50, 70, 85, 97]:
    t = X.interpret_percentile(v, "X")
    check(f"pct_{v}", len(t) > 40 and "ไม่ได้บอกว่าพรุ่งนี้" in t)
check("pct_nan", "ไม่มีข้อมูล" in X.interpret_percentile(NAN))
for z in [-2.5, -0.3, 1.5, 3.0]:
    check(f"z_{z}", len(X.interpret_zscore(z)) > 30)
check("z_nan", "ไม่มีข้อมูล" in X.interpret_zscore(NAN))

# 5) explainer ทุกตัว: ไม่ว่าง + มีท่อน "ไม่ได้บอก/ไม่ใช่" + NaN-safe
c = X.curve_explainer(0.25, -0.6, 30)
check("curve_expl", "กลับหัว" in c and "ไม่ได้" in c and "25 ใน 100" in c)
check("curve_expl_nan", len(X.curve_explainer(NAN, NAN, 0)) > 100)
m = X.macro_explainer(88.0, 2.1)
check("macro_expl", "เทอร์โมมิเตอร์" in m and "88" in m)
check("macro_expl_nan", len(X.macro_explainer(NAN, NAN)) > 100)
r = X.regime_explainer(0.9, 12.0, 45.0)
check("regime_expl", "พายุ" in r and "filtered" in r and "หลัง" in r)
check("regime_expl_nan", len(X.regime_explainer(NAN, NAN, NAN)) > 100)
t = X.trend_explainer(43.0)
check("trend_expl", "200DMA" in t and "ไม่ใช่" in t and "สัญญาณซื้อขาย" in t)
check("trend_expl_nan", len(X.trend_explainer(NAN)) > 100)
sc = X.scenario_explainer(7.0, -6.7)
check("scenario_expl", "ไม้กระดก" in sc and "-6.7" in sc)

# 6) interpret_trade_log — เคสผู้ใช้จริง (n=10, PF 0.65)
rep = {"n": 10, "wins": 5, "win_rate": 0.5, "ci_low": 0.24, "ci_high": 0.76,
       "profit_factor": 0.65, "expectancy": -0.7, "psr": 0.2}
tl = X.interpret_trade_log(rep)
check("trade_coin_analogy", "เหรียญ" in tl)
check("trade_pf_meaning", "0.65 บาท" in tl and "ขาดทุนจริง" in tl)
check("trade_ci_meaning", "24%-76%" in tl or "24%" in tl)
check("trade_precommit_rule", "100 เทรด" in tl)
rep_inf = dict(rep, profit_factor=float("inf"), psr=NAN, expectancy=NAN,
               ci_low=NAN, ci_high=NAN)
check("trade_inf_nan_safe", len(X.interpret_trade_log(rep_inf)) > 50)

# 7) GLOSSARY — ทุกคำต้องมีครบ what/read/not และ not ต้องไม่ว่าง
missing = [k for k, d in X.GLOSSARY.items()
           if not all(d.get(f) for f in ("what", "read", "not"))]
check("glossary_complete", not missing, f"missing={missing}")
check("glossary_size", len(X.GLOSSARY) >= 15, f"terms={len(X.GLOSSARY)}")

print(f"\n== {ok} passed, {fail} failed ==")
raise SystemExit(1 if fail else 0)
