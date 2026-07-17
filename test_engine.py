"""ทดสอบตรรกะคำนวณแบบ offline (ไม่ต้องใช้อินเทอร์เน็ต): python3 test_engine.py"""
import numpy as np
import pandas as pd

import engine as E

ok = 0
fail = 0

def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"PASS  {name} {detail}")
    else:
        fail += 1
        print(f"FAIL  {name} {detail}")

# 1) probit ตรงตาราง NY Fed
table = [(1.21,.05),(0.76,.10),(0.46,.15),(0.22,.20),(0.02,.25),(-0.17,.30),
         (-0.50,.40),(-0.82,.50),(-1.13,.60),(-1.46,.70),(-1.85,.80),(-2.40,.90)]
err = max(abs(E.recession_probability(s) - p) for s, p in table)
check("probit_matches_nyfed_table", err < 0.005, f"max_err={err:.4f}")
check("probit_nan_safe", np.isnan(E.recession_probability(float('nan'))))

# 2) Wilson CI — เคสผู้ใช้จริง: 5 ชนะจาก 10 (win rate 50%)
lo, hi = E.wilson_ci(5, 10)
check("wilson_5of10_wide", lo < 0.30 and hi > 0.70, f"CI=({lo:.2f},{hi:.2f})")
lo2, hi2 = E.wilson_ci(55, 100)
check("wilson_narrows_with_n", (hi2 - lo2) < (hi - lo), f"CI100=({lo2:.2f},{hi2:.2f})")
check("required_n_~385", 380 <= E.required_n(0.05, 0.95) <= 390, f"n={E.required_n()}")
check("required_n_10pct_~97", 95 <= E.required_n(0.10, 0.95) <= 100,
      f"n={E.required_n(0.10)}")

# 3) Profit factor — สร้างชุดที่ PF=0.65 พอดี (gross win 65, gross loss 100)
pnl = pd.Series([13.0]*5 + [-20.0]*5)
pf = E.profit_factor(pnl)
check("profit_factor_0.65", abs(pf - 0.65) < 1e-9, f"pf={pf}")
check("expectancy_negative", E.expectancy(pnl) < 0)
rep = E.trade_log_report(pd.DataFrame({"pnl": pnl}))
check("report_small_n_verdict", "เล็กเกิน" in rep["verdict"], rep["verdict"][:60])
check("report_winrate_50", abs(rep["win_rate"] - 0.5) < 1e-9)

# 4) PSR — ผลตอบแทนบวกชัดเจนควรได้ PSR สูง, สุ่มรอบศูนย์ควรกลางๆ
rng = np.random.default_rng(1)
good = pd.Series(rng.normal(0.5, 1.0, 300))
noise = pd.Series(rng.normal(0.0, 1.0, 300))
check("psr_good_high", E.probabilistic_sharpe_ratio(good) > 0.99,
      f"psr={E.probabilistic_sharpe_ratio(good):.3f}")
p_noise = E.probabilistic_sharpe_ratio(noise)
check("psr_noise_mid", 0.05 < p_noise < 0.95, f"psr={p_noise:.3f}")

# 5) HMM — synthetic regime switching: ต้อง recover sigma ต่างกันชัด
rng = np.random.default_rng(42)
seg = []
truth = []
state = 0
for _ in range(12):
    length = rng.integers(40, 90)
    sd = 0.5 if state == 0 else 2.0
    seg.append(rng.normal(0, sd, length))
    truth.append(np.full(length, state))
    state = 1 - state
x = np.concatenate(seg); truth = np.concatenate(truth)
res = E.fit_hmm_2state(x)
check("hmm_converged", res.converged)
check("hmm_sigma_separation", res.sigma[res.high_vol_state] > 2.5 * res.sigma[1 - res.high_vol_state],
      f"sigmas={np.round(res.sigma,2)}")
pred = (res.smoothed[:, res.high_vol_state] > 0.5).astype(int)
acc = max((pred == truth).mean(), (pred == 1 - truth).mean())
check("hmm_smoothed_accuracy>0.85", acc > 0.85, f"acc={acc:.3f}")
# filtered (real-time) ควรแม่นน้อยกว่าหรือเท่ากับ smoothed => สะท้อน lag จริง
predf = (res.filtered[:, res.high_vol_state] > 0.5).astype(int)
accf = max((predf == truth).mean(), (predf == 1 - truth).mean())
check("hmm_filtered<=smoothed", accf <= acc + 0.02, f"filtered={accf:.3f} smoothed={acc:.3f}")

# 6) Composite — equal weight, ข้าม NaN
score, used = E.composite_crisis_score({"curve": 40, "stress": 60, "credit": float("nan"),
                                        "vol": 80, "breadth": 20})
check("composite_mean_skips_nan", abs(score - 50.0) < 1e-9 and len(used) == 4,
      f"score={score}")

# 7) percentile helpers
s = pd.Series(np.arange(100, dtype=float))
check("percentile_latest_100", abs(E.percentile_of_latest(s) - 100.0) < 1e-9)
check("zscore_latest_positive", E.zscore_of_latest(s) > 1.5)
ph = E.percentile_history(s, min_window=30)
check("percentile_history_no_lookahead", np.isnan(ph.iloc[10]) and ph.iloc[-1] == 100.0)

# 8) scenario math — duration 7, convexity 60, +100bp ~ -6.7%
imp = E.bond_price_impact(7.0, 60.0, 100)
check("scenario_+100bp", -7.1 < imp < -6.5, f"impact={imp:.2f}%")
imp2 = E.bond_price_impact(7.0, 60.0, -100)
check("scenario_-100bp_positive", 7.0 < imp2 < 7.6, f"impact={imp2:.2f}%")

# 9) warnings — cap 3, เรียง tier, false-alarm text
W = E.build_warnings(composite=75, spread_10y3m=-0.5, stress_pct=95, hy_z=2.5,
                     move_pct=95, inverted_days=40)
check("warnings_capped_3", len(W) == 3, f"n={len(W)}")
check("warnings_sorted", W[0]["tier"] <= W[-1]["tier"])
check("warnings_false_alarm_text", any("false alarm" in w["msg"] for w in W))
W0 = E.build_warnings(float("nan"), 1.5, 20, 0.1, 30, 0)
check("warnings_quiet_when_calm", len(W0) == 0)

print(f"\n== {ok} passed, {fail} failed ==")
raise SystemExit(1 if fail else 0)
