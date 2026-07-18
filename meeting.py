"""
meeting.py — ห้องประชุม AI + การจับคู่ข่าวเข้าโมเดล (ส่วนตรรกะล้วน ทดสอบได้)

ความจริงที่ต้องบอกผู้ใช้ตรงๆ (แสดงบนหน้าจอด้วย):
1. "AI 10-12 ตัว" ในทางปฏิบัติคือโมเดลภาษาตัวเดียวเล่นหลายบทบาท —
   ความเห็นจึง *ไม่อิสระทางสถิติ* (ผิดก็มักผิดไปทางเดียวกัน)
   คุณค่าของห้องประชุมคือ 'โครงสร้างบังคับให้มีข้อโต้แย้ง' ไม่ใช่ฉันทามติอิสระ
2. กติกาเหล็ก: AI ห้ามคิดเลขใหม่/ตั้งราคาเป้าเอง — ทำได้แค่ เห็นด้วย/คัดค้าน/
   งดออกเสียง ต่อสัญญาณที่ engine คำนวณแล้ว พร้อมเหตุผล
3. ค่าใช้จ่าย: จำนวน persona มากขึ้น = token มากขึ้น แต่ *ไม่ได้* เพิ่มความ
   หลากหลายของข้อมูลจริง — ค่าตั้งต้นจึงเป็น 5 ตัว (เปิดถึง 12 ได้)
4. ความรุนแรงข่าว = กฎคำสำคัญแบบเปิดเผย (ไม่ใช่ ML ที่อ้างความแม่นยำ)
"""
from __future__ import annotations

from datetime import datetime, timedelta

DELTA_TRIGGER_PTS = 6.0      # โมเดลขยับเกิน 6 จุด (สเปกผู้ใช้)
NEWS_TRIGGER_SEVERITY = 50   # ข่าวรุนแรง >= 50 (สเปกผู้ใช้)
EVENT_WINDOW_HOURS = 24      # ก่อน/หลังตัวเลขสำคัญ 24 ชม.

PERSONAS: list[dict] = [
    {"id": "macro", "th": "นักมหภาค", "role": "อ่านภาพดอกเบี้ย/เงินเฟ้อ/วัฏจักร"},
    {"id": "rates", "th": "เทรดเดอร์ตราสารหนี้", "role": "curve, duration, ประมูล"},
    {"id": "credit", "th": "นักวิเคราะห์เครดิต", "role": "spread, default cycle"},
    {"id": "liquidity", "th": "ผู้เชี่ยวชาญสภาพคล่อง/ระบบธนาคาร",
     "role": "repo, เงินฝาก, หน้าต่างกู้ Fed"},
    {"id": "technical", "th": "นักเทคนิคอล", "role": "แนวโน้ม/โมเมนตัมที่เห็นในราคา"},
    {"id": "quant", "th": "ควอนต์ขี้สงสัย",
     "role": "ตรวจขนาดตัวอย่าง/ความบังเอิญ/overfitting — ต้องท้วงถ้าหลักฐานอ่อน"},
    {"id": "risk", "th": "ผู้จัดการความเสี่ยง",
     "role": "ขนาด position, ความเสียหายกรณีผิด, มีสิทธิ์คัดค้านเด็ดขาด"},
    {"id": "devil", "th": "Devil's Advocate",
     "role": "ถูกบังคับให้แย้งมติเสียงข้างมากอย่างดีที่สุด ห้ามเห็นด้วยง่ายๆ"},
    {"id": "historian", "th": "นักประวัติศาสตร์การเงิน",
     "role": "เทียบ 2013/2019/2020/2022/2023 — ครั้งนี้เหมือน/ต่างตรงไหน"},
    {"id": "behavior", "th": "นักพฤติกรรมตลาด", "role": "positioning, crowding, sentiment"},
    {"id": "fedwatch", "th": "Fed watcher", "role": "การสื่อสารและ reaction function ของ Fed"},
    {"id": "chair", "th": "ประธาน (สรุปมติ)",
     "role": "นับเสียง สรุปข้อโต้แย้งที่ดีที่สุดของทั้งสองฝั่ง ไม่กลบเสียงข้างน้อย"},
]

DEFAULT_PANEL = ["macro", "credit", "quant", "risk", "devil"]


# ---------------------------------------------------------------------------
# ข่าว → โมเดล + ระดับความรุนแรง (rule-based โปร่งใส)
# ---------------------------------------------------------------------------

MODEL_KEYWORDS: dict[str, list[str]] = {
    "inflation_oil": ["oil", "opec", "crude", "energy", "cpi", "inflation",
                      "น้ำมัน", "เงินเฟ้อ", "gasoline"],
    "yield_shock": ["yield", "treasury", "auction", "bond selloff", "10-year",
                    "term premium", "บอนด์", "ผลตอบแทนพันธบัตร", "bid-to-cover"],
    "recovery": ["soft landing", "rally", "recovery", "growth beats", "ฟื้นตัว",
                 "expansion", "pmi beats"],
    "fed_pivot": ["rate cut", "pivot", "dovish", "pause", "ลดดอกเบี้ย", "fomc",
                  "powell", "easing"],
    "credit_crisis": ["default", "downgrade", "credit", "junk", "bankruptcy",
                      "ผิดนัด", "หุ้นกู้", "spread widen", "distress"],
    "bank_run": ["bank run", "deposit", "fdic", "bank failure", "discount window",
                 "แบงก์รัน", "เงินฝากไหลออก", "ธนาคารล้ม", "bailout", "svb"],
}

SEVERITY_RULES: list[tuple[int, list[str]]] = [
    (100, ["collapse", "failure", "default", "bank run", "ล้มละลาย", "ผิดนัด",
           "แบงก์รัน", "crash", "insolvent"]),
    (75, ["emergency", "intervention", "bailout", "halt", "ฉุกเฉิน", "อุ้ม",
          "แทรกแซง", "crisis"]),
    (50, ["plunge", "surge", "spike", "inverts", "ดิ่ง", "พุ่ง", "ทะลุ",
          "widen sharply", "turmoil"]),
    (25, ["warns", "rises", "falls", "เตือน", "กังวล", "pressure"]),
]


def classify_news(title: str) -> dict:
    """จับคู่หัวข้อข่าว -> โมเดลที่เกี่ยว + ระดับความรุนแรง (0/25/50/75/100).
    เป็นกฎคำสำคัญล้วน — โปร่งใส แต่หยาบ: พาดหัวเสียดสี/ปฏิเสธข่าวจะจับผิดได้"""
    t = (title or "").lower()
    models = [m for m, kws in MODEL_KEYWORDS.items() if any(k in t for k in kws)]
    sev = 0
    for level, kws in SEVERITY_RULES:
        if any(k in t for k in kws):
            sev = level
            break
    return {"models": models, "severity": sev}


# ---------------------------------------------------------------------------
# เงื่อนไขเปิดประชุม (สเปกผู้ใช้ 3 ข้อ)
# ---------------------------------------------------------------------------

def should_convene(model_deltas: dict[str, float], max_news_severity: int,
                   event_times: list[datetime], now: datetime) -> dict:
    """คืน {"convene": bool, "reasons": [str]} — ตรวจ 3 เงื่อนไข.
    event_times: เวลาประกาศตัวเลขสำคัญ (ผู้ใช้กรอกเอง — ระบบไม่เดาปฏิทินเอง)"""
    reasons = []
    for k, d in (model_deltas or {}).items():
        if d is not None and not (isinstance(d, float) and d != d) \
                and abs(d) >= DELTA_TRIGGER_PTS:
            reasons.append(f"โมเดล {k} ขยับ {d:+.1f} จุด (เกณฑ์ ±{DELTA_TRIGGER_PTS:.0f})")
    if max_news_severity >= NEWS_TRIGGER_SEVERITY:
        reasons.append(f"ข่าวความรุนแรง {max_news_severity} (เกณฑ์ ≥{NEWS_TRIGGER_SEVERITY})")
    for t in event_times or []:
        h = (t - now).total_seconds() / 3600.0
        if -EVENT_WINDOW_HOURS <= h <= EVENT_WINDOW_HOURS:
            when = "อีก" if h >= 0 else "ผ่านมา"
            reasons.append(f"ตัวเลขสำคัญ {when} {abs(h):.0f} ชม. ({t:%d %b %H:%M})")
    return {"convene": bool(reasons), "reasons": reasons}


# ---------------------------------------------------------------------------
# สร้าง prompt (แยกจากการเรียก API เพื่อทดสอบได้)
# ---------------------------------------------------------------------------

IRON_RULES = (
    "กติกาเหล็ก (ห้ามละเมิด): (1) ใช้เฉพาะตัวเลขใน context — ห้ามคำนวณ/"
    "ประมาณ/ตั้งราคาเป้าใหม่เอง (2) ต่อแต่ละสัญญาณ ลงมติได้แค่ เห็นด้วย / "
    "คัดค้าน / งดออกเสียง พร้อมเหตุผลสั้น (3) ถ้าข้อมูลไม่พอ ให้พูดว่าไม่พอ "
    "(4) ห้ามให้คำแนะนำการลงทุนเฉพาะบุคคล (5) ตอบภาษาไทย กระชับ ตรงไปตรงมา")


def build_round1_prompt(panel_ids: list[str], context_json: str) -> str:
    ps = [p for p in PERSONAS if p["id"] in panel_ids and p["id"] != "chair"]
    roles = "\n".join(f"- {p['th']}: {p['role']}" for p in ps)
    return (f"คุณจะเล่นบทผู้เข้าประชุม {len(ps)} คนต่อไปนี้ทีละคน:\n{roles}\n\n"
            f"{IRON_RULES}\n\n"
            "ให้แต่ละคนพูด 2-4 ประโยค: มุมมองต่อสถานการณ์ + มติต่อแต่ละสัญญาณ "
            "(เห็นด้วย/คัดค้าน/งดออกเสียง + เหตุผล) "
            "Devil's Advocate ต้องแย้งเสียงข้างมาก และควอนต์ต้องท้วงเรื่อง"
            "ขนาดตัวอย่างถ้าหลักฐานอ่อน\n\n"
            f"context (ตัวเลขทั้งหมดมาจาก engine):\n{context_json}")


def build_round2_prompt() -> str:
    return ("รอบโต้แย้ง: ให้ผู้เข้าประชุมแต่ละคนตอบข้อแย้งที่แรงที่สุดที่ตนเจอ "
            "1-2 ประโยค (เปลี่ยนมติได้ถ้ายอมรับเหตุผล) " + IRON_RULES)


def build_chair_prompt() -> str:
    return ("ในบทประธาน: (1) นับมติต่อแต่ละสัญญาณ (เห็นด้วย x / คัดค้าน y / งด z) "
            "(2) สรุปข้อโต้แย้งที่ดีที่สุดของฝั่งค้านแม้เป็นเสียงข้างน้อย "
            "(3) ระบุ 'สิ่งที่จะพิสูจน์ว่ามุมมองนี้ผิด' 1-2 ข้อ "
            "(4) ย้ำว่านี่คือความเห็นจากโมเดลเดียวเล่นหลายบท ไม่ใช่ผู้เชี่ยวชาญอิสระ "
            + IRON_RULES)
