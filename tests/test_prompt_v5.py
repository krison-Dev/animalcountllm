"""เทสต์ prompt v5 + parse ที่ทนข้อความต่อท้าย · รันได้โดยไม่มีเน็ต ไม่มีโมเดล"""
from app.llm import prompt_v5
from app.llm.client import LLMResult, VisionLLM, image_type_of, look_prompt, parse
from app.schemas import LLMVerdict, SpeciesTally


def test_look_ถอยกลับ_v5_ได้ด้วย_env(monkeypatch):
    """ค่าเริ่มต้นเป็น v6 ตั้งแต่ 2026-11-09 · v5 ยังเรียกได้เพื่อถอยกลับ"""
    seen = {}

    def fake(self, system, user, image_b64, image_hash, prompt_version, blobs, frame):
        seen["version"], seen["system"] = prompt_version, system
        return LLMResult(LLMVerdict(animals=[SpeciesTally(species="human", count=1, confidence=0.9)]),
                         "raw", "stop", 1, 1.0)

    monkeypatch.setattr(VisionLLM, "_invoke", fake)
    monkeypatch.setenv("ELEPHANT_PROMPT", "v5")
    res = VisionLLM().look("x", 10, 10)
    assert seen["version"] == "v5"
    assert seen["system"] == prompt_v5.SYSTEM
    assert res.verdict.animals == []          # ด่านตัดคนยังทำงานกับ v5




def test_v5_ไม่บอกว่าช้างต้องเป็นก้อนขาวเสมอ():
    """ประโยคนี้ใน v4 ทำให้ thermal กลางวันจับได้ 20-40% และ 32b ตอบภาพสีว่างหมด"""
    system, _ = prompt_v5.build(640, 512)
    low = system.lower()
    assert "daytime" in low and "do not require it to be bright" in low
    assert "colour picture" in low                 # รับภาพจากกล้อง PTZ


def test_v5_ตัวอย่าง_json_ไม่มีค่าจริงให้ลอก_และไม่บอกว่าเจออะไรมาก่อน():
    system, user = prompt_v5.build(640, 512)
    assert '"species":"elephant"' not in user
    assert '"count":<whole number>' in user
    assert '"<elephant|unknown>"' in user
    low = (system + user).lower()
    for banned in ["detector found", "warm region", "has already found", "bbox"]:
        assert banned not in low


def test_parse_ทนข้อความต่อท้าย_json():
    """เจอจริงบน PGX 2026-11-09 · คำตอบถูก แต่ regex แบบโลภกวาดข้อความท้ายเข้าไปแล้วตก"""
    text = ('{"image_type":"unclear","animals":[{"species":"elephant","count":1,"confidence":0.95}]}\n\n'
            'The image provided is not a thermal image {as noted}, but a photograph.')
    v, err = parse(text)
    assert err is None
    assert [(a.species, a.count) for a in v.animals] == [("elephant", 1)]
    assert image_type_of(text) == "unclear"


def test_parse_ก้อนแรกพัง_ต้อง_degraded_ไม่ใช่ศูนย์ตัวเงียบๆ():
    """ห้ามไปหยิบก้อนข้างใน {"species":...} มาแทน ซึ่งไม่มีคีย์ animals = ไม่มีช้าง"""
    v, err = parse('{"image_type":"thermal","animals":[{"species":"elephant","count":2,"confidence":0.8}')
    assert v is None and err.startswith("bad JSON")


def test_parse_ยังรับ_markdown_fence_และข้อความนำ():
    v, err = parse('Here you go:\n```json\n{"animals":[{"species":"elephant","count":3,"confidence":0.7}]}\n```')
    assert err is None and v.animals[0].count == 3
