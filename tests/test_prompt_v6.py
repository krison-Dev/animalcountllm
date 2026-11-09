"""เทสต์ prompt v6 (กรอบรายตัว) · รันได้โดยไม่มีเน็ต ไม่มีโมเดล"""
import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from app.llm import prompt_v5, prompt_v6
from app.llm.client import LLMResult, VisionLLM, look_prompt, parse, valid_boxes
from app.schemas import LLMVerdict, SpeciesTally


def _verdict(text):
    v, err = parse(text)
    assert err is None
    return v


def test_ค่าเริ่มต้นเป็น_v5_ไม่มีกรอบ_v6_ต้องเปิดเอง(monkeypatch):
    """Toy สั่งถอดกรอบออก 2026-11-09 · ค่าที่ไม่รู้จักก็ต้องได้ v5 ไม่ใช่ได้กรอบโดยไม่ตั้งใจ"""
    monkeypatch.delenv("ELEPHANT_PROMPT", raising=False)
    assert look_prompt() is prompt_v5
    monkeypatch.setenv("ELEPHANT_PROMPT", "v99")
    assert look_prompt() is prompt_v5
    monkeypatch.setenv("ELEPHANT_PROMPT", "v6")
    assert look_prompt() is prompt_v6


def test_v6_ใช้_system_เดียวกับ_v5_ทุกตัวอักษร():
    assert prompt_v6.SYSTEM == prompt_v5.SYSTEM
    _, user = prompt_v6.build(640, 512)
    assert "0-1000 scale" in user and '"boxes"' in user
    assert '"species":"elephant"' not in user          # ไม่มีค่าจริงให้ลอก


def test_กรอบผิดรูปถูกทิ้งทีละกรอบ_ไม่ใช่ทั้งชุด():
    got = valid_boxes([[100, 200, 300, 400],      # ใช้ได้
                       [100, 200, 300],           # ขาดตัวเลข
                       [300, 200, 100, 400],      # กลับด้าน
                       [100, 200, 300, 1200],     # เกินขอบ: ไม่หนีบให้
                       [100, 200, 100, 400],      # แบนเป็นเส้น
                       ["a", 1, 2, 3], [True, 1, 2, 3], "x"])
    assert got == [[100.0, 200.0, 300.0, 400.0]]
    assert valid_boxes(None) == [] and valid_boxes({"x": 1}) == []


def test_กรอบเพี้ยน_จำนวนช้างต้องไม่หาย():
    v = _verdict('{"animals":[{"species":"elephant","count":3,"confidence":0.9}],'
                 '"boxes":[[0,0,2000,2000],[10,10,20,20]]}')
    assert v.animals[0].count == 3                 # จำนวนมาจาก count ไม่ใช่จากจำนวนกรอบ
    assert v.boxes == [[10.0, 10.0, 20.0, 20.0]]


def test_กรอบลอยๆ_โดยไม่มีสัตว์ถูกนับ_ถูกทิ้ง():
    v = _verdict('{"animals":[],"boxes":[[100,100,200,200]]}')
    assert v.animals == [] and v.boxes == []


def test_ค่าเริ่มต้น_v5_ไม่ส่งกรอบ_แม้โมเดลพิมพ์มาเอง(monkeypatch):
    def fake(self, *a, **k):
        return LLMResult(LLMVerdict(animals=[SpeciesTally(species="elephant", count=1, confidence=0.9)],
                                    boxes=[[1, 1, 50, 50]]), "raw", "stop", 1, 1.0)
    monkeypatch.setattr(VisionLLM, "_invoke", fake)
    monkeypatch.delenv("ELEPHANT_PROMPT", raising=False)
    res = VisionLLM().look("x", 10, 10)
    assert res.verdict.animals[0].count == 1 and res.verdict.boxes == []


def test_ตัดคนออกหมดแล้ว_กรอบต้องหายด้วย(monkeypatch):
    monkeypatch.setenv("ELEPHANT_PROMPT", "v6")
    def fake(self, *a, **k):
        return LLMResult(LLMVerdict(animals=[SpeciesTally(species="human", count=1, confidence=0.9)],
                                    boxes=[[1, 1, 50, 50]]), "raw", "stop", 1, 1.0)
    monkeypatch.setattr(VisionLLM, "_invoke", fake)
    res = VisionLLM().look("x", 10, 10)
    assert res.verdict.animals == [] and res.verdict.boxes == []


def _png(w, h):
    b = io.BytesIO()
    Image.new("L", (w, h), 40).save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


def test_endpoint_แปลงกรอบเป็นพิกเซลของภาพต้นฉบับ_ไม่ใช่ภาพที่ขยาย(monkeypatch):
    """ภาพ 400x300 ถูกขยายก่อนส่งเข้าโมเดล · กรอบต้องกลับมาเป็นพิกเซลของ 400x300"""
    from app import main

    def fake_look(self, image_b64, w, h, camera_id="unknown", image_hash=""):
        assert min(w, h) >= 960                      # โมเดลเห็นภาพที่ขยายแล้ว
        return LLMResult(LLMVerdict(
            animals=[SpeciesTally(species="elephant", count=2, confidence=0.8)],
            boxes=[[250, 500, 500, 1000], [0, 0, 100, 100]]), "raw", "stop", 30, 10.0)

    monkeypatch.setattr(VisionLLM, "look", fake_look)
    monkeypatch.setattr(main, "API_KEY", "")
    monkeypatch.setenv("ELEPHANT_PROMPT", "v6")
    r = TestClient(main.app).post("/v1/frames", json={"camera_id": "t-v6-box", "image_base64": _png(400, 300)})
    assert r.status_code == 200
    j = r.json()
    assert j["raw"]["counts"] == {"elephant": 2}
    dets = j["raw"]["detections"]
    assert [d["bbox"] for d in dets] == [[100, 150, 100, 150], [0, 0, 40, 30]]
    assert dets[0]["area_px"] == 100 * 150 and dets[0]["aspect"] == round(100 / 150, 3)
    # detection_confidence นิยามว่ามาจาก CV · กรอบจากโมเดลห้ามแอบเอาค่าของโมเดลมาใส่
    assert all(d["detection_confidence"] == 0.0 for d in dets)
    # overall_confidence ไม่ขยับเพราะมีกรอบ (โหมด llm ไม่ได้คูณกับ detection_confidence)
    assert j["raw"]["overall_confidence"] == 0.8
