"""ตัวเรียก VLM · LM Studio ตอน dev · OpenRouter ตอน prod · สลับด้วย env ตัวเดียว

ทั้งสองฝั่งเป็น OpenAI-compatible เลยใช้ทางเดียวกันได้ ไม่ต้องมี if provider== ทั่วโค้ด

🔴 ไม่ใช้ agent ไม่ใช้ memory ไม่ใช้ chain ซ้อน · งานนี้เป็น pipeline เส้นตรง
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..schemas import Blob, LLMVerdict, SpeciesTally
from . import prompt_v2 as prompt
from . import prompt_v4, prompt_v5, prompt_v6

VALID = {"elephant", "cattle", "human", "other_animal", "unknown"}

# prompt ของเส้น llm · ค่าเริ่มต้น v5 (thermal กลางวัน/กลางคืน + ภาพสีจาก PTZ · ไม่มีกรอบ)
# Toy สั่งถอดกรอบออก 2026-11-09 · v6 (= v5 + กรอบรายตัว) ยังเรียกได้ด้วย ELEPHANT_PROMPT=v6
# ถอยกลับก่อนรองรับ PTZ: ELEPHANT_PROMPT=v4 · ทั้งหมดไม่ต้อง build ใหม่
# ค่าที่ไม่รู้จักใช้ค่าเริ่มต้น ไม่ใช่พัง
LOOK_PROMPTS = {"v4": prompt_v4, "v5": prompt_v5, "v6": prompt_v6}
DEFAULT_LOOK = prompt_v5
MAX_BOXES = 20


def look_prompt():
    return LOOK_PROMPTS.get(os.environ.get("ELEPHANT_PROMPT", "").strip().lower(), DEFAULT_LOOK)


def valid_boxes(raw) -> List[List[float]]:
    """กรอบ [x1,y1,x2,y2] สเกล 0-1000 ที่ใช้ได้ · ทิ้งที่ผิดรูปทีละกรอบ ไม่ทิ้งทั้งชุด

    ผิดรูป = ไม่ใช่ตัวเลขสี่ตัว · เกิน 0-1000 · กลับด้านหรือแบนเป็นเส้น
    🔴 ทิ้งกรอบ ≠ ทิ้งช้าง · จำนวนมาจาก animals[].count เสมอ ด่านนี้ไม่แตะจำนวน
    ไม่หนีบค่าที่เกินให้เข้าขอบ เพราะกรอบที่ต้องหนีบคือกรอบที่โมเดลอ่านหน่วยผิด
    หนีบแล้วจะได้กรอบที่ดูดีแต่ชี้ผิดที่
    """
    out: List[List[float]] = []
    if not isinstance(raw, list):
        return out
    for b in raw[:MAX_BOXES]:
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            continue
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in b):
            continue
        x1, y1, x2, y2 = (float(v) for v in b)
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            continue
        out.append([x1, y1, x2, y2])
    return out


@dataclass
class LLMResult:
    verdict: Optional[LLMVerdict]
    raw: str
    finish_reason: Optional[str]
    completion_tokens: Optional[int]
    latency_ms: float
    error: Optional[str] = None
    image_type: Optional[str] = None
    """โมเดลบอกเองว่าเฟรมนี้เป็น thermal / colour / unclear (v3 เท่านั้น)

    ไม่ได้เอาไป gate อะไร เก็บไว้ตรวจย้อนหลังว่าเฟรมที่พลาดเป็นภาพแบบไหน
    เส้นนี้ไม่มี numpy แล้ว จะคำนวณเองไม่ได้
    """

    @property
    def usable(self) -> bool:
        """finish_reason='length' = มันคิดจนหมดโควตา ไม่ได้ตอบจบ

        วัดจริง 2026-08-16: ปล่อย 1500 token แล้วมันคิดวนจนหลุดไปเขียนโค้ด cv2
        ใส่ช่องคำตอบ · คำตอบที่ตัดกลางคันเชื่อไม่ได้ ต้องตี degraded ไม่ใช่รับมาใช้เฉยๆ
        """
        return self.verdict is not None and self.finish_reason != "length"


class VisionLLM:
    def __init__(self) -> None:
        self.provider = os.environ.get("LLM_PROVIDER", "lmstudio")
        self.base_url = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
        self.model = os.environ.get("LLM_MODEL", "qwen3.6-35b-a3b")
        self.api_key = os.environ.get("LLM_API_KEY", "not-needed")
        # 🔴 max_tokens ต้องโตตามจำนวนก้อน ไม่ใช่ค่าคงที่
        #
        # เจอจริง 2026-08-16 กับภาพจากไซต์: 16 ก้อน · finish=length ที่ 300 token
        # แล้วทุกก้อนกลายเป็น unknown · ตอนวัดครั้งแรกใช้ภาพ 2 ก้อน (73-86 token)
        # แล้วผมเอาตัวเลขนั้นไปตั้งเป็นค่าคงที่ ซึ่งผิดตั้งแต่ต้น
        # คำตอบยาวตามจำนวนก้อนโดยตรง เพราะต้องตอบทีละก้อนตาม schema
        # v2 ตอบสรุปรายชนิด คำตอบเลยสั้นคงที่ไม่ว่าเฟรมจะรกแค่ไหน
        # v1 ตอบรายก้อน ยาวตามจำนวนก้อน แล้วทะลุโควตาที่ 16 ก้อน
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "300"))
        self.timeout = float(os.environ.get("LLM_TIMEOUT_S", "25"))
        # 🔴 ปิด thinking · วัดจริง 2026-08-16 บน qwen3.7-flash ผ่าน OpenRouter
        #   ไม่ตั้งอะไร        13.0 วิ · 801 tok · finish=length ตอบไม่จบ
        #   reasoning enabled=false  2.9 วิ ·  86 tok · finish=stop  JSON ครบ
        # เร็วขึ้น 4.5 เท่า · output token ลด 9 เท่า ซึ่งคือค่าใช้จ่ายเกือบทั้งหมด
        # ($3.29 -> $0.89 ต่อเดือนต่อกล้อง)
        #
        # ที่ลองแล้วไม่ได้ผล อย่าเสียเวลาลองซ้ำ:
        #   reasoning.effort=low            ไม่ต่างจากไม่ตั้งเลย
        #   chat_template enable_thinking   OpenRouter ไม่ส่งต่อให้ provider
        #   reasoning.max_tokens=0          provider ตอบ 400
        #   reasoning.exclude=true          🔴 กับดัก · ซ่อนความคิดจาก response
        #                                   แต่ยังคิดจริงและยังคิดเงินเต็ม
        self.no_reasoning = os.environ.get("LLM_DISABLE_REASONING", "true").lower() == "true"
        self.sample_rate = float(os.environ.get("LANGSMITH_SAMPLE_RATE", "0.05"))
        self.tracing = os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"

    # ---------------------------------------------------------------- tracing
    def _should_trace(self, anomalous: bool) -> bool:
        """เฟรมผิดปกติเก็บ 100% เฟรมปกติเก็บตาม sample rate

        860 trace/วัน/กล้อง = 26,000/เดือน ทะลุแพ็กฟรี (~5,000) ตั้งแต่เดือนแรก
        ของที่จะกลับมาดูจริงคือเฟรมที่มีปัญหา ไม่ใช่เฟรมที่ถูกอยู่แล้ว
        """
        if not self.tracing:
            return False
        return True if anomalous else random.random() < self.sample_rate

    # ---------------------------------------------------------------- call
    def look(self, image_b64: str, w: int, h: int, camera_id: str = "unknown",
             image_hash: str = "") -> LLMResult:
        """เส้น llm · ถามโมเดลตรงๆ ไม่มีชั้น CV มาบอกใบ้ก่อน

        ตัดสินโดย Toy 2026-08-17: API เส้นนี้คือ LLM ไม่ใช่ CV
        Toy 2026-11-03: นับช้างอย่างเดียว ไม่สนใจคน (ดู prompt_v4.py)
        2026-11-09: ค่าเริ่มต้นเป็น v5 (ดู prompt_v5.py) · ELEPHANT_PROMPT=v4 ถอยกลับ
        """
        p = look_prompt()
        system, user = p.build(w, h, camera_id)
        res = self._invoke(system, user, image_b64, image_hash,
                           p.PROMPT_VERSION, blobs=0, frame=f"{w}x{h}")
        if res.verdict:
            res.verdict.animals = elephant_only(res.verdict.animals)
            if not res.verdict.animals:
                res.verdict.boxes = []   # ตัดคนออกหมดแล้วไม่เหลือสัตว์ = กรอบไม่ได้ชี้ช้าง
            if p is not prompt_v6:
                res.verdict.boxes = []   # กรอบมีเฉพาะตอนเปิด v6 · โมเดลพิมพ์มาเองก็ไม่ส่งต่อ
        return res

    def classify(self, image_b64: str, blobs: List[Blob], w: int, h: int,
                 image_hash: str = "") -> LLMResult:
        """เส้น v2 · ยังอยู่เพราะโหมด cv+llm ยังเรียกได้ และเทสต์เก่ายังอ้างถึง"""
        system, user = prompt.build(blobs, w, h)
        return self._invoke(system, user, image_b64, image_hash,
                            prompt.PROMPT_VERSION, blobs=len(blobs), frame=f"{w}x{h}")

    def _invoke(self, system: str, user: str, image_b64: str, image_hash: str,
                prompt_version: str, blobs: int, frame: str) -> LLMResult:
        max_tokens = self.max_tokens
        t0 = time.perf_counter()

        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage

            kw = {}
            if self.no_reasoning:
                # LM Studio ไม่รู้จัก field นี้แต่ไม่ error มันข้ามไปเฉยๆ
                kw["extra_body"] = {"reasoning": {"enabled": False}}
            llm = ChatOpenAI(
                base_url=self.base_url, api_key=self.api_key, model=self.model,
                max_tokens=max_tokens, temperature=0, timeout=self.timeout, **kw,
            )
            msgs = [
                SystemMessage(content=system),
                HumanMessage(content=[
                    {"type": "text", "text": user},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime_of(image_b64)};base64,{image_b64}"}},
                ]),
            ]
            # metadata ไปที่ LangSmith · 🔴 ไม่ส่งภาพ base64 เข้า trace
            # payload จะบวมและกินโควตา storage เร็วกว่ากินโควตา trace
            # ผลพลอยได้: ภาพจากไซต์อนุรักษ์ไม่ต้องไปนอนบนเซิร์ฟเวอร์เจ้าอื่น
            resp = llm.invoke(msgs, config={
                "metadata": {"image_hash": image_hash, "blobs": blobs,
                             "frame": frame, "prompt_version": prompt_version},
                "run_name": "classify_frame",
            })
            dt = (time.perf_counter() - t0) * 1000
            meta = resp.response_metadata or {}
            usage = getattr(resp, "usage_metadata", None) or {}
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            verdict, err = parse(text)
            return LLMResult(
                verdict=verdict, raw=text, image_type=image_type_of(text),
                finish_reason=meta.get("finish_reason"),
                completion_tokens=usage.get("output_tokens"),
                latency_ms=round(dt, 1), error=err,
            )
        except Exception as e:
            # เน็ตหลุด / LLM ช้า / โมเดลไม่มี → ตอบ degraded พร้อมผล CV ไม่ค้างรอ
            return LLMResult(None, "", None, None,
                             round((time.perf_counter() - t0) * 1000, 1),
                             f"{type(e).__name__}: {e}")


def elephant_only(animals: List[SpeciesTally]) -> List[SpeciesTally]:
    """ด่านในโค้ดตามที่ Toy สั่ง 2026-11-03 เผื่อโมเดลไม่ทำตาม prompt

    คนถูกตัดทิ้ง · สัตว์ชนิดอื่นนับเป็นช้าง · unknown คงไว้
    คำตอบดิบก่อนผ่านด่านยังอยู่ใน llm_raw_response
    """
    out = []
    for a in animals:
        if a.species == "human":
            continue
        if a.species in ("cattle", "other_animal"):
            a = a.model_copy(update={"species": "elephant"})
        out.append(a)
    return out


def mime_of(image_b64: str) -> str:
    """เดา mime จากไบต์แรกของ base64 · เคยฝัง image/png ไว้ตายตัว

    ภาพจากไซต์เป็น JPEG แต่เราประกาศว่า png ซึ่งบาง provider ยอม บางเจ้าไม่ยอม
    ไม่ต้อง decode ทั้งก้อน หัวไม่กี่ตัวอักษรก็พอบอกได้แล้ว
    """
    head = (image_b64 or "")[:12]
    if head.startswith("/9j/"):
        return "image/jpeg"
    if head.startswith("R0lGOD"):
        return "image/gif"
    if head.startswith("UklGR"):
        return "image/webp"
    return "image/png"


def first_json(text: str) -> Tuple[Optional[dict], Optional[str]]:
    """JSON object ก้อนแรกในคำตอบ · ไม่สนข้อความที่ตามมาข้างหลัง

    🔴 เจอจริง 2026-11-09 บน PGX · qwen3-vl-30b-a3b q8 ตอบ JSON ถูกต้องครบ
    แล้วพิมพ์ต่อว่า "The image provided is not a thermal image ..." ซึ่งมีวงเล็บปีกกา
    เดิมใช้ regex `\\{.*\\}` แบบโลภ กวาดจากปีกกาแรกถึงปีกกาสุดท้าย รวมข้อความท้ายเข้าไปด้วย
    แล้ว json.loads ตก "Extra data" · เฟรมที่โมเดลตอบถูกกลายเป็น degraded ทั้งที่คำตอบอยู่ตรงนั้น
    raw_decode อ่านก้อนเดียวแล้วหยุด เลยไม่สนว่าข้างหลังมีอะไร

    🔴 ลองแค่ปีกกาตัวแรก ห้ามวนหาปีกกาตัวถัดไปเมื่อก้อนแรกพัง ตัวถัดไปคือก้อนข้างใน
    เช่น {"species":"elephant",...} ซึ่งไม่มีคีย์ animals แล้วจะกลายเป็น "ไม่มีช้าง" เงียบๆ
    ก้อนแรกพัง = degraded พร้อมเหตุผล ไม่ใช่ศูนย์ตัว
    """
    if not text:
        return None, "empty response"
    i = text.find("{")
    if i < 0:
        return None, "no JSON found"
    try:
        data, _ = json.JSONDecoder().raw_decode(text, i)
    except json.JSONDecodeError as e:
        return None, f"bad JSON: {e}"
    if not isinstance(data, dict):
        return None, "bad JSON: not an object"
    return data, None


def image_type_of(text: str) -> Optional[str]:
    """ดึง image_type ที่โมเดลบอก · ไม่มีก็ None ไม่ใช่เรื่องผิดพลาด (v2 ไม่มีฟิลด์นี้)"""
    data, _ = first_json(text or "")
    if data is None:
        return None
    v = data.get("image_type")
    v = str(v).lower().strip() if v is not None else None
    return v if v in {"thermal", "colour", "color", "unclear"} else None


def parse(text: str) -> Tuple[Optional[LLMVerdict], Optional[str]]:
    """แกะ JSON จากคำตอบ ทนกับ markdown fence ข้อความนำ และข้อความต่อท้าย

    โมเดลชอบห่อ JSON ด้วย ```json แม้จะสั่งว่าอย่าห่อ
    """
    data, err = first_json(text)
    if data is None:
        return None, err

    animals = []
    for a in data.get("animals", []):
        sp = str(a.get("species", "unknown")).lower().strip()
        if sp not in VALID:
            sp = "unknown"
        try:
            n = max(0, int(a.get("count", 0)))
        except (TypeError, ValueError):
            n = 0
        try:
            conf = float(a.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        if n == 0:
            continue
        animals.append(SpeciesTally(species=sp, count=n,
                                    confidence=max(0.0, min(1.0, conf))))
    # กรอบที่ลอยอยู่โดยไม่มีสัตว์สักตัวถูกนับ = คำตอบขัดกันเอง · ทิ้งกรอบ
    # ไม่งั้นปลายทางเห็นกรอบบนภาพแล้วเข้าใจว่าเจอช้าง ทั้งที่ counts ว่าง
    boxes = valid_boxes(data.get("boxes")) if animals else []
    return LLMVerdict(animals=animals, boxes=boxes), None
