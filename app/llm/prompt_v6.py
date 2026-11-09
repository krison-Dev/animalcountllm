"""prompt v6 — v5 + กรอบรายตัว (bbox) ในสเกล 0-1000

Toy ขอ 2026-11-09: เจอช้างแล้วให้บอกว่าอยู่ตรงไหน

🔴 CLAUDE.md เคยตัดสินว่า "ไม่ขอ bbox จากโมเดล" เพราะได้ [64,57,519,998] บนภาพสูง 628
ซึ่งตีความว่าเกินขอบภาพ · **ข้อสรุปนั้นอ่านหน่วยผิด** Qwen-VL ถูกเทรนให้ตอบพิกัด
เป็นสเกล 0-1000 ของภาพ ไม่ใช่พิกเซล · 998 คือเกือบขอบล่าง ไม่ใช่เกินขอบ

วัดจริง 2026-11-09 บน PGX (qwen3-vl-30b-a3b Q4 ผ่าน Ollama) 14 ภาพ ครอบทุกแบบ
(thermal กลางคืน/กลางวัน · ภาพ 384x288 · PTZ กลางวัน/กลางคืน · หลายตัว · ภาพว่าง)
กรอบครอบตัวช้างทุกกรอบเมื่อดูด้วยตา · ไม่มีกรอบเกินขอบสักกรอบ · ภาพว่างตอบกรอบว่าง
ยังไม่มีพิกัดที่คนตรวจแล้วให้วัดเป็นตัวเลข · ตรวจด้วยตาเท่านั้น

SYSTEM เป็นตัวเดียวกับ v5 ทุกตัวอักษร (import มา ไม่ได้ก๊อป) · เปลี่ยนแค่รูปคำตอบ
ถ้าจะแก้ SYSTEM ของ v5 ให้ขึ้น v7 แยก ไม่ใช่แก้ v5 ทับ เพราะ v6 จะเปลี่ยนตามเงียบๆ

🔴 จำนวนยังมาจาก "count" ไม่ได้มาจากจำนวนกรอบ · กรอบเป็นข้อมูลตำแหน่งเท่านั้น
ด่านในโค้ด (client.valid_boxes) ทิ้งกรอบที่ผิดรูป แต่ไม่เคยลดหรือเพิ่มจำนวนช้าง
"""
from __future__ import annotations

from . import prompt_v5

PROMPT_VERSION = "v6"

SYSTEM = prompt_v5.SYSTEM

USER_TEMPLATE = """Frame {w}x{h} from camera {cam}. How many elephants are in it, and where?

Reply with this JSON shape and nothing else. Words in angle brackets are placeholders:
{{"image_type":"<thermal|colour|unclear>","animals":[{{"species":"<elephant|unknown>","count":<whole number>,"confidence":<0 to 1>}}],"boxes":[[<x1>,<y1>,<x2>,<y2>]]}}

"boxes" has one box per elephant you counted, tight around that elephant's body, as
x1,y1,x2,y2 on a 0-1000 scale of the image width and height (0,0 is top-left, 1000,1000 is
bottom-right). If there is no animal, "animals" and "boxes" are empty lists."""


def build(w: int, h: int, camera_id: str = "unknown"):
    return SYSTEM, USER_TEMPLATE.format(w=w, h=h, cam=camera_id)
