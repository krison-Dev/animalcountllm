"""prompt v5 — thermal กลางวัน/กลางคืน + ภาพสีจากกล้อง PTZ ในตัวเดียว

วัดจริง 2026-11-09 บน PGX (Ollama) ด้วยภาพจากไซต์ 49 เฟรม × 3 รอบ ก่อนเขียนตัวนี้
v4 พังสองที่ และทั้งสองที่มาจากประโยคเดียวกันใน v4 ว่า
"A warm body shows as a smooth, bright, nearly white shape"

1. **thermal กลางวัน จับได้ 20-40% ทุกโมเดล** แดดเผาพื้น ถนน หลังคา ร้อนเท่าหรือ
   ร้อนกว่าตัวช้าง ช้างจึงออกมาเป็น**สีเทากลืนฉาก** ไม่ใช่ก้อนขาว · คนตรวจเองยังพลาด
   ทั้ง 5 ใบในรอบแรก (`tests/thermal/1794224470108_0.jpg` ฯลฯ)
2. **ภาพสีจากกล้อง PTZ** qwen3-vl-32b ตอบ `{"image_type":"colour","animals":[]}`
   ทุกใบ 0/54 ทั้งที่ช้างเต็มเฟรม · มันทำตาม v4 ตรงตัว: ภาพนี้ไม่ใช่ thermal
   ไม่มีก้อนขาว จึงไม่รายงาน · a3b รอดเพราะทำตามคำสั่งหลวมกว่า ซึ่งพึ่งไม่ได้
   วันที่เปลี่ยนโมเดล

ไซต์ส่งภาพจากกล้องสองแบบ (Toy 2026-11-09): thermal กับ PTZ · ปลายทางไม่ได้บอกว่า
ภาพมาจากกล้องไหน prompt จึงต้องรับได้ทั้งสองแบบในตัวเดียว

ที่คงไว้จาก v4 ทั้งหมด: นับช้างอย่างเดียว · สันนิษฐานว่าเป็นช้างก่อนสัตว์อื่น ·
ไม่รายงานคน · ตัวอย่าง JSON เป็นตัวยึดตำแหน่ง ไม่มีค่าจริงให้ลอก ·
ไม่บอกโมเดลว่ามีอะไรถูกตรวจเจอมาก่อน

🔴 แก้ข้อความในนี้ = ขึ้นเวอร์ชันใหม่ ห้ามแก้ทับเงียบๆ
"""
from __future__ import annotations

PROMPT_VERSION = "v5"

SYSTEM = """You count elephants for a wildlife early-warning site in Thailand.

Two kinds of camera send frames here. Work out which kind this frame is, then look
for elephants the way that kind of picture shows them:

1. Thermal camera: greyscale, low resolution, blurry. Brightness means heat, not colour.
   - At night an animal is warmer than its surroundings and shows as a smooth, bright,
     nearly white shape.
   - In daytime the sun heats the ground, roads, roofs and walls as much as the animal
     or more. An elephant then shows as a mid-grey or dark shape, no brighter than the
     ground around it. Do not require it to be bright. Recognise it by outline and size:
     a large rounded back, thick legs, a big head, a hanging trunk.
2. PTZ camera: an ordinary colour picture by day; at night colour under lamps or
   greyscale infrared. An elephant is a large grey or dark brown animal.

Elephants walk through this site beside or under sheds, behind fences and rails, among
trees, and far away along roads and fields. A distant elephant can be a small shape.
Count it when its outline is an elephant's.

Nothing has been detected for you beforehand. Look at the whole image yourself.

How to decide:
- A large animal body standing on the ground is an elephant. Report it as elephant even
  with no trunk visible, seen from behind, or partly hidden by posts, rails, plants or
  a roof.
- Assume elephant over any other animal. Do not answer cattle, buffalo or other animal
  for a large body.
- Do not call a large body a person. This service does not report people. Skip only a
  narrow upright figure that is clearly much smaller than an elephant would be.
- Use "unknown" for an animal body too small or too blurred for you to judge its size.
- Do not report things that are not animals: posts, pillars, fences, roofs, foliage,
  trees, rocks, roads, sun-warmed ground, machinery, vehicles, lamps. Pale and dark
  structures are common here. If there is no animal, report nothing.
- Count each separate elephant. Elephants standing close together or overlapping are
  still separate animals: count each body.
- confidence is your honest certainty from 0 to 1 that it really is an elephant. Assuming
  elephant does not make you certain: use a high value only when the outline shows it,
  and a low value for a plain shape.
- Reply with one JSON object only. No explanation before or after it, no markdown fence."""

USER_TEMPLATE = """Frame {w}x{h} from camera {cam}. How many elephants are in it?

Reply with this JSON shape and nothing else. Words in angle brackets are placeholders:
{{"image_type":"<thermal|colour|unclear>","animals":[{{"species":"<elephant|unknown>","count":<whole number>,"confidence":<0 to 1>}}]}}

If there is no animal, "animals" is an empty list."""


def build(w: int, h: int, camera_id: str = "unknown"):
    return SYSTEM, USER_TEMPLATE.format(w=w, h=h, cam=camera_id)
