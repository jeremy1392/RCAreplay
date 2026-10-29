"""Level-6 attachments: photographed forms, delivery slips and packing labels carrying QR codes.

Images are rendered deterministically from the attachment spec stored in the world state, so
trajectories only store references ({"type": "image_ref"}) and are expanded at call time.
"""
from __future__ import annotations

import base64
import functools
import io
import math
import random
from pathlib import Path

import numpy as np
import segno
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"), Path("/Library/Fonts")]
FONTS = {"print": ["arial.ttf", "DejaVuSans.ttf", "Arial.ttf"],
         "bold": ["arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"],
         "mono": ["consola.ttf", "DejaVuSansMono.ttf", "Courier New.ttf"],
         "hand": ["Inkfree.ttf", "segoepr.ttf", "comic.ttf", "DejaVuSans-Oblique.ttf"]}
PAPER, INK, PEN = (246, 243, 232), (25, 25, 30), (20, 45, 140)


@functools.lru_cache(maxsize=64)
def _font(kind: str, size: int):
    for d in FONT_DIRS:
        for name in FONTS[kind]:
            if (d / name).exists():
                return ImageFont.truetype(str(d / name), size)
    return ImageFont.load_default(size=size)


def _qr(payload: str, scale: int) -> Image.Image:
    buf = io.BytesIO()
    segno.make(payload, error="m", micro=False).save(buf, kind="png", scale=scale, border=2)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _scribble(draw, x, y, w, rng):
    pts = [(x + i * w / 14, y + rng.uniform(-12, 12) + 8 * math.sin(i * 1.3)) for i in range(15)]
    draw.line(pts, fill=PEN, width=3, joint="curve")


def _photo(img: Image.Image, rng: random.Random) -> Image.Image:
    """Make a clean render look like a phone photo: tint, rotation, noise, blur."""
    angle = rng.uniform(-7, 7)
    bg = tuple(rng.randint(90, 140) for _ in range(3))
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=bg)
    arr = np.asarray(img).astype(np.float32)
    gy = np.linspace(rng.uniform(0.85, 1.0), rng.uniform(0.95, 1.08), arr.shape[0])[:, None, None]
    arr = arr * gy + np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(5, 11), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 0.9)))


def _damage_form(spec, rng):
    img = Image.new("RGB", (900, 620), PAPER)
    d = ImageDraw.Draw(img)
    d.text((40, 30), spec.get("title", "DAMAGE FORM"), font=_font("bold", 40), fill=INK)
    d.text((40, 82), "Maison&Co logistics - stock loss declaration", font=_font("print", 20), fill=INK)
    d.line([(40, 115), (860, 115)], fill=INK, width=2)
    rows = [("Incident", spec["incident"]), ("Warehouse", spec["warehouse"]), ("Date", spec["date"]),
            ("Product", "see label code"), ("Damaged quantity", spec["qty"]), ("Cause", spec["cause"])]
    y = 140
    for label, value in rows:
        d.text((40, y), f"{label}:", font=_font("print", 24), fill=INK)
        d.text((290, y), value, font=_font("mono", 26), fill=INK)
        if label == "Damaged quantity" and spec.get("correction"):
            w = d.textlength(value, font=_font("mono", 26))
            d.line([(285, y + 15), (295 + w, y + 13)], fill=PEN, width=4)
            d.text((300 + w, y - 12), spec["correction"]["hand"], font=_font("hand", 40), fill=PEN)
            d.text((300 + w + 90, y + 4), spec["correction"]["initials"], font=_font("hand", 24), fill=PEN)
        y += 52
    qr = _qr(spec["qr"], 5)
    img.paste(qr, (900 - qr.width - 40, 135))
    d.text((900 - qr.width - 40, 140 + qr.height), "product label", font=_font("print", 16), fill=INK)
    d.text((40, 540), "Declared by:", font=_font("print", 22), fill=INK)
    _scribble(d, 200, 548, 180, rng)
    return img


def _delivery_slip(spec, rng):
    img = Image.new("RGB", (860, 560), PAPER)
    d = ImageDraw.Draw(img)
    d.text((36, 26), f"{spec['carrier']}  -  BON DE LIVRAISON / DELIVERY SLIP", font=_font("bold", 30), fill=INK)
    d.line([(36, 72), (824, 72)], fill=INK, width=2)
    d.text((36, 96), "Shipment reference: see code", font=_font("print", 22), fill=INK)
    d.text((36, 140), "Recipient: " + spec["recipient"], font=_font("print", 22), fill=INK)
    d.text((36, 210), "Date de reception (JJ/MM/AAAA)", font=_font("print", 22), fill=INK)
    d.text((36, 238), "Received on (DD/MM/YYYY):", font=_font("print", 22), fill=INK)
    d.text((360, 214), spec["hand_date"], font=_font("hand", 46), fill=PEN)
    d.text((36, 330), "Signature:", font=_font("print", 22), fill=INK)
    _scribble(d, 170, 340, 220, rng)
    d.text((36, 470), "Le destinataire certifie avoir recu le colis en bon etat.", font=_font("print", 18), fill=INK)
    qr = _qr(spec["qr"], 5)
    img.paste(qr, (860 - qr.width - 36, 96))
    return img


def _packing_label(spec, rng):
    img = Image.new("RGB", (760, 440), (236, 226, 200))
    d = ImageDraw.Draw(img)
    d.rectangle([(30, 30), (730, 410)], outline=INK, width=3)
    d.text((55, 50), "MAISON&CO - PACKING LABEL", font=_font("bold", 30), fill=INK)
    d.text((55, 110), spec["product"], font=_font("print", 24), fill=INK)
    d.text((55, 160), f"LOT {spec['printed_lot']}", font=_font("mono", 44), fill=INK)
    d.text((55, 230), f"Packed {spec['packed']}", font=_font("print", 22), fill=INK)
    if spec.get("stamp"):
        st = Image.new("RGBA", (330, 80), (0, 0, 0, 0))
        sd = ImageDraw.Draw(st)
        sd.rectangle([(2, 2), (327, 77)], outline=(190, 30, 30, 255), width=5)
        sd.text((18, 14), spec["stamp"], font=_font("bold", 40), fill=(190, 30, 30, 255))
        st = st.rotate(rng.uniform(-18, 18), expand=True)
        img.paste(st, (60, 290), st)
    qr = _qr(spec["qr"], 6)
    img.paste(qr, (760 - qr.width - 50, 110))
    return img


RENDERERS = {"damage_form": _damage_form, "delivery_slip": _delivery_slip, "packing_label": _packing_label}


def render(att_id: str, att: dict) -> bytes:
    rng = random.Random(f"{att_id}:{att['kind']}:{att['spec'].get('qr', '')}")
    img = _photo(RENDERERS[att["kind"]](att["spec"], rng), rng)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return buf.getvalue()


@functools.lru_cache(maxsize=512)
def _data_uri(att_id: str, att_json: str) -> str:
    import json
    return "data:image/jpeg;base64," + base64.b64encode(render(att_id, json.loads(att_json))).decode()


def data_uri(task: dict, att_id: str) -> str:
    import json
    return _data_uri(att_id, json.dumps(task["state"]["attachments"][att_id], sort_keys=True))


def image_message(att_ids: list[str]) -> dict:
    content = []
    for a in att_ids:
        content += [{"type": "text", "text": f"Attachment {a} (opened with view_attachment):"},
                    {"type": "image_ref", "attachment_id": a}]
    return {"role": "user", "content": content}


def expand(messages: list[dict], task: dict, vision: bool = True) -> list[dict]:
    """Replace image_ref parts by data URIs (or by a text stub for text-only models)."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list) and any(p.get("type") == "image_ref" for p in c):
            parts = []
            for p in c:
                if p.get("type") != "image_ref":
                    parts.append(p)
                elif vision:
                    parts.append({"type": "image_url", "image_url": {"url": data_uri(task, p["attachment_id"])}})
                else:
                    parts.append({"type": "text", "text": "[image not available to a text-only model]"})
            m = {**m, "content": parts}
        out.append(m)
    return out
