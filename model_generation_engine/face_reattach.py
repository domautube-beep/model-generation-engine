"""Universal Face Re-Attachment Engine (`얼굴 덮기 1c` & `머리 덧그리기 face_detail`).

Ported, unified, and generalized from AI모델스튜디오 (`studio/face_detail.py` & `work/2026-11-18-인물교체/facelock.py`):
1. `FaceLockCompositor` (`얼굴 덮기 1c` — Photographic Face Lock):
   - Aligns the true reference photograph to the target 5K portrait via Apple Vision `leftPupil`, `rightPupil`, and `chin`.
   - Intersects BOTH eroded `src_person` AND `dst_person` segmentation masks (`MinFilter`) to guarantee zero background fringe.
   - Applies Normalized-Convolution Weighted Frequency Separation (`wblur`) with configurable reference skin-tone retention
     so 100% of the real eyes, glasses, nose, mouth, and skin texture come from the reference photo while matching scene lighting.
2. `HeadRedrawCompositor` (`머리 덧그리기` — 2-Stage High-Res Head Redraw & Chin-Fade Stitch):
   - Extracts an aspect-balanced `2:3` head-and-shoulders crop (`1024x1536`) for distant/small-face shots.
   - Aligns the redrawn high-res head via Apple Vision `(leftPupil, rightPupil, chin)` and stitches strictly above the chin
     (`y_full = chin_y - 0.10*fw` .. `y_zero = chin_y + 0.04*fw`) so collars/garments never double-ghost.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .apple_vision import AppleVisionAnalyzer


class FaceLockCompositor:
    """DEPRECATED 2D Oval Pixel Overlay (`facelock 1c` — disabled by default in production).

    WARNING: Flat 2D oval pixel pasting cuts across eyewear temple arms (`cx ± 0.45*fw`), skews 2D facial axes
    against 3D rotated heads, distorts cheek-to-skull proportions, and breaks scene lighting/tone-and-manner harmony.
    Always use `ModelGenerationPipeline.run_tonematched_5k_synthesis` (Tone-Matched 3D In-Place Upper-Body Redraw) instead.
    """

    def __init__(self, vision_analyzer: AppleVisionAnalyzer) -> None:
        self.vision = vision_analyzer

    def apply_facelock(
        self,
        source_face_pil: Image.Image,
        target_canvas_pil: Image.Image,
        radius_k: float = 0.15,
        reference_tone_retention: float = 0.45,
    ) -> Tuple[Image.Image, Dict[str, object]]:
        """Overlay the true photographic face onto `target_canvas_pil` using weighted frequency separation."""
        src_rgb = source_face_pil.convert("RGB")
        dst_rgb = target_canvas_pil.convert("RGB")

        S = self.vision.extract_key_points(src_rgb)
        T = self.vision.extract_key_points(dst_rgb)
        if S is None or T is None:
            raise RuntimeError("Apple Vision failed to locate pupils/contour on source or target portrait.")

        src_pm = self.vision.person_mask(src_rgb)
        src_person = (
            src_pm.resize(src_rgb.size).point(lambda v: 255 if v > 140 else 0)
            if src_pm is not None
            else Image.new("L", src_rgb.size, 255)
        )

        # Compute ROI on target_canvas_pil so person segmentation runs tightly and fast on 5K canvases
        fw = float(T["width"])
        cx = (min(p[0] for p in T["contour"]) + max(p[0] for p in T["contour"])) / 2.0
        rx1 = max(0, int(round(cx - fw * 1.8)))
        ry1 = max(0, int(round(T["mid"][1] - fw * 1.6)))
        rx2 = min(dst_rgb.width, int(round(cx + fw * 1.8)))
        ry2 = min(dst_rgb.height, int(round(T["chin"][1] + fw * 1.2)))
        dst_roi = dst_rgb.crop((rx1, ry1, rx2, ry2))
        dst_pm_roi = self.vision.person_mask(dst_roi)
        dst_person = Image.new("L", dst_rgb.size, 0)
        if dst_pm_roi is not None:
            dst_person.paste(dst_pm_roi.resize(dst_roi.size).point(lambda v: 255 if v > 140 else 0), (rx1, ry1))
        else:
            dst_person.paste(Image.new("L", dst_roi.size, 255), (rx1, ry1))

        s = (
            math.dist(T["lp"], T["rp"]) / math.dist(S["lp"], S["rp"])
            + math.dist(T["mid"], T["chin"]) / math.dist(S["mid"], S["chin"])
        ) / 2.0
        ang = math.degrees(math.atan2(T["rp"][1] - T["lp"][1], T["rp"][0] - T["lp"][0])) - math.degrees(
            math.atan2(S["rp"][1] - S["lp"][1], S["rp"][0] - S["lp"][0])
        )

        big = src_rgb.resize((round(src_rgb.width * s), round(src_rgb.height * s)), Image.Resampling.LANCZOS)
        bigp = src_person.resize(big.size, Image.Resampling.NEAREST)
        c = (S["mid"][0] * s, S["mid"][1] * s)
        big = big.rotate(-ang, resample=Image.Resampling.BICUBIC, center=c, fillcolor=(255, 255, 255))
        bigp = bigp.rotate(-ang, resample=Image.Resampling.NEAREST, center=c, fillcolor=0)
        off = (round(T["mid"][0] - c[0]), round(T["mid"][1] - c[1]))

        layer = Image.new("RGB", dst_rgb.size, (255, 255, 255))
        layer.paste(big, off)
        person_src_aligned = Image.new("L", dst_rgb.size, 0)
        person_src_aligned.paste(bigp, off)

        # Dual-person-mask intersection (eroded source person ∩ eroded target person) prevents any jaw fringe
        erode_px = max(7, int(round(fw * 0.035)) | 1)
        dual_person = ImageChops.multiply(
            person_src_aligned.filter(ImageFilter.MinFilter(erode_px)),
            dst_person.filter(ImageFilter.MinFilter(erode_px)),
        )

        top = T["mid"][1] - fw * 0.60
        bottom = T["chin"][1] + fw * 0.02
        full = Image.new("L", dst_rgb.size, 0)
        ImageDraw.Draw(full).ellipse((cx - fw * 0.45, top, cx + fw * 0.45, bottom), fill=255)
        full = ImageChops.multiply(full, dual_person)
        mask = full.filter(ImageFilter.GaussianBlur(max(1.5, fw * 0.05)))

        core = Image.new("L", dst_rgb.size, 0)
        ImageDraw.Draw(core).ellipse(
            (cx - fw * 0.36, top + fw * 0.12, cx + fw * 0.36, bottom - fw * 0.10), fill=255
        )
        r = max(1.5, fw * radius_k)
        inner = core.filter(ImageFilter.GaussianBlur(2))

        # Crop computation to ROI for fast 5K execution
        layer_roi = layer.crop((rx1, ry1, rx2, ry2))
        inner_roi = inner.crop((rx1, ry1, rx2, ry2))
        mask_roi = mask.crop((rx1, ry1, rx2, ry2))

        def _wblur(im_pil: Image.Image) -> np.ndarray:
            num = Image.composite(im_pil, Image.new("RGB", im_pil.size), inner_roi).filter(
                ImageFilter.GaussianBlur(r)
            )
            den = inner_roi.filter(ImageFilter.GaussianBlur(r))
            return np.asarray(num, dtype=np.float32) / (np.asarray(den, dtype=np.float32)[..., None] / 255.0 + 1e-3)

        arr_layer = np.asarray(layer_roi, dtype=np.float32)
        wb_layer = _wblur(layer_roi)
        wb_target = _wblur(dst_roi)
        # Blend low-frequency illumination between scene lighting and reference true skin tone
        wb_blended = (1.0 - reference_tone_retention) * wb_target + reference_tone_retention * wb_layer
        face_arr = np.clip(arr_layer - wb_layer + wb_blended, 0, 255).astype(np.uint8)
        comp_roi = Image.composite(Image.fromarray(face_arr), dst_roi, mask_roi)

        out = dst_rgb.copy()
        out.paste(comp_roi, (rx1, ry1))

        src_sh = self.vision.compute_face_shape(S["marks"])
        out_marks = self.vision.landmarks(comp_roi)
        out_sh = self.vision.compute_face_shape(out_marks)
        cmp = self.vision.compare_face_shapes(src_sh, out_sh)
        return out, {
            "scale": round(s, 3),
            "angleDiff": round(ang, 2),
            "targetFaceWidth": round(fw, 1),
            "shapeComparison": cmp,
        }


class HeadRedrawCompositor:
    """2-Stage High-Res Head Redraw & Chin-Fade Stitch (`AI모델스튜디오/studio/face_detail.py`)."""

    CROP_WIDTH_MULT = 3.2  # Tight head-and-shoulders crop (2:3 vertical -> 1024x1536)

    def __init__(self, vision_analyzer: AppleVisionAnalyzer) -> None:
        self.vision = vision_analyzer

    def extract_head_crop(
        self, scene_pil: Image.Image, target_size: Tuple[int, int] = (1024, 1536)
    ) -> Tuple[Image.Image, Dict[str, object]]:
        """Extract a 2:3 vertical head-and-shoulders crop centered on the detected face."""
        rgb = scene_pil.convert("RGB")
        pts = self.vision.extract_key_points(rgb)
        if pts is None:
            raise RuntimeError("No face contour detected for head_crop extraction.")
        fw = float(pts["width"])
        cx = (min(p[0] for p in pts["contour"]) + max(p[0] for p in pts["contour"])) / 2.0
        top_face = min(p[1] for p in pts["contour"])
        cw = fw * self.CROP_WIDTH_MULT
        ch = cw * 1.5
        x0 = round(cx - cw / 2.0)
        y0 = min(round(pts["mid"][1] - fw * 1.45), round(top_face - fw * 0.85))
        box = (
            max(0, x0),
            max(0, y0),
            min(rgb.width, round(x0 + cw)),
            min(rgb.height, round(y0 + ch)),
        )
        crop = rgb.crop(box).resize(target_size, Image.Resampling.LANCZOS)
        return crop, {"cropBox": box, "sourceFaceWidth": round(fw, 1)}

    def blend_redrawn_head(
        self, target_canvas_pil: Image.Image, redrawn_head_pil: Image.Image
    ) -> Tuple[Image.Image, Dict[str, object]]:
        """Align `redrawn_head_pil` by pupils + chin and blend strictly above the chin (`face_detail.blend`)."""
        dst_rgb = target_canvas_pil.convert("RGB")
        head_rgb = redrawn_head_pil.convert("RGB")

        src = self.vision.extract_key_points(dst_rgb)
        head = self.vision.extract_key_points(head_rgb)
        if src is None or head is None:
            raise RuntimeError("Apple Vision failed to detect pupils/chin on target canvas or redrawn head.")

        mid, chin = src["mid"], src["chin"]
        lp, rp = src["lp"], src["rp"]
        s = (
            math.dist(lp, rp) / math.dist(head["lp"], head["rp"])
            + math.dist(mid, chin) / math.dist(head["mid"], head["chin"])
        ) / 2.0
        ang = math.degrees(math.atan2(head["rp"][1] - head["lp"][1], head["rp"][0] - head["lp"][0])) - math.degrees(
            math.atan2(rp[1] - lp[1], rp[0] - lp[0])
        )

        layer_head = head_rgb.resize(
            (round(head_rgb.width * s), round(head_rgb.height * s)), Image.Resampling.LANCZOS
        )
        h_mid = (head["mid"][0] * s, head["mid"][1] * s)
        layer_head = layer_head.rotate(ang, resample=Image.Resampling.BICUBIC, center=h_mid)
        ox, oy = round(mid[0] - h_mid[0]), round(mid[1] - h_mid[1])

        fx0 = min(p[0] for p in src["contour"])
        fx1 = max(p[0] for p in src["contour"])
        fw = fx1 - fx0
        cx = (fx0 + fx1) / 2.0

        # Build ROI around target head to keep 5K compositing fast and intersect person mask cleanly
        rx1 = max(0, int(round(cx - fw * 2.0)))
        ry1 = max(0, int(round(mid[1] - fw * 2.0)))
        rx2 = min(dst_rgb.width, int(round(cx + fw * 2.0)))
        ry2 = min(dst_rgb.height, int(round(chin[1] + fw * 1.0)))

        mask = Image.new("L", dst_rgb.size, 0)
        ImageDraw.Draw(mask).ellipse(
            (cx - fw * 0.90, mid[1] - fw * 1.02, cx + fw * 0.90, chin[1] + fw * 0.16), fill=255
        )
        mask = mask.filter(ImageFilter.GaussianBlur(max(1.5, fw * 0.10)))

        # Strictly exclude below the chin (neck/collar) so clothing collars never ghost
        y_full = chin[1] - fw * 0.10
        y_zero = chin[1] + fw * 0.04
        fade = Image.new("L", (1, dst_rgb.height), 0)
        for y in range(dst_rgb.height):
            val = (
                255
                if y <= y_full
                else 0
                if y >= y_zero
                else round(255.0 * (y_zero - y) / max(1e-3, y_zero - y_full))
            )
            fade.putpixel((0, y), val)
        mask = ImageChops.multiply(mask, fade.resize(dst_rgb.size))

        layer = Image.new("RGB", dst_rgb.size)
        layer.paste(layer_head, (ox, oy))

        # Feather `cover` inward from layer_head borders so top/side crop boundaries never leave a visible line
        inner_cover = Image.new("L", layer_head.size, 0)
        pad_c = max(16, int(round(min(layer_head.size) * 0.06)))
        ImageDraw.Draw(inner_cover).rectangle(
            (pad_c, pad_c, max(pad_c + 2, layer_head.width - pad_c), max(pad_c + 2, layer_head.height - pad_c)),
            fill=255,
        )
        inner_cover = inner_cover.filter(ImageFilter.GaussianBlur(max(8.0, pad_c * 0.55)))
        cover = Image.new("L", dst_rgb.size, 0)
        cover.paste(inner_cover, (ox, oy))
        mask = ImageChops.multiply(mask, cover)

        out = Image.composite(layer, dst_rgb, mask)
        return out, {
            "scale": round(s, 3),
            "angle": round(ang, 2),
            "offset": [ox, oy],
            "faceWidthPrint": round(fw, 1),
            "roi": (rx1, ry1, rx2, ry2),
        }
