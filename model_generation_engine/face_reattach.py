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


class RelitOpticalFaceIntegrator:
    """Method 3: Zero-AI 3D Relit Quotient Frequency-Separated High-Res Face Integrator.

    Transfers high-resolution (400px–1600px, 870+ token equivalent) authentic facial micro-anatomy
    (eyebrows, thin double-eyelid crease, iris, rounded nasal tip, philtrum, lips, inner cheeks) from
    untouched Gen-0 or the angle-matched Pure 1-Hop 8-Direction Bank onto a 1K->4K full-body/medium
    scene canvas while inheriting 100% of the target scene's 3D illumination envelope, Kelvin white
    balance, ambient bounce, foreground hair strands, and optical lens MTF (ZERO AI re-generation!).
    """

    def __init__(self, vision_analyzer: AppleVisionAnalyzer) -> None:
        self.vision = vision_analyzer

    def select_angle_matched_highres_source(
        self,
        target_bgr: np.ndarray,
        gen0_path: Path,
        angle_bank: Optional[Dict[str, Path]] = None,
    ) -> Tuple[Path, str, Dict[str, float]]:
        """Pick untouched Gen-0 (when within ~14 deg) or the closest 1-Hop 8-Dir anchor matching target 3D head turn."""
        import cv2

        dst_pil = Image.fromarray(cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB))
        t_marks = self.vision.landmarks(dst_pil)
        t_shape = self.vision.compute_face_shape(t_marks)
        t_turn = float(t_shape["turn"]) if t_shape else 0.0

        candidates: list[Tuple[str, Path]] = [("gen0_origin", Path(gen0_path))]
        if angle_bank:
            for k, p in angle_bank.items():
                if p is not None and Path(p).exists() and Path(p).resolve() != Path(gen0_path).resolve():
                    candidates.append((str(k), Path(p)))

        scored: list[Tuple[float, str, Path, float]] = []
        for label, cand_p in candidates:
            im = cv2.imread(str(cand_p))
            if im is None:
                continue
            c_marks = self.vision.landmarks(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
            c_shape = self.vision.compute_face_shape(c_marks)
            if not c_shape:
                continue
            c_turn = float(c_shape["turn"])
            diff = abs(c_turn - t_turn)
            # Strongly prefer untouched Gen-0 whenever angular delta is small (<= 0.14 turn ~= 13 deg) -> 0% AI drift!
            eff_cost = diff - 0.065 if label == "gen0_origin" else diff
            scored.append((eff_cost, label, cand_p, c_turn))

        if not scored:
            return Path(gen0_path), "gen0_origin", {"targetTurn": round(t_turn, 3), "sourceTurn": 0.0}
        scored.sort(key=lambda x: x[0])
        _, chosen_label, chosen_path, chosen_turn = scored[0]
        return chosen_path, chosen_label, {
            "targetTurn": round(t_turn, 3),
            "sourceTurn": round(chosen_turn, 3),
            "turnDelta": round(abs(chosen_turn - t_turn), 3),
        }

    @staticmethod
    def _norm_gaussian(arr_f32: np.ndarray, weight_f32: np.ndarray, sigma: float) -> np.ndarray:
        import cv2

        w = weight_f32[..., None] if arr_f32.ndim == 3 else weight_f32
        num = cv2.GaussianBlur(arr_f32 * w, (0, 0), sigma)
        den = cv2.GaussianBlur(weight_f32, (0, 0), sigma)
        if arr_f32.ndim == 3:
            den = den[..., None]
        return num / np.maximum(den, 1e-3)

    def integrate_relit_face(
        self,
        source_face_bgr: np.ndarray,
        target_canvas_bgr: np.ndarray,
        detail_strength: float = 0.94,
        eyebrow_socket_lift: float = 0.14,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """Align high-res source face via rigid craniofacial frame and blend via 3D quotient relighting + hair-strand gate."""
        import cv2

        src_rgb = Image.fromarray(cv2.cvtColor(source_face_bgr, cv2.COLOR_BGR2RGB))
        dst_rgb = Image.fromarray(cv2.cvtColor(target_canvas_bgr, cv2.COLOR_BGR2RGB))
        S = self.vision.extract_key_points(src_rgb)
        T = self.vision.extract_key_points(dst_rgb)
        if S is None or T is None:
            raise RuntimeError("Apple Vision failed to locate facial landmarks on source or target canvas.")

        h_dst, w_dst = target_canvas_bgr.shape[:2]
        fw = float(T["width"])
        t_contour = T["contour"]
        cx = (min(p[0] for p in t_contour) + max(p[0] for p in t_contour)) / 2.0
        cy_mid = float(T["mid"][1])
        chin_y = float(T["chin"][1])

        rx1 = max(0, int(round(cx - fw * 1.15)))
        ry1 = max(0, int(round(cy_mid - fw * 1.15)))
        rx2 = min(w_dst, int(round(cx + fw * 1.15)))
        ry2 = min(h_dst, int(round(chin_y + fw * 0.45)))
        rw, rh = rx2 - rx1, ry2 - ry1
        dst_roi = target_canvas_bgr[ry1:ry2, rx1:rx2].copy()

        # Rigid similarity transform anchored on inter-pupillary span + ocular-menton height (preserves 100% real proportions!)
        s_iod = math.dist(T["lp"], T["rp"]) / max(1.0, math.dist(S["lp"], S["rp"]))
        s_len = math.dist(T["mid"], T["chin"]) / max(1.0, math.dist(S["mid"], S["chin"]))
        scale = 0.62 * s_iod + 0.38 * s_len
        ang_rad = math.atan2(T["rp"][1] - T["lp"][1], T["rp"][0] - T["lp"][0]) - math.atan2(
            S["rp"][1] - S["lp"][1], S["rp"][0] - S["lp"][0]
        )
        cos_a, sin_a = math.cos(ang_rad) * scale, math.sin(ang_rad) * scale
        sx0, sy0 = float(S["mid"][0]), float(S["mid"][1])
        tx0, ty0 = float(T["mid"][0]) - rx1, float(T["mid"][1]) - ry1
        M = np.array(
            [
                [cos_a, -sin_a, tx0 - (cos_a * sx0 - sin_a * sy0)],
                [sin_a, cos_a, ty0 - (sin_a * sx0 + cos_a * sy0)],
            ],
            dtype=np.float32,
        )
        warped_src = cv2.warpAffine(
            source_face_bgr, M, (rw, rh), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101
        )

        def _xform_pts(pts_list):
            out_p = []
            for px, py in pts_list:
                nx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
                ny = M[1, 0] * px + M[1, 1] * py + M[1, 2]
                out_p.append([nx, ny])
            return out_p

        s_marks = S["marks"]
        t_marks = T["marks"]

        # Build anatomical inner-facial feature mask strictly inside jawline/temples (zero jaw/collar/ear seam!)
        organ_pts: list[list[float]] = []
        for k in ("leftEyebrow", "rightEyebrow", "leftEye", "rightEye", "nose", "noseCrest", "outerLips"):
            if s_marks.get(k):
                organ_pts.extend(_xform_pts(s_marks[k]))
            elif t_marks.get(k):
                organ_pts.extend([[p[0] - rx1, p[1] - ry1] for p in t_marks[k]])

        # Use 94% of aligned SOURCE contour (so real cheek fullness & jaw breadth transfer!) intersected safely inside dst person/hair
        c_center_x = tx0
        c_center_y = ty0 + 0.22 * ((chin_y - ry1) - ty0)
        src_jaw_pts = _xform_pts(s_marks["contour"]) if s_marks.get("contour") else [[p[0] - rx1, p[1] - ry1] for p in t_contour]
        inset_jaw: list[list[float]] = []
        for lx, ly in src_jaw_pts:
            inset_jaw.append([c_center_x + 0.93 * (lx - c_center_x), c_center_y + 0.92 * (ly - c_center_y)])

        poly_mask = np.zeros((rh, rw), dtype=np.float32)
        all_inner = organ_pts + inset_jaw
        if len(all_inner) >= 4:
            hull = cv2.convexHull(np.array(all_inner, dtype=np.float32)).astype(np.int32)
            cv2.fillConvexPoly(poly_mask, hull, 1.0)
        else:
            cv2.ellipse(
                poly_mask,
                (int(round(tx0)), int(round(ty0 + 0.16 * fw))),
                (max(8, int(round(0.42 * fw))), max(8, int(round(0.46 * fw)))),
                math.degrees(ang_rad),
                0,
                360,
                1.0,
                -1,
            )

        # Clip strictly inside union of source+target jaw inside dst_person so background/collar NEVER leak, yet full cheeks fit!
        outer_limit = np.zeros((rh, rw), dtype=np.float32)
        jaw_union = np.vstack([
            np.array([[p[0] - rx1, p[1] - ry1] for p in t_contour], dtype=np.float32),
            np.array(src_jaw_pts, dtype=np.float32),
            np.array([[tx0 - 0.36 * fw, ty0 - 0.38 * fw], [tx0, ty0 - 0.52 * fw], [tx0 + 0.36 * fw, ty0 - 0.38 * fw]], dtype=np.float32),
        ])
        dome_hull = cv2.convexHull(jaw_union).astype(np.int32)
        cv2.fillConvexPoly(outer_limit, dome_hull, 1.0)
        pm_roi = self.vision.person_mask(Image.fromarray(cv2.cvtColor(dst_roi, cv2.COLOR_BGR2RGB)))
        if pm_roi is not None:
            pm_f = (np.array(pm_roi.resize((rw, rh), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0)
            outer_limit = outer_limit * np.clip((pm_f - 0.35) / 0.45, 0.0, 1.0)
        erode_k = max(3, int(round(fw * 0.032)) | 1)
        outer_limit = cv2.erode(outer_limit, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k)))

        raw_mask = poly_mask * outer_limit
        feather_sigma = max(4.0, fw * 0.082)
        soft_mask = cv2.GaussianBlur(raw_mask, (0, 0), feather_sigma)
        soft_mask = np.clip(soft_mask * 1.08, 0.0, 1.0) * cv2.GaussianBlur(outer_limit, (0, 0), max(3.0, fw * 0.045))

        # Foreground Hair-Strand Protection Gate: ONLY protect thin stray hair strands inside forehead/upper-temple, NEVER crush cheek breadth!
        src_lab = cv2.cvtColor(warped_src, cv2.COLOR_BGR2LAB).astype(np.float32)
        dst_lab = cv2.cvtColor(dst_roi, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Protect eyes/brows/lashes/lips AND mid-lower cheeks from hair-gate attenuation
        eye_brow_keep = np.zeros((rh, rw), dtype=np.float32)
        for k in ("leftEyebrow", "rightEyebrow", "leftEye", "rightEye", "nose", "outerLips"):
            pts_k = _xform_pts(s_marks[k]) if s_marks.get(k) else []
            if len(pts_k) >= 3:
                hk = cv2.convexHull(np.array(pts_k, dtype=np.float32)).astype(np.int32)
                cv2.fillConvexPoly(eye_brow_keep, hk, 1.0)
        # Below pupil line (y >= ty0), keep full cheek breadth so narrow dst hair never pinches the subject's cheeks
        cheek_band = np.zeros((rh, rw), dtype=np.float32)
        cheek_band[max(0, int(round(ty0))) :, :] = 0.75
        eye_brow_keep = np.maximum(
            cv2.dilate(eye_brow_keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, int(fw * 0.08) | 1),) * 2)),
            cheek_band,
        )
        eye_brow_keep = cv2.GaussianBlur(eye_brow_keep, (0, 0), max(2.0, fw * 0.03))

        dst_L = dst_lab[:, :, 0]
        src_L = src_lab[:, :, 0]
        skin_p50 = float(np.median(dst_L[soft_mask > 0.35])) if np.any(soft_mask > 0.35) else 150.0
        fg_hair_prob = np.clip((skin_p50 - 42.0 - dst_L) / 22.0, 0.0, 1.0) * np.clip((src_L - dst_L - 22.0) / 22.0, 0.0, 1.0)
        fg_hair_gate = 1.0 - fg_hair_prob * (1.0 - np.clip(eye_brow_keep, 0.0, 1.0))
        fg_hair_gate = cv2.GaussianBlur(fg_hair_gate, (0, 0), max(1.5, fw * 0.018))
        final_mask = np.clip(soft_mask * fg_hair_gate * detail_strength, 0.0, 1.0)

        # Optical MTF Bandwidth Parity: match ultra-high-frequency grain of warped_src to dst camera plane
        if scale < 0.72:
            mtf_sigma = float(np.clip((0.78 - scale) * 1.05, 0.30, 0.75))
            warped_src_mtf = cv2.GaussianBlur(warped_src, (0, 0), mtf_sigma)
            src_lab = cv2.cvtColor(warped_src_mtf, cv2.COLOR_BGR2LAB).astype(np.float32)
            src_L = src_lab[:, :, 0]

        # 3D Hemisphere Studio Illumination Envelope Transfer (`sigma = 0.50 * fw` preserves 100% mid-frequency facial anatomy!)
        support = np.clip((outer_limit > 0.1).astype(np.float32) * fg_hair_gate, 0.05, 1.0)
        sigma_env = max(16.0, fw * 0.50)
        L_src_env = self._norm_gaussian(src_L, support, sigma_env)
        L_dst_env = self._norm_gaussian(dst_L, support, sigma_env)
        sigma_chroma = max(12.0, fw * 0.38)
        A_src_env = self._norm_gaussian(src_lab[:, :, 1], support, sigma_chroma)
        B_src_env = self._norm_gaussian(src_lab[:, :, 2], support, sigma_chroma)
        A_dst_env = self._norm_gaussian(dst_lab[:, :, 1], support, sigma_chroma)
        B_dst_env = self._norm_gaussian(dst_lab[:, :, 2], support, sigma_chroma)

        # Normalize src envelope mean to dst envelope mean for gentle socket fill without boundary shift
        valid_m = final_mask > 0.25
        mean_diff = float(np.mean(L_dst_env[valid_m] - L_src_env[valid_m])) if np.any(valid_m) else 0.0
        L_src_env_shifted = np.clip(L_src_env + mean_diff, 15.0, 245.0)
        L_target_env = (1.0 - eyebrow_socket_lift * final_mask) * L_dst_env + (eyebrow_socket_lift * final_mask) * L_src_env_shifted

        quotient = (src_L + 32.0) / np.maximum(L_src_env + 32.0, 20.0)
        L_relit = np.clip((L_target_env + 32.0) * quotient - 32.0, 0.0, 255.0)
        A_relit = np.clip(A_dst_env + 0.85 * (src_lab[:, :, 1] - A_src_env), 0.0, 255.0)
        B_relit = np.clip(B_dst_env + 0.85 * (src_lab[:, :, 2] - B_src_env), 0.0, 255.0)

        relit_lab = np.stack([L_relit, A_relit, B_relit], axis=-1).astype(np.uint8)
        relit_bgr = cv2.cvtColor(relit_lab, cv2.COLOR_LAB2BGR).astype(np.float32)

        m3 = final_mask[..., None]
        blended_roi = np.clip(relit_bgr * m3 + dst_roi.astype(np.float32) * (1.0 - m3), 0.0, 255.0).astype(np.uint8)

        # Measure boundary seam L1 diff along the outer feather ring (0.04 < mask < 0.16)
        ring = (final_mask > 0.04) & (final_mask < 0.16)
        seam_l1 = (
            float(np.mean(np.abs(blended_roi.astype(np.float32)[ring] - dst_roi.astype(np.float32)[ring])))
            if np.any(ring)
            else 0.0
        )

        out_canvas = target_canvas_bgr.copy()
        out_canvas[ry1:ry2, rx1:rx2] = blended_roi
        return out_canvas, {
            "scale": round(scale, 3),
            "angleDeg": round(math.degrees(ang_rad), 2),
            "targetFaceWidthPx": round(fw, 1),
            "seamL1Diff": round(seam_l1, 3),
            "roi": (rx1, ry1, rx2, ry2),
        }
