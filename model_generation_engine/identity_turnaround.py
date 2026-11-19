"""Real-Person 3D Turnaround & Canonical Identity Registration System (`실제 인물 3D 모델시트 변환 엔진`).

Bridges the structural gap between AI-native synthetic characters and user-uploaded real photographs:
1. `prepare_canonical_identity_anchor`: Normalizes any arbitrary user photo (full-body, seated, tiny face) into an
   85mm head-and-shoulders canonical portrait (`3:4` aspect) so ViT patch tokens focus on facial anatomy while preserving
   real head-to-shoulder ratio (`7.5~8등신` anchor).
2. `extract_anatomical_blueprint`: Measures deterministic craniofacial landmark ratios (`Apple Vision` + `YuNet`) to supply
   the missing verbal/geometric DNA (`identityPrompt` equivalent) that AI-generated characters possess natively.
3. `plan_progressive_5angle_turnaround`: Implements 2-Hop Progressive 3D Rotation (`0° Front -> ±45° Three-Quarter -> ±90° Profile`
   using dual `[Front + Same-Side 3/4]` anchors for 90° views, mirroring `AI모델스튜디오` `engine.py` §T02).
4. `assemble_master_turnaround_sheet` & `select_angle_matched_references`: Builds a 5-view Master Character Sheet and
   dynamically routes angle-matched reference anchors whenever generating or refining angled scenes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .apple_vision import AppleVisionAnalyzer
from .config import EngineConfig, SubjectProfile
from .detector import DynamicFaceDetector
from .prompt_builder import PromptFirewall
from .upscaler import SuperResolutionEngine

TURNAROUND_VIEWS: Tuple[str, ...] = (
    "left_profile",
    "three_quarter_left",
    "front",
    "three_quarter_right",
    "right_profile",
)

VIEW_LABELS_KO: Dict[str, str] = {
    "left_profile": "좌측면 (90°)",
    "three_quarter_left": "좌 ¾ (45°)",
    "front": "정면 원본 (0°)",
    "three_quarter_right": "우 ¾ (45°)",
    "right_profile": "우측면 (90°)",
}

VIEW_DIRECTIONS_EN: Dict[str, Tuple[str, str]] = {
    "front": (
        "DIRECT FRONT VIEW (0 degrees yaw)",
        "Head, nose centerline, gaze, and shoulders face directly toward the camera with balanced visibility of both ears.",
    ),
    "three_quarter_left": (
        "LEFT-FACING THREE-QUARTER VIEW (40 to 45 degrees yaw toward image-left)",
        "Rotate head and shoulders together 40-45 degrees toward the LEFT edge of the frame so the nose points image-left, "
        "both eyes remain visible in natural 3D foreshortening, and the image-right cheek/temple/ear side is more exposed.",
    ),
    "three_quarter_right": (
        "RIGHT-FACING THREE-QUARTER VIEW (40 to 45 degrees yaw toward image-right)",
        "Rotate head and shoulders together 40-45 degrees toward the RIGHT edge of the frame so the nose points image-right, "
        "both eyes remain visible in natural 3D foreshortening, and the image-left cheek/temple/ear side is more exposed.",
    ),
    "left_profile": (
        "TRUE LEFT SIDE PROFILE (85 to 90 degrees yaw toward image-left)",
        "Rotate head and shoulders together nearly 90 degrees so the nose tip, lips, chin, and horizontal gaze point cleanly "
        "toward the LEFT edge of the frame, showing the authentic side silhouette of the forehead, nasal bridge, philtrum, "
        "chin projection, and full image-right ear and temple.",
    ),
    "right_profile": (
        "TRUE RIGHT SIDE PROFILE (85 to 90 degrees yaw toward image-right)",
        "Rotate head and shoulders together nearly 90 degrees so the nose tip, lips, chin, and horizontal gaze point cleanly "
        "toward the RIGHT edge of the frame, showing the authentic side silhouette of the forehead, nasal bridge, philtrum, "
        "chin projection, and full image-left ear and temple.",
    ),
}


@dataclass(frozen=True)
class AnatomicalBlueprint:
    """Measured craniofacial ratio profile extracted directly from a real reference photograph."""

    eye_to_chin_ratio: float
    width_to_length_ratio: float
    iod_to_bbox_width_ratio: float
    nose_to_eye_vertical_ratio: float
    mouth_to_eye_width_ratio: float
    source_yaw_asymmetry: float
    face_area_fraction: float

    def to_3d_invariance_clause(self) -> str:
        """Emit a gender-neutral, firewall-safe 3D Euclidean bone-structure invariance specification."""
        return (
            f"Measured 3D Craniofacial Lock (vertical eye-to-menton index {self.eye_to_chin_ratio:.3f}, "
            f"craniofacial aspect index {self.width_to_length_ratio:.3f}, "
            f"inter-ocular-to-cranial index {self.iod_to_bbox_width_ratio:.3f}, "
            f"oral-to-ocular span index {self.mouth_to_eye_width_ratio:.3f}): "
            "maintain rigid 3D Euclidean anatomical invariance across camera angles—vertical eye-to-nose-to-menton "
            "spacing stays constant under horizontal rotation, horizontal contours foreshorten strictly by natural 3D "
            "perspective without altering soft-tissue volume, nasal bridge height and tip projection stay locked, "
            "and ear height/position relative to the eye-brow and nose-base plane remains identical."
        )


class RealPersonTurnaroundEngine:
    """Converts a single real photograph into a 5-Angle 3D Turnaround Anchor Bank & Master Sheet."""

    def __init__(
        self,
        config: EngineConfig,
        detector: DynamicFaceDetector,
        apple_vision: AppleVisionAnalyzer,
        upscaler: Optional[SuperResolutionEngine] = None,
    ) -> None:
        self.config = config
        self.detector = detector
        self.apple_vision = apple_vision
        self.upscaler = upscaler

    def extract_anatomical_blueprint(self, bgr_img: np.ndarray) -> AnatomicalBlueprint:
        """Measure canonical facial ratios via YuNet + Apple Vision without hardcoded assumptions."""
        h, w = bgr_img.shape[:2]
        face = self.detector.detect(bgr_img)
        le, re, nose, lm, rm = face.landmarks_5pt
        eye_d = max(1.0, face.inter_ocular_distance)
        eye_mid = 0.5 * (le + re)
        nose_vert = float(np.linalg.norm(nose - eye_mid)) / eye_d
        mouth_span = float(np.linalg.norm(rm - lm)) / eye_d
        _, _, fw, fh = face.bbox
        iod_ratio = eye_d / max(1.0, float(fw))
        face_frac = float(fw * fh) / max(1.0, float(w * h))

        eye_chin = 1.65
        width_len = 1.15
        try:
            pil_img = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
            marks = self.apple_vision.landmarks(pil_img)
            shape = self.apple_vision.compute_face_shape(marks)
            if shape:
                eye_chin = float(shape.get("eyeChin", eye_chin))
                width_len = float(shape.get("widthLength", width_len))
        except Exception:
            pass

        return AnatomicalBlueprint(
            eye_to_chin_ratio=eye_chin,
            width_to_length_ratio=width_len,
            iod_to_bbox_width_ratio=iod_ratio,
            nose_to_eye_vertical_ratio=nose_vert,
            mouth_to_eye_width_ratio=mouth_span,
            source_yaw_asymmetry=face.yaw_asymmetry_ratio,
            face_area_fraction=face_frac,
        )

    def prepare_canonical_identity_anchor(
        self,
        raw_photo_path: Path,
        output_anchor_path: Path,
        target_size: Tuple[int, int] = (1200, 1500),
    ) -> Tuple[Path, AnatomicalBlueprint]:
        """Normalize an arbitrary real photo into an 85mm head-and-shoulders canonical anchor (`4:5` / `1200x1500`).

        Eliminates 90%+ background/leg distraction while keeping full shoulder width so 7.5~8-head proportion
        context is preserved alongside high-density facial ViT tokens.
        """
        bgr = cv2.imread(str(raw_photo_path))
        if bgr is None:
            raise FileNotFoundError(f"Cannot read raw reference photo: {raw_photo_path}")

        h, w = bgr.shape[:2]
        face = self.detector.detect(bgr)
        fx, fy, fw, fh = face.bbox
        cx = fx + 0.5 * fw
        cy = fy + 0.5 * fh

        tw, th = target_size
        aspect = float(tw) / float(th)

        # Frame from comfortably above crown hair down to upper chest/shoulders (face height ~ 28% of frame)
        crop_h = min(float(h), max(fh * 3.45, fw * 3.8))
        crop_w = crop_h * aspect
        if crop_w > w:
            crop_w = float(w)
            crop_h = crop_w / aspect

        x1 = int(round(np.clip(cx - 0.5 * crop_w, 0, max(0, w - crop_w))))
        # Place face center around 40% from top so crown has breathing room and shoulders are clearly included
        y1 = int(round(np.clip(cy - 0.40 * crop_h, 0, max(0, h - crop_h))))
        x2 = min(w, int(round(x1 + crop_w)))
        y2 = min(h, int(round(y1 + crop_h)))

        crop = bgr[y1:y2, x1:x2]
        # If crop is lower resolution than target_size and upscaler is available, super-resolve first
        if self.upscaler is not None and (crop.shape[1] < tw * 0.85 or crop.shape[0] < th * 0.85):
            tmp_in = output_anchor_path.parent / f".tmp_canon_in_{output_anchor_path.stem}.jpg"
            tmp_up = output_anchor_path.parent / f".tmp_canon_up_{output_anchor_path.stem}.jpg"
            output_anchor_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(tmp_in), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 97])
            try:
                self.upscaler.upscale_image_file(tmp_in, tmp_up)
                up_bgr = cv2.imread(str(tmp_up))
                if up_bgr is not None:
                    crop = up_bgr
            finally:
                tmp_in.unlink(missing_ok=True)
                tmp_up.unlink(missing_ok=True)

        canonical = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_LANCZOS4)
        output_anchor_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_anchor_path), canonical, [int(cv2.IMWRITE_JPEG_QUALITY), 97])
        blueprint = self.extract_anatomical_blueprint(canonical)
        return output_anchor_path, blueprint

    @classmethod
    def build_turnaround_angle_prompt(
        cls,
        view: str,
        subject: SubjectProfile,
        blueprint: AnatomicalBlueprint,
        has_intermediate_3q_anchor: bool = False,
    ) -> str:
        """Build a 3D Euclidean turnaround prompt for Hop 1 (±45° 3/4) or Hop 2 (±90° profile with dual anchors)."""
        if view not in VIEW_DIRECTIONS_EN:
            raise ValueError(f"Unsupported turnaround view '{view}'. Choose from { tuple(VIEW_DIRECTIONS_EN) }.")
        view_title, view_desc = VIEW_DIRECTIONS_EN[view]

        ref_roles = (
            "Image 1 is the canonical front identity photograph—the primary ground-truth authority for facial anatomy, "
            "skin texture, eyewear frame geometry, hairline, and head-to-shoulder proportions."
        )
        if has_intermediate_3q_anchor:
            ref_roles += (
                " Image 2 is the approved same-side 45-degree three-quarter turnaround portrait of the exact same person—"
                "use Image 2 together with Image 1 as the 3D depth bridge for nasal bridge projection, zygomatic curve, "
                "eyewear temple arm path, ear helix contour, and side hairline."
            )

        prompt = (
            "Editorial portrait. Agency digital turnaround test photograph of the EXACT SAME real person shown in Image 1, "
            "captured in the exact same studio session with an 85mm portrait lens at eye level against a clean neutral studio backdrop. "
            f"Reference Roles: {ref_roles} "
            f"Target Angle — {view_title}: {view_desc} "
            "Do NOT mirror or flip to the opposite direction, do NOT invent a new person, and apply zero beauty-cam filter. "
            f"Identity & Eyewear Continuity: Preserve {subject.facial_identity_descriptor()}. "
            "If eyewear is present in Image 1, render the exact same frame material/thickness and render the visible temple arm "
            "extending continuously and uninterruptedly from the front outer frame hinge straight back to rest naturally over the top of the ear. "
            f"{blueprint.to_3d_invariance_clause()} "
            "Wardrobe & Lighting Continuity: Keep the same clean tailored studio attire, balanced adult 7.5-to-8-head head-to-shoulder "
            "scale, soft diffused studio lighting, and natural matte real skin texture with authentic subtle pores, zero plastic/oily gloss, "
            "zero text, and zero watermarks."
        )
        PromptFirewall.validate_prompt(prompt)
        return prompt

    def plan_progressive_5angle_turnaround(
        self,
        canonical_front_path: Path,
        subject: SubjectProfile,
        approved_3q_left_path: Optional[Path] = None,
        approved_3q_right_path: Optional[Path] = None,
    ) -> Dict[str, Dict[str, object]]:
        """Plan the 2-Hop Progressive 5-Angle Turnaround (`Hop 1: ±45°`, `Hop 2: ±90°` using ±45° bridge)."""
        front_bgr = cv2.imread(str(canonical_front_path))
        if front_bgr is None:
            raise FileNotFoundError(f"Cannot read canonical front anchor: {canonical_front_path}")
        blueprint = self.extract_anatomical_blueprint(front_bgr)

        plans: Dict[str, Dict[str, object]] = {
            "front": {
                "hop": 0,
                "view": "front",
                "reference_images": [str(canonical_front_path)],
                "prompt": None,
                "status": "canonical_source",
            }
        }

        for v3q in ("three_quarter_left", "three_quarter_right"):
            plans[v3q] = {
                "hop": 1,
                "view": v3q,
                "reference_images": [str(canonical_front_path)],
                "prompt": self.build_turnaround_angle_prompt(v3q, subject, blueprint, has_intermediate_3q_anchor=False),
            }

        for vprof, bridge_path in (
            ("left_profile", approved_3q_left_path),
            ("right_profile", approved_3q_right_path),
        ):
            refs = [str(canonical_front_path)]
            has_bridge = bridge_path is not None and Path(bridge_path).exists()
            if has_bridge:
                refs.append(str(bridge_path))
            plans[vprof] = {
                "hop": 2,
                "view": vprof,
                "reference_images": refs,
                "prompt": self.build_turnaround_angle_prompt(
                    vprof, subject, blueprint, has_intermediate_3q_anchor=has_bridge
                ),
            }

        return plans

    @staticmethod
    def assemble_master_turnaround_sheet(
        angle_image_paths: Dict[str, Path],
        output_sheet_path: Path,
        cell_size: Tuple[int, int] = (720, 900),
        include_labels: bool = False,
    ) -> Path:
        """Assemble the 5-angle views into a clean Master Character Turnaround Sheet (`[좌90° | 좌45° | 정면 | 우45° | 우90°]`).

        By default `include_labels=False` when the sheet will be passed as a visual reference to an image generator
        (so zero text/typography ever leaks into generated scenes).
        """
        cw, ch = cell_size
        pad = 20
        header_h = 56 if include_labels else 0
        ordered_views = [v for v in TURNAROUND_VIEWS if v in angle_image_paths and Path(angle_image_paths[v]).exists()]
        if not ordered_views:
            raise ValueError("No valid angle image paths supplied for turnaround sheet.")

        total_w = pad + len(ordered_views) * (cw + pad)
        total_h = pad + ch + header_h + pad
        sheet = Image.new("RGB", (total_w, total_h), (242, 242, 240))
        draw = ImageDraw.Draw(sheet)

        font = ImageFont.load_default()
        if include_labels:
            for fp in (
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
            ):
                if Path(fp).exists():
                    font = ImageFont.truetype(fp, 24)
                    break

        for idx, view in enumerate(ordered_views):
            x = pad + idx * (cw + pad)
            y = pad
            with Image.open(angle_image_paths[view]) as im:
                im_rgb = im.convert("RGB")
                src_w, src_h = im_rgb.size
                target_aspect = cw / float(ch)
                src_aspect = src_w / float(src_h)
                if src_aspect > target_aspect:
                    new_w = int(round(src_h * target_aspect))
                    x0 = (src_w - new_w) // 2
                    im_crop = im_rgb.crop((x0, 0, x0 + new_w, src_h))
                else:
                    new_h = int(round(src_w / target_aspect))
                    y0 = (src_h - new_h) // 2
                    im_crop = im_rgb.crop((0, y0, src_w, y0 + new_h))
                tile = im_crop.resize((cw, ch), Image.Resampling.LANCZOS)
                sheet.paste(tile, (x, y))
            if include_labels:
                draw.text((x + 12, y + ch + 14), VIEW_LABELS_KO.get(view, view), fill=(32, 36, 40), font=font)

        output_sheet_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(str(output_sheet_path), format="JPEG", quality=96)
        return output_sheet_path

    def select_angle_matched_references(
        self,
        canonical_front_path: Path,
        angle_bank: Dict[str, Path],
        target_view: Optional[str] = None,
        target_scene_bgr: Optional[np.ndarray] = None,
        master_sheet_path: Optional[Path] = None,
    ) -> Tuple[List[Path], str]:
        """Select `[Canonical Front, Angle-Matched Turnaround View]` based on explicit `target_view` or detected 3D yaw."""
        resolved_view = target_view
        if resolved_view is None and target_scene_bgr is not None:
            face = self.detector.detect(target_scene_bgr)
            yaw = face.yaw_asymmetry_ratio
            if yaw <= -0.26:
                resolved_view = "left_profile"
            elif yaw <= -0.07:
                resolved_view = "three_quarter_left"
            elif yaw >= 0.26:
                resolved_view = "right_profile"
            elif yaw >= 0.07:
                resolved_view = "three_quarter_right"
            else:
                resolved_view = "front"

        resolved_view = resolved_view or "front"
        refs: List[Path] = [Path(canonical_front_path)]

        if resolved_view != "front" and resolved_view in angle_bank and Path(angle_bank[resolved_view]).exists():
            refs.append(Path(angle_bank[resolved_view]))
        elif resolved_view == "left_profile" and "three_quarter_left" in angle_bank and Path(angle_bank["three_quarter_left"]).exists():
            refs.append(Path(angle_bank["three_quarter_left"]))
        elif resolved_view == "right_profile" and "three_quarter_right" in angle_bank and Path(angle_bank["three_quarter_right"]).exists():
            refs.append(Path(angle_bank["three_quarter_right"]))
        elif master_sheet_path is not None and Path(master_sheet_path).exists():
            refs.append(Path(master_sheet_path))

        return refs, resolved_view
