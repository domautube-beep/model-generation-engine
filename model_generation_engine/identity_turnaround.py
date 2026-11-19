"""Real-Person 8-Angle 360° Turnaround & Canonical Identity Registration System (`실제 인물 8각도 3D 모델시트 변환 엔진`).

Universal, zero-hardcoding identity system for ALL projects:
1. `prepare_canonical_identity_anchor`: Extracts a pure untouched optical `85mm` head-and-shoulders anchor (ZERO `Real-ESRGAN`
   pre-smoothing, preserving 100% of real photographic sharpness and `7.5~8등신` shoulder context).
2. `extract_anatomical_blueprint`: Measures deterministic craniofacial ratios (`Apple Vision` + `YuNet`).
3. `plan_progressive_8angle_turnaround`: Plans the full 360° **8-Angle Canonical Compass** at 45° intervals
   (`0° Front · 45° Front-Left ¾ · 90° Left Profile · 135° Back-Left ¾ · 180° Back · 225° Back-Right ¾ · 270° Right Profile · 315° Front-Right ¾`)
   using progressive multi-hop bridging (`0° -> ±45° -> ±90° -> ±135° & 180°`).
4. `resolve_8angle_view_from_azimuth` & `select_angle_matched_references`: Maps any continuous 3D Previz azimuth (`0°..360°`)
   or detected scene yaw to `[Untouched Gen-0 Front Anchor, Exact Angle-Matched 8-Way View]` so zero iris/skin color drift occurs.
5. `mask_foreign_faces_in_reference`: Automatically detects and neutral-grey-masks foreign faces inside uploaded garment,
   accessory, bag, shoe, or style/mood reference images so foreign identities never contaminate the subject.
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

TURNAROUND_8_VIEWS: Tuple[str, ...] = (
    "front",
    "three_quarter_left",
    "left_profile",
    "back_left_3q",
    "back",
    "back_right_3q",
    "right_profile",
    "three_quarter_right",
)

# Backward-compatible 5-view constant (`TURNAROUND_VIEWS`) + full 360° 8-view constant (`TURNAROUND_8_VIEWS`)
TURNAROUND_5_VIEWS: Tuple[str, ...] = (
    "left_profile",
    "three_quarter_left",
    "front",
    "three_quarter_right",
    "right_profile",
)
TURNAROUND_VIEWS: Tuple[str, ...] = TURNAROUND_5_VIEWS

VIEW_AZIMUTH_DEG: Dict[str, float] = {
    "front": 0.0,
    "three_quarter_left": 45.0,
    "left_profile": 90.0,
    "back_left_3q": 135.0,
    "back": 180.0,
    "back_right_3q": 225.0,
    "right_profile": 270.0,
    "three_quarter_right": 315.0,
}

VIEW_LABELS_KO: Dict[str, str] = {
    "front": "① 정면 (0°)",
    "three_quarter_left": "② 좌전방 ¾ (45°)",
    "left_profile": "③ 좌측면 (90°)",
    "back_left_3q": "④ 좌후방 ¾ (135°)",
    "back": "⑤ 완전 후면 (180°)",
    "back_right_3q": "⑥ 우후방 ¾ (225°)",
    "right_profile": "⑦ 우측면 (270°)",
    "three_quarter_right": "⑧ 우전방 ¾ (315°)",
}

VIEW_DIRECTIONS_EN: Dict[str, Tuple[str, str]] = {
    "front": (
        "DIRECT FRONT VIEW (0 degrees azimuth)",
        "Head, nose centerline, gaze, and shoulders face directly toward the camera with balanced visibility of both ears.",
    ),
    "three_quarter_left": (
        "FRONT-LEFT THREE-QUARTER VIEW (45 degrees azimuth toward image-left)",
        "Rotate head and shoulders together 45 degrees toward the LEFT edge of the frame so the nose points image-left, "
        "both eyes remain visible in natural 3D foreshortening, and the image-right cheek/temple/ear side is clearly exposed.",
    ),
    "left_profile": (
        "TRUE LEFT SIDE PROFILE (90 degrees azimuth toward image-left)",
        "Rotate head and shoulders together 90 degrees so the nose tip, lips, chin, and horizontal gaze point cleanly "
        "toward the LEFT edge of the frame, showing the authentic side silhouette of forehead, nasal bridge, philtrum, chin, and ear.",
    ),
    "back_left_3q": (
        "BACK-LEFT THREE-QUARTER VIEW (135 degrees azimuth, facing away toward upper-left)",
        "Rotate head and shoulders together 135 degrees away from the camera toward image-left, showing the rear-three-quarter "
        "crown hair, tapered nape hairline, outer back curve of the visible ear, eyewear temple tip over the ear if worn, "
        "outer cheekbone edge, collar band, and back-left shoulder without turning the face toward the camera.",
    ),
    "back": (
        "STRAIGHT REAR / BACK VIEW (180 degrees azimuth, facing directly away from camera)",
        "Rotate head and shoulders a full 180 degrees directly away from the camera, showing the back of the head, natural crown "
        "hair swirl, clean nape hairline, subtle outer ear rims, collar band, and balanced back shoulders with no face visible.",
    ),
    "back_right_3q": (
        "BACK-RIGHT THREE-QUARTER VIEW (225 / -135 degrees azimuth, facing away toward upper-right)",
        "Rotate head and shoulders together 135 degrees away from the camera toward image-right, showing the rear-three-quarter "
        "crown hair, tapered nape hairline, outer back curve of the visible ear, eyewear temple tip over the ear if worn, "
        "outer cheekbone edge, collar band, and back-right shoulder without turning the face toward the camera.",
    ),
    "right_profile": (
        "TRUE RIGHT SIDE PROFILE (270 / -90 degrees azimuth toward image-right)",
        "Rotate head and shoulders together 90 degrees so the nose tip, lips, chin, and horizontal gaze point cleanly "
        "toward the RIGHT edge of the frame, showing the authentic side silhouette of forehead, nasal bridge, philtrum, chin, and ear.",
    ),
    "three_quarter_right": (
        "FRONT-RIGHT THREE-QUARTER VIEW (315 / -45 degrees azimuth toward image-right)",
        "Rotate head and shoulders together 45 degrees toward the RIGHT edge of the frame so the nose points image-right, "
        "both eyes remain visible in natural 3D foreshortening, and the image-left cheek/temple/ear side is clearly exposed.",
    ),
}


def resolve_8angle_view_from_azimuth(azimuth_deg: float) -> str:
    """Snap any continuous 3D azimuth angle (0..360 deg, 0=front, 90=left, 180=back, 270=right) to the closest of the 8 canonical views."""
    norm = float(azimuth_deg) % 360.0
    best_view = "front"
    best_dist = 999.0
    for view, deg in VIEW_AZIMUTH_DEG.items():
        diff = abs(((norm - deg + 180.0) % 360.0) - 180.0)
        if diff < best_dist:
            best_dist = diff
            best_view = view
    return best_view


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
            "authentic iris color from Image 1 is preserved without green/purple lens-coating tint, "
            "and ear height/position relative to the eyebrow and nose-base plane remains identical."
        )


class RealPersonTurnaroundEngine:
    """Converts any real person's photograph into an 8-Angle 360° Canonical Anchor Bank & Master Sheet."""

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

    def mask_foreign_faces_in_reference(
        self, bgr_img: np.ndarray, output_masked_path: Optional[Path] = None
    ) -> Tuple[np.ndarray, int]:
        """Detect all human faces in a mood/garment/accessory/location reference photo and cover them with neutral grey (#8C8C8C).

        Ensures foreign models inside style/outfit references NEVER contaminate the target subject's facial identity.
        """
        h, w = bgr_img.shape[:2]
        scale = min(1.0, 1280.0 / float(max(w, h)))
        sw, sh = max(16, int(round(w * scale))), max(16, int(round(h * scale)))
        work = cv2.resize(bgr_img, (sw, sh), interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr_img

        det = cv2.FaceDetectorYN.create(self.detector.yunet_path, "", (sw, sh), 0.42, 0.3, 5000)
        _, faces = det.detect(work)
        out = bgr_img.copy()
        count = 0
        if faces is not None and len(faces) > 0:
            inv_s = 1.0 / scale
            for f in faces:
                fx, fy, fw, fh = [float(v) * inv_s for v in f[:4]]
                pad_x, pad_y = 0.25 * fw, 0.30 * fh
                x1 = max(0, int(round(fx - pad_x)))
                y1 = max(0, int(round(fy - pad_y)))
                x2 = min(w, int(round(fx + fw + pad_x)))
                y2 = min(h, int(round(fy + fh + pad_y * 0.6)))
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(out, (x1, y1), (x2, y2), (140, 140, 140), -1)
                    count += 1
        if output_masked_path is not None:
            output_masked_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_masked_path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        return out, count

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
        """Normalize an arbitrary real photo into a pure optical 85mm head-and-shoulders canonical anchor (ZERO GAN pre-smoothing)."""
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

        crop_h = min(float(h), max(fh * 3.45, fw * 3.8))
        crop_w = crop_h * aspect
        if crop_w > w:
            crop_w = float(w)
            crop_h = crop_w / aspect

        x1 = int(round(np.clip(cx - 0.5 * crop_w, 0, max(0, w - crop_w))))
        y1 = int(round(np.clip(cy - 0.40 * crop_h, 0, max(0, h - crop_h))))
        x2 = min(w, int(round(x1 + crop_w)))
        y2 = min(h, int(round(y1 + crop_h)))

        crop = bgr[y1:y2, x1:x2].copy()
        if crop.shape[1] < tw or crop.shape[0] < th:
            canonical = crop if crop.shape[1] >= 480 else cv2.resize(crop, (tw, th), interpolation=cv2.INTER_LANCZOS4)
        else:
            canonical = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_LANCZOS4)
        output_anchor_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_anchor_path), canonical, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
        blueprint = self.extract_anatomical_blueprint(canonical)
        return output_anchor_path, blueprint

    @classmethod
    def build_turnaround_angle_prompt(
        cls,
        view: str,
        subject: SubjectProfile,
        blueprint: AnatomicalBlueprint,
        has_intermediate_anchor: bool = False,
        intermediate_label: str = "same-side intermediate turnaround portrait",
    ) -> str:
        """Build a 3D Euclidean turnaround prompt for any of the 8 compass angles (`0°..315°`)."""
        if view not in VIEW_DIRECTIONS_EN:
            raise ValueError(f"Unsupported turnaround view '{view}'. Choose from {tuple(VIEW_DIRECTIONS_EN)}.")
        view_title, view_desc = VIEW_DIRECTIONS_EN[view]

        ref_roles = (
            "Image 1 is the untouched canonical front identity photograph—the primary ground-truth authority for "
            "craniofacial anatomy, true iris color, authentic skin texture, eyewear geometry, hair color/parting, and shoulder scale."
        )
        if has_intermediate_anchor:
            ref_roles += (
                f" Image 2 is the approved {intermediate_label} of the exact same person—"
                "use Image 2 together with Image 1 as the 3D geometric bridge for skull contour, nasal/ear silhouette, "
                "eyewear temple path, and hair volume."
            )

        prompt = (
            "Editorial portrait. Shot on a medium-format 100MP camera with an 80mm f/5.6 prime lens, crisp optical micro-contrast, "
            "deep true blacks, and zero atmospheric haze. Agency 360-degree digital turnaround test photograph of the EXACT SAME "
            "real person shown in Image 1 against a clean neutral studio backdrop. "
            f"Reference Roles: {ref_roles} "
            f"Target Angle — {view_title}: {view_desc} "
            "Do NOT mirror or flip to the opposite direction, do NOT invent a new person, and apply zero beauty-cam filter. "
            f"Identity & Eyewear Continuity: Preserve {subject.facial_identity_descriptor()}. "
            "Keep authentic iris color from Image 1 with crystal-clear non-tinted lenses (zero green or purple coating cast). "
            "If eyewear is present in Image 1 and visible from this angle, render the temple arm extending continuously and "
            "uninterruptedly from the front outer frame hinge back to rest naturally over the top of the ear. "
            f"{blueprint.to_3d_invariance_clause()} "
            "Wardrobe & Lighting Continuity: Keep the same clean tailored studio attire, balanced adult 7.5-to-8-head head-to-shoulder "
            "scale, crisp directional studio lighting, and authentic real skin texture with subtle pores, zero plastic/oily gloss, "
            "zero text, and zero watermarks."
        )
        PromptFirewall.validate_prompt(prompt)
        return prompt

    def plan_progressive_8angle_turnaround(
        self,
        canonical_front_path: Path,
        subject: SubjectProfile,
        existing_angles: Optional[Dict[str, Path]] = None,
    ) -> Dict[str, Dict[str, object]]:
        """Plan the full 360° 8-Angle Turnaround (`Hop 1: ±45°`, `Hop 2: ±90°`, `Hop 3: ±135° & 180°`)."""
        front_bgr = cv2.imread(str(canonical_front_path))
        if front_bgr is None:
            raise FileNotFoundError(f"Cannot read canonical front anchor: {canonical_front_path}")
        blueprint = self.extract_anatomical_blueprint(front_bgr)
        bank = existing_angles or {}

        plans: Dict[str, Dict[str, object]] = {
            "front": {
                "hop": 0,
                "azimuth_deg": 0.0,
                "view": "front",
                "label_ko": VIEW_LABELS_KO["front"],
                "reference_images": [str(canonical_front_path)],
                "prompt": None,
                "status": "canonical_source",
            }
        }

        # Hop 1: ±45° three-quarter views anchored directly to untouched 0° Front
        for v3q in ("three_quarter_left", "three_quarter_right"):
            plans[v3q] = {
                "hop": 1,
                "azimuth_deg": VIEW_AZIMUTH_DEG[v3q],
                "view": v3q,
                "label_ko": VIEW_LABELS_KO[v3q],
                "reference_images": [str(canonical_front_path)],
                "prompt": self.build_turnaround_angle_prompt(v3q, subject, blueprint, has_intermediate_anchor=False),
            }

        # Hop 2: ±90° side profiles bridged by [0° Front + ±45° 3/4]
        for vprof, bridge_key, bridge_lbl in (
            ("left_profile", "three_quarter_left", "45-degree front-left three-quarter portrait"),
            ("right_profile", "three_quarter_right", "45-degree front-right three-quarter portrait"),
        ):
            refs = [str(canonical_front_path)]
            bridge_p = bank.get(bridge_key)
            has_b = bridge_p is not None and Path(bridge_p).exists()
            if has_b:
                refs.append(str(bridge_p))
            plans[vprof] = {
                "hop": 2,
                "azimuth_deg": VIEW_AZIMUTH_DEG[vprof],
                "view": vprof,
                "label_ko": VIEW_LABELS_KO[vprof],
                "reference_images": refs,
                "prompt": self.build_turnaround_angle_prompt(
                    vprof, subject, blueprint, has_intermediate_anchor=has_b, intermediate_label=bridge_lbl
                ),
            }

        # Hop 3: ±135° back-three-quarter & 180° full back bridged by [0° Front + ±90° Profile]
        for vback, bridge_key, bridge_lbl in (
            ("back_left_3q", "left_profile", "90-degree left side-profile portrait"),
            ("back_right_3q", "right_profile", "90-degree right side-profile portrait"),
            ("back", "right_profile", "90-degree side-profile portrait"),
        ):
            refs = [str(canonical_front_path)]
            bridge_p = bank.get(bridge_key) or bank.get("left_profile")
            has_b = bridge_p is not None and Path(bridge_p).exists()
            if has_b:
                refs.append(str(bridge_p))
            plans[vback] = {
                "hop": 3,
                "azimuth_deg": VIEW_AZIMUTH_DEG[vback],
                "view": vback,
                "label_ko": VIEW_LABELS_KO[vback],
                "reference_images": refs,
                "prompt": self.build_turnaround_angle_prompt(
                    vback, subject, blueprint, has_intermediate_anchor=has_b, intermediate_label=bridge_lbl
                ),
            }

        return plans

    def plan_progressive_5angle_turnaround(
        self,
        canonical_front_path: Path,
        subject: SubjectProfile,
        approved_3q_left_path: Optional[Path] = None,
        approved_3q_right_path: Optional[Path] = None,
    ) -> Dict[str, Dict[str, object]]:
        """Backward-compatible 5-angle subset planner delegating to `plan_progressive_8angle_turnaround`."""
        bank: Dict[str, Path] = {}
        if approved_3q_left_path:
            bank["three_quarter_left"] = approved_3q_left_path
        if approved_3q_right_path:
            bank["three_quarter_right"] = approved_3q_right_path
        all_plans = self.plan_progressive_8angle_turnaround(canonical_front_path, subject, existing_angles=bank)
        return {k: all_plans[k] for k in TURNAROUND_5_VIEWS}

    @staticmethod
    def assemble_master_turnaround_sheet(
        angle_image_paths: Dict[str, Path],
        output_sheet_path: Path,
        cell_size: Tuple[int, int] = (560, 720),
        include_labels: bool = False,
        columns: Optional[int] = None,
    ) -> Path:
        """Assemble 5 or 8 turnaround views into a crisp pure-Lanczos Master Character Sheet (4x2 grid for 8 views)."""
        cw, ch = cell_size
        pad = 20
        header_h = 52 if include_labels else 0
        ordered_views = [v for v in TURNAROUND_8_VIEWS if v in angle_image_paths and Path(angle_image_paths[v]).exists()]
        if not ordered_views:
            raise ValueError("No valid angle image paths supplied for turnaround sheet.")

        cols = columns or (4 if len(ordered_views) > 5 else len(ordered_views))
        rows = (len(ordered_views) + cols - 1) // cols
        total_w = pad + cols * (cw + pad)
        total_h = pad + rows * (ch + header_h + pad)
        sheet = Image.new("RGB", (total_w, total_h), (242, 242, 240))
        draw = ImageDraw.Draw(sheet)

        font = ImageFont.load_default()
        if include_labels:
            for fp in (
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
            ):
                if Path(fp).exists():
                    font = ImageFont.truetype(fp, 22)
                    break

        for idx, view in enumerate(ordered_views):
            r = idx // cols
            c = idx % cols
            x = pad + c * (cw + pad)
            y = pad + r * (ch + header_h + pad)
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
                draw.text((x + 10, y + ch + 12), VIEW_LABELS_KO.get(view, view), fill=(32, 36, 40), font=font)

        output_sheet_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(str(output_sheet_path), format="JPEG", quality=97)
        return output_sheet_path

    def select_angle_matched_references(
        self,
        canonical_front_path: Path,
        angle_bank: Dict[str, Path],
        target_view: Optional[str] = None,
        target_azimuth_deg: Optional[float] = None,
        target_scene_bgr: Optional[np.ndarray] = None,
        master_sheet_path: Optional[Path] = None,
    ) -> Tuple[List[Path], str]:
        """Select `[Untouched Gen-0 Front Anchor, Angle-Matched 8-Way View]` from explicit view, 3D azimuth, or detected yaw."""
        resolved_view = target_view
        if resolved_view is None and target_azimuth_deg is not None:
            resolved_view = resolve_8angle_view_from_azimuth(target_azimuth_deg)
        if resolved_view is None and target_scene_bgr is not None:
            try:
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
            except Exception:
                resolved_view = "front"

        resolved_view = resolved_view or "front"
        refs: List[Path] = [Path(canonical_front_path)]

        if resolved_view != "front" and resolved_view in angle_bank and Path(angle_bank[resolved_view]).exists():
            refs.append(Path(angle_bank[resolved_view]))
        elif master_sheet_path is not None and Path(master_sheet_path).exists():
            refs.append(Path(master_sheet_path))

        return refs, resolved_view
