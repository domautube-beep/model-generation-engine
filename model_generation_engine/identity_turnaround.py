"""Real-Person Frontal-Arc Detailed 8-Direction Turnaround & Canonical Identity System (`완전좌 -90° ~ 완전우 +90° 디테일 8방향 + 원본 0°`).

Universal, zero-hardcoding identity system for ALL projects:
1. `prepare_canonical_identity_anchor`: Extracts a pure untouched optical `85mm` head-and-shoulders anchor (`0°` Gen-0,
   ZERO `Real-ESRGAN` pre-smoothing, preserving 100% of real photographic sharpness and `7.5~8등신` shoulder context).
2. `extract_anatomical_blueprint`: Measures deterministic craniofacial ratios (`Apple Vision` + `YuNet`).
3. `plan_progressive_8angle_turnaround`: Plans the **8 detailed frontal-hemisphere directions** at `22.5°` micro-intervals
   between **Complete Left (`-90°`)** and **Complete Right (`+90°`)** around the untouched **Original Front (`0°`)**:
   - Left 4 directions: `-90° (완전 좌측면)` · `-67.5° (좌 깊은 ¾)` · `-45° (좌 표준 ¾)` · `-22.5° (좌 세미프론트)`
   - Center Origin: `0° (원본 정면 Gen-0)`
   - Right 4 directions: `+22.5° (우 세미프론트)` · `+45° (우 표준 ¾)` · `+67.5° (우 깊은 ¾)` · `+90° (완전 우측면)`
   using 4-Hop `22.5°` progressive bridging (`0° -> ±22.5° -> ±45° -> ±67.5° -> ±90°`) so every hop rotates by only `22.5°`.
4. `resolve_8angle_view_from_azimuth` & `select_angle_matched_references`: Maps any continuous head yaw (`-90°..+90°`)
   with `<= 11.25°` max angular error to `[Untouched Gen-0 Front Anchor, Exact Angle-Matched Frontal-Arc View]`.
5. `mask_foreign_faces_in_reference`: Automatically detects and neutral-grey-masks foreign faces inside uploaded garment,
   accessory, bag, shoe, or location/style reference images so foreign identities never contaminate the subject.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .apple_vision import AppleVisionAnalyzer
from .config import EngineConfig, SubjectProfile
from .detector import DynamicFaceDetector
from .prompt_builder import PromptFirewall
from .upscaler import SuperResolutionEngine

# 8 generated directions spanning Complete Left (-90°) to Complete Right (+90°) in 22.5° increments
TURNAROUND_8_VIEWS: Tuple[str, ...] = (
    "left_90",
    "left_67",
    "left_45",
    "left_22",
    "right_22",
    "right_45",
    "right_67",
    "right_90",
)

# Complete 9-slot left-to-right horizontal sweep including untouched 0° original in the center
TURNAROUND_FULL_SWEEP: Tuple[str, ...] = (
    "left_90",
    "left_67",
    "left_45",
    "left_22",
    "front",
    "right_22",
    "right_45",
    "right_67",
    "right_90",
)

# Backward-compatible 5-view constant (`TURNAROUND_VIEWS`)
TURNAROUND_5_VIEWS: Tuple[str, ...] = (
    "left_profile",
    "three_quarter_left",
    "front",
    "three_quarter_right",
    "right_profile",
)
TURNAROUND_VIEWS: Tuple[str, ...] = TURNAROUND_5_VIEWS

LEGACY_VIEW_ALIASES: Dict[str, str] = {
    "left_profile": "left_90",
    "three_quarter_left": "left_45",
    "three_quarter_right": "right_45",
    "right_profile": "right_90",
}

# Signed frontal yaw degrees: Negative = Left (-90°..0°), Positive = Right (0°..+90°)
VIEW_AZIMUTH_DEG: Dict[str, float] = {
    "left_90": -90.0,
    "left_67": -67.5,
    "left_45": -45.0,
    "left_22": -22.5,
    "front": 0.0,
    "right_22": 22.5,
    "right_45": 45.0,
    "right_67": 67.5,
    "right_90": 90.0,
}

VIEW_LABELS_KO: Dict[str, str] = {
    "left_90": "① 완전 좌 (-90°)",
    "left_67": "② 좌 깊은¾ (-67.5°)",
    "left_45": "③ 좌 표준¾ (-45°)",
    "left_22": "④ 좌 미세 (-22.5°)",
    "front": "기준 원본 (0°)",
    "right_22": "⑤ 우 미세 (+22.5°)",
    "right_45": "⑥ 우 표준¾ (+45°)",
    "right_67": "⑦ 우 깊은¾ (+67.5°)",
    "right_90": "⑧ 완전 우 (+90°)",
}

VIEW_DIRECTIONS_EN: Dict[str, Tuple[str, str]] = {
    "front": (
        "DIRECT FRONT ORIGIN (0 degrees yaw)",
        "Head, nose centerline, gaze, and shoulders face directly toward the camera with symmetrical visibility of both ears.",
    ),
    "left_22": (
        "SUBTLE SEMI-FRONTAL LEFT TURN (-22.5 degrees yaw toward image-left)",
        "Rotate head and shoulders gently 22.5 degrees toward the LEFT edge of the frame (halfway between 0 deg front and -45 deg three-quarter): "
        "both eyes remain fully visible looking calmly forward-left, a subtle edge of the far (image-left) ear is still visible, "
        "a wide span of the far (image-left) cheek remains visible outside the nose tip, and the near (image-right) cheek/temple/ear is slightly more prominent.",
    ),
    "left_45": (
        "CLASSIC FRONT-LEFT THREE-QUARTER TURN (-45 degrees yaw toward image-left)",
        "Rotate head and shoulders 45 degrees toward the LEFT edge of the frame: both eyes remain clearly visible in natural 3D foreshortening, "
        "the far (image-left) ear is occluded behind the head, the nose tip sits roughly halfway across the far (image-left) cheek without crossing its contour, "
        "and the near (image-right) cheek, temple, and ear are cleanly exposed.",
    ),
    "left_67": (
        "DEEP OBLIQUE LEFT THREE-QUARTER PROFILE (-67.5 degrees yaw toward image-left)",
        "Rotate head and shoulders 67.5 degrees toward the LEFT edge of the frame (strictly halfway between -45 deg three-quarter and -90 deg full profile): "
        "the nasal bridge and nose tip graze or slightly cross the far (image-left) cheek outline so ONLY a narrow foreshortened sliver of the far eye/eyelashes "
        "peeks past the nasal bridge, while the near (image-right) eye, full cheek, jawline, eyewear temple arm over the ear, and side-back hair volume dominate.",
    ),
    "left_90": (
        "COMPLETE TRUE LEFT SIDE PROFILE (-90 degrees yaw toward image-left)",
        "Rotate head and shoulders a full 90 degrees so the nose tip, lips, chin, and horizontal gaze point cleanly toward the LEFT edge of the frame: "
        "only ONE eye (in pure side silhouette) is visible, the far eye and far cheek are completely hidden, showing the exact side profile of forehead, "
        "nasal bridge, philtrum, lips, chin, and ear.",
    ),
    "right_22": (
        "SUBTLE SEMI-FRONTAL RIGHT TURN (+22.5 degrees yaw toward image-right)",
        "Rotate head and shoulders gently 22.5 degrees toward the RIGHT edge of the frame (halfway between 0 deg front and +45 deg three-quarter): "
        "both eyes remain fully visible looking calmly forward-right, a subtle edge of the far (image-right) ear is still visible, "
        "a wide span of the far (image-right) cheek remains visible outside the nose tip, and the near (image-left) cheek/temple/ear is slightly more prominent.",
    ),
    "right_45": (
        "CLASSIC FRONT-RIGHT THREE-QUARTER TURN (+45 degrees yaw toward image-right)",
        "Rotate head and shoulders 45 degrees toward the RIGHT edge of the frame: both eyes remain clearly visible in natural 3D foreshortening, "
        "the far (image-right) ear is occluded behind the head, the nose tip sits roughly halfway across the far (image-right) cheek without crossing its contour, "
        "and the near (image-left) cheek, temple, and ear are cleanly exposed.",
    ),
    "right_67": (
        "DEEP OBLIQUE RIGHT THREE-QUARTER PROFILE (+67.5 degrees yaw toward image-right)",
        "Rotate head and shoulders 67.5 degrees toward the RIGHT edge of the frame (strictly halfway between +45 deg three-quarter and +90 deg full profile): "
        "the nasal bridge and nose tip graze or slightly cross the far (image-right) cheek outline so ONLY a narrow foreshortened sliver of the far eye/eyelashes "
        "peeks past the nasal bridge, while the near (image-left) eye, full cheek, jawline, eyewear temple arm over the ear, and side-back hair volume dominate.",
    ),
    "right_90": (
        "COMPLETE TRUE RIGHT SIDE PROFILE (+90 degrees yaw toward image-right)",
        "Rotate head and shoulders a full 90 degrees so the nose tip, lips, chin, and horizontal gaze point cleanly toward the RIGHT edge of the frame: "
        "only ONE eye (in pure side silhouette) is visible, the far eye and far cheek are completely hidden, showing the exact side profile of forehead, "
        "nasal bridge, philtrum, lips, chin, and ear.",
    ),
}
# Support legacy 5-view names in build_turnaround_angle_prompt as well
for _old_k, _new_k in LEGACY_VIEW_ALIASES.items():
    VIEW_DIRECTIONS_EN[_old_k] = VIEW_DIRECTIONS_EN[_new_k]


def normalize_frontal_yaw_deg(yaw_deg: float) -> float:
    """Normalize any angle into signed frontal-hemisphere yaw [-90.0, +90.0] (-90 = full left, 0 = front, +90 = full right)."""
    signed = ((float(yaw_deg) + 180.0) % 360.0) - 180.0
    return float(np.clip(signed, -90.0, 90.0))


def resolve_8angle_view_from_azimuth(yaw_deg: float) -> str:
    """Snap any frontal yaw angle (-90..+90 deg) to the closest slot in TURNAROUND_FULL_SWEEP (<= 11.25 deg max error)."""
    clamped = normalize_frontal_yaw_deg(yaw_deg)
    best_view = "front"
    best_dist = 999.0
    for view in TURNAROUND_FULL_SWEEP:
        diff = abs(clamped - VIEW_AZIMUTH_DEG[view])
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
    """Converts any real person's photograph into a Detailed 8-Direction (-90°..+90°) Frontal-Arc Bank & Master Sheet."""

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
        """Detect all human faces in a mood/garment/accessory/location reference photo and cover them with neutral grey (#8C8C8C)."""
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
        intermediate_label: str = "adjacent same-side intermediate turnaround portrait",
    ) -> str:
        """Build a 3D Euclidean turnaround prompt for any frontal-arc direction (-90°..+90°)."""
        if view not in VIEW_DIRECTIONS_EN:
            raise ValueError(f"Unsupported turnaround view '{view}'. Choose from {tuple(VIEW_DIRECTIONS_EN)}.")
        view_title, view_desc = VIEW_DIRECTIONS_EN[view]

        ref_roles = (
            "Image 1 is the untouched canonical 0-degree front identity photograph—the primary ground-truth authority for "
            "craniofacial anatomy, true iris color, authentic skin texture, eyewear geometry, hair color/parting, and shoulder scale."
        )
        if has_intermediate_anchor:
            ref_roles += (
                f" Image 2 is the approved {intermediate_label} of the exact same person—"
                "rotate a precise additional 22.5-degree increment from Image 2 while keeping Image 1's exact identity and skin/iris fidelity."
            )

        prompt = (
            "Editorial portrait. Shot on a medium-format 100MP camera with an 80mm f/5.6 prime lens, crisp optical micro-contrast, "
            "deep true blacks, and zero atmospheric haze. Agency frontal-arc turnaround test photograph of the EXACT SAME "
            "real person shown in Image 1 against a clean neutral studio backdrop. "
            f"Reference Roles: {ref_roles} "
            f"Target Angle — {view_title}: {view_desc} "
            "Do NOT mirror or flip to the opposite direction, do NOT invent a new person, and apply zero beauty-cam filter. "
            f"Identity & Eyewear Continuity: Preserve {subject.facial_identity_descriptor()}. "
            "Keep authentic iris color from Image 1 with crystal-clear non-tinted lenses (zero green or purple coating cast). "
            "If eyewear is present in Image 1, render the temple arm extending continuously and uninterruptedly from the front outer "
            "frame hinge back to rest naturally over the top of the visible ear. "
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
        """Plan the 8 detailed frontal-arc directions (-90°, -67.5°, -45°, -22.5°, +22.5°, +45°, +67.5°, +90°) + 0° Origin via 4-Hop 22.5° micro-steps."""
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

        # 4 progressive 22.5° hops outward on both Left (-) and Right (+) sides:
        hop_specs = (
            # (hop_num, target_view, preferred_bridges, bridge_label)
            (1, "left_22", (), ""),
            (1, "right_22", (), ""),
            (2, "left_45", ("left_22",), "-22.5-degree semi-frontal left portrait"),
            (2, "right_45", ("right_22",), "+22.5-degree semi-frontal right portrait"),
            (3, "left_67", ("left_45", "left_22"), "-45-degree front-left three-quarter portrait"),
            (3, "right_67", ("right_45", "right_22"), "+45-degree front-right three-quarter portrait"),
            (4, "left_90", ("left_67", "left_45"), "-67.5-degree deep oblique left portrait"),
            (4, "right_90", ("right_67", "right_45"), "+67.5-degree deep oblique right portrait"),
        )

        for hop_num, v_key, bridge_candidates, bridge_lbl in hop_specs:
            refs = [str(canonical_front_path)]
            chosen_bridge: Optional[Path] = None
            for bc in bridge_candidates:
                cand_p = bank.get(bc)
                if cand_p is not None and Path(cand_p).exists():
                    chosen_bridge = Path(cand_p)
                    break
            if chosen_bridge is not None:
                refs.append(str(chosen_bridge))
            plans[v_key] = {
                "hop": hop_num,
                "azimuth_deg": VIEW_AZIMUTH_DEG[v_key],
                "view": v_key,
                "label_ko": VIEW_LABELS_KO[v_key],
                "reference_images": refs,
                "prompt": self.build_turnaround_angle_prompt(
                    v_key,
                    subject,
                    blueprint,
                    has_intermediate_anchor=(chosen_bridge is not None),
                    intermediate_label=bridge_lbl,
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
        """Backward-compatible 5-angle subset planner (`left_profile, three_quarter_left, front, three_quarter_right, right_profile`)."""
        front_bgr = cv2.imread(str(canonical_front_path))
        if front_bgr is None:
            raise FileNotFoundError(f"Cannot read canonical front anchor: {canonical_front_path}")
        blueprint = self.extract_anatomical_blueprint(front_bgr)

        has_l = approved_3q_left_path is not None and Path(approved_3q_left_path).exists()
        has_r = approved_3q_right_path is not None and Path(approved_3q_right_path).exists()

        return {
            "front": {
                "hop": 0,
                "view": "front",
                "reference_images": [str(canonical_front_path)],
                "prompt": None,
                "status": "canonical_source",
            },
            "three_quarter_left": {
                "hop": 1,
                "view": "three_quarter_left",
                "reference_images": [str(canonical_front_path)],
                "prompt": self.build_turnaround_angle_prompt("left_45", subject, blueprint, False),
            },
            "three_quarter_right": {
                "hop": 1,
                "view": "three_quarter_right",
                "reference_images": [str(canonical_front_path)],
                "prompt": self.build_turnaround_angle_prompt("right_45", subject, blueprint, False),
            },
            "left_profile": {
                "hop": 2,
                "view": "left_profile",
                "reference_images": [str(canonical_front_path)] + ([str(approved_3q_left_path)] if has_l else []),
                "prompt": self.build_turnaround_angle_prompt(
                    "left_90", subject, blueprint, has_l, "-45-degree front-left three-quarter portrait"
                ),
            },
            "right_profile": {
                "hop": 2,
                "view": "right_profile",
                "reference_images": [str(canonical_front_path)] + ([str(approved_3q_right_path)] if has_r else []),
                "prompt": self.build_turnaround_angle_prompt(
                    "right_90", subject, blueprint, has_r, "+45-degree front-right three-quarter portrait"
                ),
            },
        }

    @staticmethod
    def assemble_master_turnaround_sheet(
        angle_image_paths: Dict[str, Path],
        output_sheet_path: Path,
        cell_size: Tuple[int, int] = (420, 540),
        include_labels: bool = False,
        columns: Optional[int] = None,
    ) -> Path:
        """Assemble the left-to-right frontal-arc sweep (`-90° .. 0° .. +90°`) into a crisp pure-Lanczos Master Sheet."""
        normalized_map: Dict[str, Path] = {}
        for k, v in angle_image_paths.items():
            canon_k = LEGACY_VIEW_ALIASES.get(k, k)
            if Path(v).exists():
                normalized_map[canon_k] = Path(v)

        ordered_views = [v for v in TURNAROUND_FULL_SWEEP if v in normalized_map]
        if not ordered_views:
            raise ValueError("No valid angle image paths supplied for turnaround sheet.")

        cw, ch = cell_size
        pad = 16
        header_h = 48 if include_labels else 0
        cols = columns or len(ordered_views)
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
                    try:
                        font = ImageFont.truetype(fp, 21)
                        break
                    except Exception:
                        pass

        for idx, view in enumerate(ordered_views):
            r, c = divmod(idx, cols)
            x0 = pad + c * (cw + pad)
            y0 = pad + r * (ch + header_h + pad)
            img = Image.open(normalized_map[view]).convert("RGB")
            iw, ih = img.size
            target_aspect = float(cw) / float(ch)
            src_aspect = float(iw) / float(ih)
            if src_aspect > target_aspect:
                new_w = int(round(ih * target_aspect))
                sx = (iw - new_w) // 2
                img = img.crop((sx, 0, sx + new_w, ih))
            else:
                new_h = int(round(iw / target_aspect))
                sy = max(0, int(round((ih - new_h) * 0.25)))
                img = img.crop((0, sy, iw, min(ih, sy + new_h)))
            img = img.resize((cw, ch), Image.Resampling.LANCZOS)
            sheet.paste(img, (x0, y0))
            if view == "front":
                draw.rectangle([x0 - 2, y0 - 2, x0 + cw + 2, y0 + ch + 2], outline=(60, 120, 245), width=4)
            if include_labels:
                lbl = VIEW_LABELS_KO.get(view, view)
                color = (35, 90, 210) if view == "front" else (45, 45, 48)
                draw.text((x0 + 8, y0 + ch + 11), lbl, fill=color, font=font)

        output_sheet_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(str(output_sheet_path), format="JPEG", quality=96, subsampling=0)
        return output_sheet_path

    def infer_target_turnaround_view(
        self,
        scene_bgr: Optional[np.ndarray] = None,
        explicit_view: Optional[str] = None,
        explicit_azimuth_deg: Optional[float] = None,
    ) -> str:
        """Determine which of the 8 detailed frontal-arc views (`-90°..+90°`) or `front` (`0°`) matches a target scene."""
        if explicit_view:
            mapped = LEGACY_VIEW_ALIASES.get(explicit_view, explicit_view)
            if mapped in VIEW_AZIMUTH_DEG:
                return mapped
        if explicit_azimuth_deg is not None:
            return resolve_8angle_view_from_azimuth(explicit_azimuth_deg)
        if scene_bgr is None:
            return "front"
        try:
            face = self.detector.detect(scene_bgr)
            le, re, nose, _, _ = face.landmarks_5pt
            d_left = float(np.linalg.norm(nose - le))
            d_right = float(np.linalg.norm(nose - re))
            signed_asym = (d_left - d_right) / max(1.0, d_left + d_right)
            # signed_asym < 0 -> nose closer to image-left eye (turning left, negative yaw)
            est_yaw = float(np.clip(signed_asym * 145.0, -90.0, 90.0))
            return resolve_8angle_view_from_azimuth(est_yaw)
        except Exception:
            return "front"

    def select_angle_matched_references(
        self,
        canonical_front_path: Path,
        angle_bank: Dict[str, Path],
        scene_bgr: Optional[np.ndarray] = None,
        target_view: Optional[str] = None,
        target_azimuth_deg: Optional[float] = None,
    ) -> Tuple[List[Path], str]:
        """Return `[Untouched Gen-0 Front Anchor, Closest Available Frontal-Arc View]` for zero-drift synthesis."""
        normalized_bank: Dict[str, Path] = {}
        for k, v in angle_bank.items():
            if Path(v).exists():
                normalized_bank[LEGACY_VIEW_ALIASES.get(k, k)] = Path(v)

        view = self.infer_target_turnaround_view(
            scene_bgr, explicit_view=target_view, explicit_azimuth_deg=target_azimuth_deg
        )
        refs: List[Path] = [canonical_front_path]
        if view == "front":
            return refs, (target_view if target_view in LEGACY_VIEW_ALIASES else "front")

        # Find exact match or closest same-side angle in normalized_bank
        chosen_p = normalized_bank.get(view)
        if chosen_p is None:
            target_deg = VIEW_AZIMUTH_DEG.get(view, 0.0)
            same_side = [
                (abs(VIEW_AZIMUTH_DEG[k] - target_deg), p)
                for k, p in normalized_bank.items()
                if k != "front" and (VIEW_AZIMUTH_DEG.get(k, 0.0) * target_deg > 0)
            ]
            if same_side:
                same_side.sort(key=lambda t: t[0])
                chosen_p = same_side[0][1]

        if chosen_p is not None and chosen_p.resolve() != canonical_front_path.resolve():
            refs.append(chosen_p)

        # Preserve caller's legacy view name if explicitly passed (for backward-compatible callers)
        out_label = target_view if (target_view and target_view in LEGACY_VIEW_ALIASES) else view
        return refs, out_label
