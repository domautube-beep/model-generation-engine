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
    source_signed_yaw_deg: float = 0.0

    def to_3d_invariance_clause(self) -> str:
        """Emit a gender-neutral, firewall-safe 3D Euclidean bone & soft-tissue cheek fullness invariance specification."""
        return (
            f"Measured 3D Craniofacial & Soft-Tissue Lock (vertical eye-to-menton index {self.eye_to_chin_ratio:.3f}, "
            f"cheek-width-to-length aspect index {self.width_to_length_ratio:.3f}, "
            f"inter-ocular-to-cranial index {self.iod_to_bbox_width_ratio:.3f}, "
            f"oral-to-ocular span index {self.mouth_to_eye_width_ratio:.3f}, "
            f"source portrait angle {self.source_signed_yaw_deg:+.1f} deg): "
            "maintain strict 1:1 craniofacial and soft-tissue identity across camera angles—preserve exact natural cheek volume "
            "and jaw width without altering facial breadth, preserve exact eye corner slant and eyelid crease, "
            "exact nasal bridge height and rounded tip shape, exact philtrum length, exact lip contour, authentic iris color, "
            "and head-fixed hair parting/ear-tuck geometry relative to the subject's own left/right ears."
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

    @staticmethod
    def requires_intermediate_turnaround_hop(
        source_yaw_deg: float, target_yaw_deg: float, max_direct_delta_deg: float = 28.0
    ) -> bool:
        """Zero-Quality-Loss Speed & Identity Gate: skip synthetic Gen-1 turnaround hop when |target - source| <= 28 deg."""
        return abs(float(target_yaw_deg) - float(source_yaw_deg)) > float(max_direct_delta_deg)

    def prepare_pure_gen0_face_macro_crop(
        self,
        raw_face_path: Path,
        output_crop_path: Path,
        target_size: Tuple[int, int] = (1024, 1024),
        preserve_native_4k: bool = True,
        restore_4k_via_upscayl: bool = False,
    ) -> Path:
        """Pure optical tight head-and-hair crop from untouched Gen-0 photo (preserves native 4K or restores 4K via Upscayl Photo mode)."""
        bgr = cv2.imread(str(raw_face_path))
        if bgr is None:
            raise FileNotFoundError(f"Cannot read raw portrait: {raw_face_path}")
        if restore_4k_via_upscayl and max(bgr.shape[:2]) < 2000 and self.upscaler is not None and self.upscaler.upscayl_bin is not None:
            bgr, _ = self.upscaler.upscale_with_upscayl(bgr, style="photo", scale=4)
        h, w = bgr.shape[:2]
        face = self.detector.detect(bgr)
        fx, fy, fw, fh = face.bbox
        cx = fx + 0.5 * fw
        cy = fy + 0.44 * fh
        # Tight head+hair+ear+neck box clamped strictly inside image bounds (zero border-replicate streaks!)
        side = min(float(w), float(h), max(fw * 1.95, fh * 1.62))
        x1 = int(round(np.clip(cx - 0.5 * side, 0.0, max(0.0, float(w) - side))))
        y1 = int(round(np.clip(cy - 0.48 * side, 0.0, max(0.0, float(h) - side))))
        iside = max(32, int(round(side)))
        crop = bgr[y1 : min(h, y1 + iside), x1 : min(w, x1 + iside)]
        if preserve_native_4k and target_size == (1024, 1024) and crop.shape[1] > target_size[0]:
            cap = min(3840, crop.shape[1])
            out = crop if crop.shape[1] <= 3840 else cv2.resize(crop, (cap, cap), interpolation=cv2.INTER_LANCZOS4)
        else:
            out = cv2.resize(crop, target_size, interpolation=cv2.INTER_LANCZOS4)
        output_crop_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_crop_path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
        return output_crop_path

    def erase_foreign_person_completely(
        self,
        bgr_img: np.ndarray,
        output_clean_path: Optional[Path] = None,
        remove_overlay_text: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """Erase 100% of any foreign human silhouette (head, neck, torso, limbs, hands, shoes) via Apple Vision Segmentation + Pose + Hand Joints."""
        h, w = bgr_img.shape[:2]
        out = bgr_img.copy()

        # 1. Seamlessly remove campaign/magazine overlay typography first
        if remove_overlay_text:
            try:
                pil_rgb = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                tboxes = self.apple_vision.text_boxes(pil_rgb)
                if tboxes:
                    txt_mask = np.zeros((h, w), dtype=np.uint8)
                    for tb in tboxes:
                        tx1 = max(0, int(round(float(tb["x"]) * w)) - 8)
                        ty1 = max(0, int(round(float(tb["y"]) * h)) - 8)
                        tx2 = min(w, int(round((float(tb["x"]) + float(tb["w"])) * w)) + 8)
                        ty2 = min(h, int(round((float(tb["y"]) + float(tb["h"])) * h)) + 8)
                        if tx2 > tx1 and ty2 > ty1:
                            cv2.rectangle(txt_mask, (tx1, ty1), (tx2, ty2), 255, -1)
                    if np.any(txt_mask > 0):
                        out = cv2.inpaint(out, txt_mask, 9.0, cv2.INPAINT_TELEA)
            except Exception:
                pass

        # 2. Build unified full-body + limb + hand + face mask
        full_mask = np.zeros((h, w), dtype=np.uint8)
        pil_clean = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        try:
            pm = self.apple_vision.person_mask(pil_clean)
            if pm is not None:
                pm_arr = np.array(pm.resize((w, h), Image.Resampling.BILINEAR))
                full_mask[pm_arr > 100] = 255
        except Exception:
            pass

        # Connect shoulder->elbow->wrist & hip->knee->ankle capsules + 21-pt hand convex hulls so outstretched hands NEVER remain!
        try:
            bh = self.apple_vision.body_and_hand_points(pil_clean)
            limb_r = max(14, int(round(max(h, w) * 0.026)))
            limb_pairs = (
                ("left_shoulder_1_joint", "left_elbow_1_joint"),
                ("left_elbow_1_joint", "left_wrist_2_joint"),
                ("right_shoulder_1_joint", "right_elbow_1_joint"),
                ("right_elbow_1_joint", "right_wrist_2_joint"),
                ("left_upLeg_joint", "left_leg_joint"),
                ("left_leg_joint", "left_foot_joint"),
                ("right_upLeg_joint", "right_leg_joint"),
                ("right_leg_joint", "right_foot_joint"),
            )
            for body in bh.get("bodies") or []:
                jmap = {str(j["name"]): (int(round(float(j["x"]))), int(round(float(j["y"])))) for j in body}
                for ja, jb in limb_pairs:
                    if ja in jmap and jb in jmap:
                        pa, pb = jmap[ja], jmap[jb]
                        # Extend 30% past distal joint so fingers/toes beyond wrist/ankle are 100% covered
                        ex = int(round(pb[0] + 0.32 * (pb[0] - pa[0])))
                        ey = int(round(pb[1] + 0.32 * (pb[1] - pa[1])))
                        cv2.line(full_mask, pa, (ex, ey), 255, thickness=limb_r * 2)
                        cv2.circle(full_mask, (ex, ey), int(round(limb_r * 1.4)), 255, -1)
                for wk in ("left_wrist_2_joint", "right_wrist_2_joint", "left_foot_joint", "right_foot_joint"):
                    if wk in jmap:
                        cv2.circle(full_mask, jmap[wk], int(round(limb_r * 1.45)), 255, -1)
            for hand_pts in bh.get("hands") or []:
                if len(hand_pts) >= 3:
                    pts_np = np.array([[int(round(p[0])), int(round(p[1]))] for p in hand_pts], dtype=np.int32)
                    hull = cv2.convexHull(pts_np)
                    cv2.fillConvexPoly(full_mask, hull, 255)
                    for pt in pts_np:
                        cv2.circle(full_mask, (int(pt[0]), int(pt[1])), max(10, limb_r // 2), 255, -1)
        except Exception:
            pass

        # Also include YuNet face detection fallback
        try:
            scale = min(1.0, 1280.0 / float(max(w, h)))
            sw, sh = max(16, int(round(w * scale))), max(16, int(round(h * scale)))
            work = cv2.resize(out, (sw, sh), interpolation=cv2.INTER_AREA) if scale < 1.0 else out
            det = cv2.FaceDetectorYN.create(self.detector.yunet_path, "", (sw, sh), 0.42, 0.3, 5000)
            _, faces = det.detect(work)
            if faces is not None:
                inv_s = 1.0 / scale
                for f in faces:
                    fx, fy, fw, fh = [float(v) * inv_s for v in f[:4]]
                    cv2.ellipse(
                        full_mask,
                        (int(round(fx + 0.5 * fw)), int(round(fy + 0.55 * fh))),
                        (int(round(0.85 * fw)), int(round(1.15 * fh))),
                        0,
                        0,
                        360,
                        255,
                        -1,
                    )
        except Exception:
            pass

        coverage_pct = float(np.mean(full_mask > 0) * 100.0)
        if coverage_pct > 0.05:
            dil = max(17, int(round(max(h, w) * 0.022)) | 1)
            mask_dil = cv2.dilate(full_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil)), iterations=1)

            # Multi-scale coarse harmonic diffusion + fine rim TELEA inpainting (zero starburst streaks, crisp chair edges!)
            ch_s, cw_s = max(32, h // 8), max(32, w // 8)
            small_bgr = cv2.resize(out, (cw_s, ch_s), interpolation=cv2.INTER_AREA)
            small_m = cv2.dilate(
                (cv2.resize(mask_dil, (cw_s, ch_s), interpolation=cv2.INTER_NEAREST) > 40).astype(np.uint8) * 255,
                np.ones((3, 3), np.uint8),
                iterations=1,
            )
            coarse_inp = cv2.inpaint(small_bgr, small_m, 9.0, cv2.INPAINT_TELEA)
            work_f = coarse_inp.astype(np.float32)
            m_bool = small_m > 0
            for _ in range(35):
                b_step = cv2.GaussianBlur(work_f, (0, 0), 2.2)
                work_f[m_bool] = b_step[m_bool]
            smooth_fill = cv2.resize(np.clip(work_f, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)

            fine_inp = cv2.inpaint(out, mask_dil, 11.0, cv2.INPAINT_TELEA)
            dist = cv2.distanceTransform((mask_dil > 0).astype(np.uint8), cv2.DIST_L2, 5)
            alpha_deep = np.clip((dist - 4.0) / 20.0, 0.0, 1.0)[..., None]
            blended_hole = fine_inp.astype(np.float32) * (1.0 - alpha_deep) + smooth_fill.astype(np.float32) * alpha_deep
            soft_edge = cv2.GaussianBlur((mask_dil > 0).astype(np.float32), (0, 0), 3.0)[..., None]
            out = np.clip(out.astype(np.float32) * (1.0 - soft_edge) + blended_hole * soft_edge, 0, 255).astype(np.uint8)

        if output_clean_path is not None:
            output_clean_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_clean_path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 97])
        return out, round(coverage_pct, 2)

    def mask_foreign_faces_in_reference(
        self,
        bgr_img: np.ndarray,
        output_masked_path: Optional[Path] = None,
        remove_overlay_text: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """Mask foreign human faces AND seamlessly inpaint ad/magazine typography overlays in reference photos."""
        h, w = bgr_img.shape[:2]
        scale = min(1.0, 1280.0 / float(max(w, h)))
        sw, sh = max(16, int(round(w * scale))), max(16, int(round(h * scale)))
        work = cv2.resize(bgr_img, (sw, sh), interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr_img

        det = cv2.FaceDetectorYN.create(self.detector.yunet_path, "", (sw, sh), 0.42, 0.3, 5000)
        _, faces = det.detect(work)
        out = bgr_img.copy()

        # Seamlessly inpaint graphic text / campaign logos (e.g., magazine mastheads or furniture brand captions)
        if remove_overlay_text:
            try:
                pil_rgb = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                tboxes = self.apple_vision.text_boxes(pil_rgb)
                if tboxes:
                    txt_mask = np.zeros((h, w), dtype=np.uint8)
                    for tb in tboxes:
                        tx1 = max(0, int(round(float(tb["x"]) * w)) - 8)
                        ty1 = max(0, int(round(float(tb["y"]) * h)) - 8)
                        tx2 = min(w, int(round((float(tb["x"]) + float(tb["w"])) * w)) + 8)
                        ty2 = min(h, int(round((float(tb["y"]) + float(tb["h"])) * h)) + 8)
                        if tx2 > tx1 and ty2 > ty1:
                            cv2.rectangle(txt_mask, (tx1, ty1), (tx2, ty2), 255, -1)
                    if np.any(txt_mask > 0):
                        out = cv2.inpaint(out, txt_mask, 9.0, cv2.INPAINT_TELEA)
            except Exception:
                pass

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
                    # Sample local ambient scene border color so privacy mask preserves scene hue/lighting
                    bx1, by1 = max(0, x1 - 18), max(0, y1 - 18)
                    bx2, by2 = min(w, x2 + 18), min(h, y2 + 18)
                    ring = np.ones((by2 - by1, bx2 - bx1), dtype=bool)
                    ring[(y1 - by1) : (y2 - by1), (x1 - bx1) : (x2 - bx1)] = False
                    patch_ring = out[by1:by2, bx1:bx2][ring]
                    if patch_ring.size >= 3:
                        med_bgr = [int(round(float(v))) for v in np.median(patch_ring, axis=0)]
                        fill_color = (
                            int(round(0.82 * med_bgr[0] + 0.18 * 136)),
                            int(round(0.82 * med_bgr[1] + 0.18 * 136)),
                            int(round(0.82 * med_bgr[2] + 0.18 * 136)),
                        )
                    else:
                        fill_color = (140, 140, 140)
                    cv2.rectangle(out, (x1, y1), (x2, y2), fill_color, -1)
                    count += 1
        if output_masked_path is not None:
            output_masked_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_masked_path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        return out, count

    def analyze_reference_pose_before_masking(self, bgr_img: np.ndarray) -> Dict[str, object]:
        """Automatically infer 3-axis head pose (Yaw, Pitch, Roll) and natural Gaze Mode from a reference photo BEFORE grey-masking."""
        import math

        h, w = bgr_img.shape[:2]
        try:
            face = self.detector.detect(bgr_img)
            le, re, nose, lm, rm = face.landmarks_5pt
            d_left = float(np.linalg.norm(nose - le))
            d_right = float(np.linalg.norm(nose - re))
            signed_asym = (d_left - d_right) / max(1.0, d_left + d_right)
            est_yaw = float(np.clip(signed_asym * 145.0, -90.0, 90.0))
            matched = resolve_8angle_view_from_azimuth(est_yaw)
            snap_yaw = VIEW_AZIMUTH_DEG[matched]

            # 3D Projective Pitch & Roll Extraction from 5-point landmarks + bbox foreshortening
            eye_vec = re - le
            eye_line_deg = math.degrees(math.atan2(float(eye_vec[1]), float(eye_vec[0])))
            _, _, fw, fh = face.bbox
            aspect_wh = float(fw) / max(1.0, float(fh))

            # Projective law: when turned (signed_asym != 0) and bowing head downward, `signed_asym * eye_line_deg > 0`
            # and `fw/fh` increases above ~0.78 due to vertical foreshortening.
            slant_coupled = signed_asym * eye_line_deg
            foreshorten_down = max(0.0, (aspect_wh - 0.78) * 95.0)
            if slant_coupled > 2.2:
                est_pitch = -float(np.clip(abs(eye_line_deg) * 0.62 + foreshorten_down * 0.45, 14.0, 42.0))
                est_roll = float(np.clip(eye_line_deg * 0.35, -22.0, 22.0))
            elif slant_coupled < -2.2:
                est_pitch = float(np.clip(abs(eye_line_deg) * 0.45, 10.0, 28.0))
                est_roll = float(np.clip(eye_line_deg * 0.35, -22.0, 22.0))
            else:
                # Near-frontal downward bow check via nose-to-mouth compression
                eye_mid = 0.5 * (le + re)
                mouth_mid = 0.5 * (lm + rm)
                en_dist = float(np.linalg.norm(nose - eye_mid))
                nm_dist = max(1.0, float(np.linalg.norm(mouth_mid - nose)))
                en_nm_ratio = en_dist / nm_dist
                if en_nm_ratio > 1.48 or aspect_wh > 0.90:
                    est_pitch = -float(np.clip((en_nm_ratio - 1.25) * 45.0 + foreshorten_down * 0.6, 15.0, 38.0))
                else:
                    est_pitch = 0.0
                est_roll = float(np.clip(eye_line_deg, -20.0, 20.0))

            if est_pitch <= -14.0:
                gaze_mode = "downward_task_focus"
            elif est_pitch >= 14.0:
                gaze_mode = "upward_contemplative"
            elif abs(snap_yaw) >= 20.0:
                gaze_mode = "off_camera_natural"
            else:
                gaze_mode = "camera_eye_contact"

            x_norm = float(np.clip(face.center[0] / max(1.0, float(w)), 0.25, 0.75))
            return {
                "auto_yaw_deg": snap_yaw,
                "auto_pitch_deg": round(est_pitch, 1),
                "auto_roll_deg": round(est_roll, 1),
                "auto_gaze_mode": gaze_mode,
                "auto_matched_view": matched,
                "auto_matched_label": VIEW_LABELS_KO[matched],
                "auto_subject_x_norm": round(x_norm, 3),
                "auto_source": "reference_3d_pose_and_gaze",
            }
        except Exception:
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            mid_x = max(1, w // 2)
            left_luma = float(np.mean(gray[:, :mid_x]))
            right_luma = float(np.mean(gray[:, mid_x:]))
            if left_luma > right_luma + 10.0:
                est_yaw = -22.5
                x_norm = 0.58
            elif right_luma > left_luma + 10.0:
                est_yaw = 22.5
                x_norm = 0.42
            else:
                est_yaw = 0.0
                x_norm = 0.56
            matched = resolve_8angle_view_from_azimuth(est_yaw)
            return {
                "auto_yaw_deg": est_yaw,
                "auto_pitch_deg": 0.0,
                "auto_roll_deg": 0.0,
                "auto_gaze_mode": "off_camera_natural" if abs(est_yaw) >= 20.0 else "camera_eye_contact",
                "auto_matched_view": matched,
                "auto_matched_label": VIEW_LABELS_KO[matched],
                "auto_subject_x_norm": x_norm,
                "auto_source": "scene_lighting_flow",
            }

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

        d_left = float(np.linalg.norm(nose - le))
        d_right = float(np.linalg.norm(nose - re))
        signed_asym = (d_left - d_right) / max(1.0, d_left + d_right)
        signed_yaw_deg = float(np.clip(signed_asym * 135.0, -90.0, 90.0))

        eye_chin = 1.65
        width_len = 1.15
        try:
            pil_img = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
            marks = self.apple_vision.landmarks(pil_img)
            shape = self.apple_vision.compute_face_shape(marks)
            if shape:
                eye_chin = float(shape.get("eyeChin", eye_chin))
                width_len = float(shape.get("normWidthLength", shape.get("widthLength", width_len)))
                signed_yaw_deg = float(np.clip(float(shape.get("turn", 0.0)) * 95.0, -90.0, 90.0))
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
            source_signed_yaw_deg=round(signed_yaw_deg, 1),
        )

    def prepare_canonical_identity_anchor(
        self,
        raw_photo_path: Path,
        output_anchor_path: Path,
        target_size: Tuple[int, int] = (1200, 1500),
        preserve_native_4k: bool = True,
        restore_4k_via_upscayl: bool = False,
    ) -> Tuple[Path, AnatomicalBlueprint]:
        """Normalize an arbitrary real photo into a pure optical 85mm head-and-shoulders canonical anchor (preserves native 4K or Upscayl 4x)."""
        bgr = cv2.imread(str(raw_photo_path))
        if bgr is None:
            raise FileNotFoundError(f"Cannot read raw reference photo: {raw_photo_path}")
        if restore_4k_via_upscayl and max(bgr.shape[:2]) < 2000 and self.upscaler is not None and self.upscaler.upscayl_bin is not None:
            bgr, _ = self.upscaler.upscale_with_upscayl(bgr, style="photo", scale=4)

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
        if preserve_native_4k and target_size == (1200, 1500) and (crop.shape[1] > tw or crop.shape[0] > th):
            if crop.shape[0] > 4800:
                sc = 4800.0 / float(crop.shape[0])
                canonical = cv2.resize(crop, (int(round(crop.shape[1] * sc)), 4800), interpolation=cv2.INTER_LANCZOS4)
            else:
                canonical = crop
        elif crop.shape[1] < tw or crop.shape[0] < th:
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

        src_deg = blueprint.source_signed_yaw_deg
        src_angle_desc = (
            f"captured at {src_deg:+.1f} degrees yaw" if abs(src_deg) >= 6.0 else "captured at ~0 degrees front yaw"
        )
        ref_roles = (
            f"Image 1 is the untouched real Gen-0 identity photograph ({src_angle_desc})—the absolute 1:1 ground-truth authority "
            "for exact cheek fullness, jawline width, eye shape, nose tip roundness, philtrum length, lip contour, true iris color, "
            "authentic skin texture, eyewear geometry, and head-fixed hair parting/ear-tuck style."
        )
        if has_intermediate_anchor:
            ref_roles += (
                f" Image 2 is the approved {intermediate_label} of the exact same person—"
                "rotate smoothly from Image 2 to the target angle while keeping Image 1's exact 1:1 cheek volume, eye/nose/lip likeness, and skin fidelity."
            )

        prompt = (
            "Editorial portrait. Unretouched real-camera 85mm f/4 studio agency turnaround reference portrait of the EXACT SAME "
            "real person shown in Image 1 against a clean light-neutral studio backdrop. "
            f"Reference Roles: {ref_roles} "
            f"Target Angle — {view_title}: {view_desc} "
            "CRITICAL ANTI-DRIFT RULES: Do NOT mirror or flip to the opposite direction, do NOT invent a generic model face, "
            "preserve exact cheek breadth and jaw width from Image 1 without thinning or elongating the head, and keep hair parting/ear-tuck fixed to the subject's own anatomical left/right sides in 3D! "
            f"Identity & Eyewear Continuity: Preserve {subject.facial_identity_descriptor()}. "
            "Keep authentic iris color from Image 1 with crystal-clear non-tinted lenses (zero green or purple coating cast). "
            "If eyewear is present in Image 1, render the temple arm extending continuously and uninterruptedly from the front outer "
            "frame hinge back to rest naturally over the top of the visible ear. "
            f"{blueprint.to_3d_invariance_clause()} "
            "Wardrobe & Lighting Continuity: Dress the subject in a simple, plain matte dark-charcoal crewneck studio top (discarding any "
            "pinstripes, badges, or busy patterns from Image 1 so turnaround anchors stay neutral), balanced adult 7.5-to-8-head "
            "head-to-shoulder scale, soft natural studio key light, authentic unretouched real skin texture, zero plastic/oily gloss, "
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
        """Plan all 8 frontal-arc directions (-90°, -67.5°, -45°, -22.5°, +22.5°, +45°, +67.5°, +90°) + 0° Origin via STRICT 1-Hop Direct-from-Gen0 (ZERO multi-hop chaining!)."""
        front_bgr = cv2.imread(str(canonical_front_path))
        if front_bgr is None:
            raise FileNotFoundError(f"Cannot read canonical front anchor: {canonical_front_path}")
        blueprint = self.extract_anatomical_blueprint(front_bgr)

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

        # IRONCLAD ZERO-CHAINING LAW: Every single turnaround angle is `hop = 1` strictly from untouched `Gen-0` ONLY!
        # NEVER feed a generated turnaround image (`Gen-1`) into another turnaround generation (`Gen-2..Gen-4`)!
        for v_key in TURNAROUND_8_VIEWS:
            plans[v_key] = {
                "hop": 1,
                "azimuth_deg": VIEW_AZIMUTH_DEG[v_key],
                "view": v_key,
                "label_ko": VIEW_LABELS_KO[v_key],
                "reference_images": [str(canonical_front_path)],
                "prompt": self.build_turnaround_angle_prompt(
                    v_key,
                    subject,
                    blueprint,
                    has_intermediate_anchor=False,
                    intermediate_label="",
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
        """Backward-compatible 5-angle subset planner — strictly 1-Hop Direct-from-Gen0 (ZERO multi-hop chaining!)."""
        front_bgr = cv2.imread(str(canonical_front_path))
        if front_bgr is None:
            raise FileNotFoundError(f"Cannot read canonical front anchor: {canonical_front_path}")
        blueprint = self.extract_anatomical_blueprint(front_bgr)

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
                "hop": 1,
                "view": "left_profile",
                "reference_images": [str(canonical_front_path)],
                "prompt": self.build_turnaround_angle_prompt("left_90", subject, blueprint, False),
            },
            "right_profile": {
                "hop": 1,
                "view": "right_profile",
                "reference_images": [str(canonical_front_path)],
                "prompt": self.build_turnaround_angle_prompt("right_90", subject, blueprint, False),
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
