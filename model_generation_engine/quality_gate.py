from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import cv2
import numpy as np

from .config import EngineConfig
from .detector import DynamicFaceDetector
from .skin_processor import MatteSkinProcessor


@dataclass(frozen=True)
class QualityReport:
    """Automated 7-Gate Anti-Smudge & Quality Report for a 5K model generation output."""

    passed: bool
    width: int
    height: int
    face_confidence: float
    face_laplacian_variance: float
    seam_border_l1_diff: float
    eye_tilt_degrees: float
    inter_ocular_px: float
    specular_glare_ratio: float = 0.0
    skin_micro_texture_std: float = 2.0
    min_organ_gradient: float = 10.0
    anatomical_geometry_valid: bool = True
    message: str = "ALL QUALITY GATES PASSED"


class QualityGateAuditor:
    """Enforces 7 mandatory automated quality gates (including Per-Organ Anti-Smudge & Anatomy Gate) before accepting any portrait output."""

    def __init__(self, config: EngineConfig, detector: DynamicFaceDetector) -> None:
        self.config = config
        self.detector = detector
        self.skin_processor = MatteSkinProcessor(detector)

    @staticmethod
    def measure_per_organ_sharpness(
        patch_bgr: np.ndarray, face
    ) -> Tuple[float, dict]:
        """Measure scale-adaptive high-frequency anatomical edge energy independently on left eye, right eye, nose tip, and mouth.

        Uses second-order / high-pass detail energy rather than first-order slope so smooth blurry ramps ('얼굴/눈코입 뭉개짐')
        are accurately detected even when lighting gradients or glasses frames are present.
        """
        h, w = patch_bgr.shape[:2]
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        eye_d = max(8.0, face.inter_ocular_distance)
        scale_factor = float(np.clip(eye_d / 110.0, 0.5, 3.0))
        high_freq = gray - cv2.GaussianBlur(gray, (0, 0), max(1.2, 1.8 * scale_factor))

        le, re, nose, lm, rm = face.landmarks_5pt
        mouth_mid = 0.5 * (lm + rm)
        r = max(5, int(round(0.18 * eye_d)))

        organ_pts = {
            "left_eye": le,
            "right_eye": re,
            "nose": nose,
            "mouth": mouth_mid,
        }
        scores = {}
        for name, pt in organ_pts.items():
            cx = int(round(np.clip(pt[0], 0, w - 1)))
            cy = int(round(np.clip(pt[1], 0, h - 1)))
            x1, x2 = max(0, cx - r), min(w, cx + r + 1)
            y1, y2 = max(0, cy - r), min(h, cy + r + 1)
            roi = high_freq[y1:y2, x1:x2]
            scores[name] = float(np.std(roi)) if roi.size > 0 else 0.0

        min_grad = min(scores.values()) if scores else 0.0
        return min_grad, scores

    @staticmethod
    def validate_anatomical_geometry(face) -> Tuple[bool, str]:
        """Verify that the 5 canonical landmarks form an anatomically valid, un-melted human facial geometry."""
        le, re, nose, lm, rm = face.landmarks_5pt
        eye_d = float(np.linalg.norm(re - le))
        if eye_d < 6.0:
            return False, f"Collapsed inter-ocular distance ({eye_d:.1f}px)"

        eye_mid = 0.5 * (le + re)
        mouth_mid = 0.5 * (lm + rm)
        ux = (re - le) / eye_d
        # Downward perpendicular unit vector in image coordinates
        uy = np.array([-ux[1], ux[0]], dtype=np.float32)
        if np.dot(mouth_mid - eye_mid, uy) < 0:
            uy = -uy

        nose_y = float(np.dot(nose - eye_mid, uy))
        mouth_y = float(np.dot(mouth_mid - eye_mid, uy))
        mouth_w = float(np.linalg.norm(rm - lm))

        if nose_y < 0.12 * eye_d:
            return False, f"Anatomical landmark collapse: nose tip too close to or above eye line (nose_y={nose_y:.1f}px)"
        if mouth_y < nose_y + 0.10 * eye_d:
            return False, f"Anatomical landmark collapse: mouth too close to or above nose tip (mouth_y={mouth_y:.1f}px)"
        if not (0.35 * eye_d <= mouth_w <= 1.75 * eye_d):
            return False, f"Anatomical landmark distortion: abnormal mouth-to-eye width ratio ({mouth_w / eye_d:.2f})"
        return True, "OK"

    def audit(
        self,
        canvas_5k_bgr: np.ndarray,
        crop_box_5k: Tuple[int, int, int, int],
        seam_l1_diff: float,
    ) -> QualityReport:
        """Verify resolution, sharpness, seam continuity, landmark symmetry, zero plastic gloss, and natural skin texture."""
        if canvas_5k_bgr is None or canvas_5k_bgr.ndim != 3:
            return QualityReport(
                passed=False,
                width=0,
                height=0,
                face_confidence=0.0,
                face_laplacian_variance=0.0,
                seam_border_l1_diff=float(seam_l1_diff),
                eye_tilt_degrees=0.0,
                inter_ocular_px=0.0,
                specular_glare_ratio=0.0,
                skin_micro_texture_std=0.0,
                message="Invalid or empty canvas image.",
            )

        h, w = canvas_5k_bgr.shape[:2]
        exp_w = self.config.canvas.target_width
        exp_h = self.config.canvas.target_height
        if (w, h) != (exp_w, exp_h):
            return QualityReport(
                passed=False,
                width=w,
                height=h,
                face_confidence=0.0,
                face_laplacian_variance=0.0,
                seam_border_l1_diff=float(seam_l1_diff),
                eye_tilt_degrees=0.0,
                inter_ocular_px=0.0,
                specular_glare_ratio=0.0,
                skin_micro_texture_std=0.0,
                message=f"Resolution mismatch: got ({w}x{h}), expected ({exp_w}x{exp_h}).",
            )

        x1, y1, x2, y2 = crop_box_5k
        x1_c, y1_c = max(0, min(w - 1, int(x1))), max(0, min(h - 1, int(y1)))
        x2_c, y2_c = max(x1_c + 1, min(w, int(x2))), max(y1_c + 1, min(h, int(y2)))
        patch = canvas_5k_bgr[y1_c:y2_c, x1_c:x2_c]

        try:
            face = self.detector.detect(patch)
        except Exception as exc:
            return QualityReport(
                passed=False,
                width=w,
                height=h,
                face_confidence=0.0,
                face_laplacian_variance=0.0,
                seam_border_l1_diff=float(seam_l1_diff),
                eye_tilt_degrees=0.0,
                inter_ocular_px=0.0,
                specular_glare_ratio=0.0,
                skin_micro_texture_std=0.0,
                message=f"Face detection failed in quality audit: {exc}",
            )

        fx, fy, fw, fh = [int(round(v)) for v in face.bbox]
        fx1, fy1 = max(0, fx), max(0, fy)
        fx2, fy2 = min(patch.shape[1], max(fx1 + 1, fx + fw)), min(patch.shape[0], max(fy1 + 1, fy + fh))
        face_roi = patch[fy1:fy2, fx1:fx2]
        if face_roi.size == 0:
            return QualityReport(
                passed=False,
                width=w,
                height=h,
                face_confidence=face.confidence,
                face_laplacian_variance=0.0,
                seam_border_l1_diff=float(seam_l1_diff),
                eye_tilt_degrees=face.eye_tilt_degrees,
                inter_ocular_px=face.inter_ocular_distance,
                specular_glare_ratio=0.0,
                skin_micro_texture_std=0.0,
                message="Empty face ROI slice.",
            )

        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
        skin_metrics = self.skin_processor.measure_metrics(patch, face)
        min_organ_grad, organ_scores = self.measure_per_organ_sharpness(patch, face)
        anat_ok, anat_msg = self.validate_anatomical_geometry(face)

        failures = []
        if not np.isfinite(seam_l1_diff) or seam_l1_diff > self.config.max_seam_l1_diff:
            failures.append(
                f"Seam border L1 diff ({seam_l1_diff:.2f}) exceeds max ({self.config.max_seam_l1_diff:.2f})"
            )
        if not np.isfinite(lap_var) or lap_var < self.config.min_face_laplacian_var:
            failures.append(
                f"Face Laplacian sharpness ({lap_var:.2f}) below min ({self.config.min_face_laplacian_var:.2f}) — face is too blurry"
            )
        if min_organ_grad < self.config.min_organ_gradient_energy:
            worst_organ = min(organ_scores, key=organ_scores.get)
            failures.append(
                f"Facial organ smudge detected on '{worst_organ}' (gradient={min_organ_grad:.2f} < min {self.config.min_organ_gradient_energy:.2f})"
            )
        if not anat_ok:
            failures.append(anat_msg)
        if abs(face.eye_tilt_degrees) > 20.0:
            failures.append(f"Abnormal eye tilt ({face.eye_tilt_degrees:.1f} deg) indicates distortion")
        if face.inter_ocular_distance < 25.0:
            failures.append(f"Inter-ocular distance ({face.inter_ocular_distance:.1f}px) too small for 5K")
        if skin_metrics.specular_glare_ratio > self.config.max_specular_glare_ratio:
            failures.append(
                f"Plastic specular glare ratio ({skin_metrics.specular_glare_ratio:.3f}) exceeds max ({self.config.max_specular_glare_ratio:.3f})"
            )
        if skin_metrics.skin_micro_texture_std < self.config.min_skin_micro_texture_std:
            failures.append(
                f"Skin micro-texture std ({skin_metrics.skin_micro_texture_std:.2f}) below min ({self.config.min_skin_micro_texture_std:.2f}) — skin is over-smoothed"
            )

        passed = len(failures) == 0
        msg = "ALL QUALITY GATES PASSED" if passed else "; ".join(failures)
        return QualityReport(
            passed=passed,
            width=w,
            height=h,
            face_confidence=face.confidence,
            face_laplacian_variance=lap_var,
            seam_border_l1_diff=float(seam_l1_diff),
            eye_tilt_degrees=face.eye_tilt_degrees,
            inter_ocular_px=face.inter_ocular_distance,
            specular_glare_ratio=skin_metrics.specular_glare_ratio,
            skin_micro_texture_std=skin_metrics.skin_micro_texture_std,
            min_organ_gradient=min_organ_grad,
            anatomical_geometry_valid=anat_ok,
            message=msg,
        )
