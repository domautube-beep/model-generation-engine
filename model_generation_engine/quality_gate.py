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
    """Automated 6-Gate Quality Report for a 5K model generation output."""

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
    message: str = "ALL QUALITY GATES PASSED"


class QualityGateAuditor:
    """Enforces 6 mandatory automated quality gates before accepting any portrait output."""

    def __init__(self, config: EngineConfig, detector: DynamicFaceDetector) -> None:
        self.config = config
        self.detector = detector
        self.skin_processor = MatteSkinProcessor(detector)

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

        failures = []
        if not np.isfinite(seam_l1_diff) or seam_l1_diff > self.config.max_seam_l1_diff:
            failures.append(
                f"Seam border L1 diff ({seam_l1_diff:.2f}) exceeds max ({self.config.max_seam_l1_diff:.2f})"
            )
        if not np.isfinite(lap_var) or lap_var < self.config.min_face_laplacian_var:
            failures.append(
                f"Face Laplacian sharpness ({lap_var:.2f}) below min ({self.config.min_face_laplacian_var:.2f}) — face is too blurry"
            )
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
            message=msg,
        )
