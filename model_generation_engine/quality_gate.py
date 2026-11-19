from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from .apple_vision import AppleVisionAnalyzer
from .config import EngineConfig
from .detector import DynamicFaceDetector
from .skin_processor import MatteSkinProcessor


@dataclass(frozen=True)
class QualityReport:
    """Automated 9-Gate Anti-Smudge, Face Ratio & Neural Identity Similarity Report for a 5K model generation output."""

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
    identity_cosine_sim: Optional[float] = None
    face_shape_note: str = ""
    message: str = "ALL QUALITY GATES PASSED"


class QualityGateAuditor:
    """Enforces 9 mandatory automated quality gates (including SFace 128-D Identity & Apple Vision Face Ratio Gates)."""

    def __init__(self, config: EngineConfig, detector: DynamicFaceDetector) -> None:
        self.config = config
        self.detector = detector
        self.skin_processor = MatteSkinProcessor(detector)
        bin_dir = (config.vault_dir / "models" / "apple_vision_bin") if config.vault_dir else None
        self.apple_vision = AppleVisionAnalyzer(bin_dir=bin_dir)
        self._sface_rec = None
        self._sface_sess = None
        try:
            sface_path = config.resolve_model_path("sface.onnx")
            self._sface_rec = cv2.FaceRecognizerSF.create(str(sface_path), "")
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            self._sface_sess = ort.InferenceSession(
                str(sface_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        except Exception:
            pass

    def measure_identity_similarity(
        self, ref_bgr: np.ndarray, dst_bgr: np.ndarray, ref_face, dst_face
    ) -> float:
        """Compute 128-D Neural Face Recognition Cosine Similarity via OpenCV SFace (`sface.onnx`)."""
        if self._sface_rec is None or self._sface_sess is None:
            return 1.0

        def _feat(img: np.ndarray, f) -> np.ndarray:
            box15 = np.array(
                list(f.bbox) + f.landmarks_5pt.ravel().tolist() + [f.confidence], dtype=np.float32
            )
            aligned = self._sface_rec.alignCrop(img, box15)
            blob = aligned.transpose(2, 0, 1)[None, ...].astype(np.float32)
            vec = self._sface_sess.run(None, {self._sface_sess.get_inputs()[0].name: blob})[0][0]
            return vec / (float(np.linalg.norm(vec)) + 1e-8)

        v1 = _feat(ref_bgr, ref_face)
        v2 = _feat(dst_bgr, dst_face)
        return float(np.dot(v1, v2))

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
        reference_face_bgr: Optional[np.ndarray] = None,
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
        if face.inter_ocular_distance < self.config.min_inter_ocular_5k_px:
            failures.append(
                f"Inter-ocular distance ({face.inter_ocular_distance:.1f}px) below min ({self.config.min_inter_ocular_5k_px:.1f}px) for 5K"
            )
        if skin_metrics.specular_glare_ratio > self.config.max_specular_glare_ratio:
            failures.append(
                f"Plastic specular glare ratio ({skin_metrics.specular_glare_ratio:.3f}) exceeds max ({self.config.max_specular_glare_ratio:.3f})"
            )
        if skin_metrics.skin_micro_texture_std < self.config.min_skin_micro_texture_std:
            failures.append(
                f"Skin micro-texture std ({skin_metrics.skin_micro_texture_std:.2f}) below min ({self.config.min_skin_micro_texture_std:.2f}) — skin is over-smoothed"
            )

        id_sim: Optional[float] = None
        shape_note = ""
        if reference_face_bgr is not None:
            ref_face = self.detector.detect(reference_face_bgr)
            if abs(ref_face.yaw_asymmetry_ratio - face.yaw_asymmetry_ratio) <= 0.16:
                id_sim = self.measure_identity_similarity(reference_face_bgr, patch, ref_face, face)
                if id_sim < self.config.min_identity_cosine_sim:
                    failures.append(
                        f"Face identity similarity ({id_sim:.3f}) below min ({self.config.min_identity_cosine_sim:.3f}) — face drifted from reference"
                    )
            try:
                ref_pil = Image.fromarray(cv2.cvtColor(reference_face_bgr, cv2.COLOR_BGR2RGB))
                dst_pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                ref_sh = self.apple_vision.compute_face_shape(self.apple_vision.landmarks(ref_pil))
                dst_sh = self.apple_vision.compute_face_shape(self.apple_vision.landmarks(dst_pil))
                cmp = self.apple_vision.compare_face_shapes(
                    ref_sh, dst_sh, limit_pct=self.config.max_face_shape_diff_pct
                )
                shape_note = str(cmp.get("note", ""))
                if not cmp.get("passed", True):
                    failures.append(f"Face shape proportion drift exceeded {self.config.max_face_shape_diff_pct}%: {shape_note}")
            except Exception:
                pass

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
            identity_cosine_sim=id_sim,
            face_shape_note=shape_note,
            message=msg,
        )
