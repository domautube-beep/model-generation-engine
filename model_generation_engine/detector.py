from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np


@dataclass(frozen=True)
class AdaptiveSkinToneProfile:
    """Dynamically sampled CIE LAB skin tone distribution for an arbitrary model and lighting condition."""

    l_median: float
    l_iqr: float
    a_median: float
    a_iqr: float
    b_median: float
    b_iqr: float

    def build_chrominance_gate(self, lab_img: np.ndarray) -> np.ndarray:
        """Compute an adaptive skin gate on a LAB image without hardcoded skin-color thresholds."""
        L, A, B = lab_img[:, :, 0], lab_img[:, :, 1], lab_img[:, :, 2]
        l_tol = max(45.0, 2.8 * self.l_iqr)
        a_tol = max(14.0, 3.0 * self.a_iqr)
        b_tol = max(16.0, 3.0 * self.b_iqr)

        l_low = max(12.0, self.l_median - l_tol)
        l_high = min(252.0, self.l_median + l_tol + 15.0)
        gate = (
            (L >= l_low)
            & (L <= l_high)
            & (np.abs(A - self.a_median) <= a_tol)
            & (np.abs(B - self.b_median) <= b_tol)
        )
        return gate.astype(np.float32)


@dataclass(frozen=True)
class FaceDetectionResult:
    """Sub-pixel face bounding box, 5 canonical landmarks, and 3D pose metrics detected dynamically."""

    bbox: Tuple[float, float, float, float]  # (x, y, w, h)
    landmarks_5pt: np.ndarray  # shape (5, 2): [left_eye, right_eye, nose_tip, left_mouth, right_mouth]
    confidence: float

    @property
    def center(self) -> Tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + 0.5 * w, y + 0.5 * h)

    @property
    def inter_ocular_distance(self) -> float:
        le, re = self.landmarks_5pt[0], self.landmarks_5pt[1]
        return float(np.linalg.norm(re - le))

    @property
    def eye_tilt_degrees(self) -> float:
        le, re = self.landmarks_5pt[0], self.landmarks_5pt[1]
        return float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0])))

    @property
    def yaw_asymmetry_ratio(self) -> float:
        """Estimate 3D head yaw from nose horizontal offset relative to eye midpoint (-0.5..+0.5)."""
        le, re, nose = self.landmarks_5pt[0], self.landmarks_5pt[1], self.landmarks_5pt[2]
        eye_mid = 0.5 * (le + re)
        eye_vec = re - le
        dist = float(np.linalg.norm(eye_vec)) + 1e-6
        ux = eye_vec / dist
        return float(np.clip(np.dot(nose - eye_mid, ux) / dist, -0.6, 0.6))


class DynamicFaceDetector:
    """Zero-hardcoding neural face, 3D pose, and adaptive skin tone detector powered by OpenCV YuNet."""

    def __init__(self, yunet_onnx_path: Path, score_threshold: float = 0.45, nms_threshold: float = 0.3) -> None:
        if not Path(yunet_onnx_path).exists():
            raise FileNotFoundError(f"YuNet model not found at {yunet_onnx_path}")
        self.yunet_path = str(yunet_onnx_path)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

    def detect(self, bgr_img: np.ndarray) -> FaceDetectionResult:
        """Detect the primary subject face and 5 canonical landmarks on an arbitrary image."""
        if bgr_img is None or bgr_img.ndim != 3:
            raise ValueError("Input must be a valid 3-channel BGR image.")
        h, w = bgr_img.shape[:2]
        faces = None
        used_scale = 1.0
        # YuNet's anchor strides [8, 16, 32] are calibrated for ~640..1280px images; running raw 5K (5040px)
        # first causes large close-up/angled faces (>450px) to exceed anchor stride and false-trigger on background slats.
        scales_to_try = (1280, 800, 0) if max(w, h) > 1280 else (0, 640)
        for max_dim in scales_to_try:
            if max_dim == 0:
                work_img = bgr_img
                sw, sh = w, h
                scale = 1.0
            else:
                if max(w, h) <= max_dim:
                    continue
                scale = float(max_dim) / float(max(w, h))
                sw, sh = max(16, int(round(w * scale))), max(16, int(round(h * scale)))
                work_img = cv2.resize(bgr_img, (sw, sh), interpolation=cv2.INTER_AREA)

            detector = cv2.FaceDetectorYN.create(
                self.yunet_path, "", (sw, sh), self.score_threshold, self.nms_threshold, 5000
            )
            _, cand = detector.detect(work_img)
            if cand is not None and len(cand) > 0:
                faces = cand
                used_scale = scale
                break

        if faces is None or len(faces) == 0:
            raise RuntimeError(f"No human face detected in image of shape ({w}x{h}).")

        inv_s = 1.0 / used_scale
        best = max(faces, key=lambda f: float(f[2] * f[3]) * float(f[14]))
        bbox = (
            float(best[0]) * inv_s,
            float(best[1]) * inv_s,
            float(best[2]) * inv_s,
            float(best[3]) * inv_s,
        )
        pts5 = (
            np.array(
                [
                    [best[4], best[5]],    # viewer's left eye
                    [best[6], best[7]],    # viewer's right eye
                    [best[8], best[9]],    # nose tip
                    [best[10], best[11]],  # viewer's left mouth corner
                    [best[12], best[13]],  # viewer's right mouth corner
                ],
                dtype=np.float32,
            )
            * inv_s
        )
        return FaceDetectionResult(bbox=bbox, landmarks_5pt=pts5, confidence=float(best[14]))

    @staticmethod
    def sample_adaptive_skin_profile(
        bgr_img: np.ndarray, face: FaceDetectionResult
    ) -> AdaptiveSkinToneProfile:
        """Dynamically sample the subject's true skin tone in CIE LAB from the inner cheek/nose-bridge zones.

        Works universally across all human skin tones (fair, olive, tan, deep dark) and lighting colors.
        """
        h, w = bgr_img.shape[:2]
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        le, re, nose, lm, rm = face.landmarks_5pt
        eye_d = max(8.0, face.inter_ocular_distance)

        sample_mask = np.zeros((h, w), dtype=np.uint8)
        # Sample left cheek, right cheek, and nose dorsum midpoints derived from the 5 landmarks
        cheek_l = 0.45 * le + 0.35 * lm + 0.20 * nose
        cheek_r = 0.45 * re + 0.35 * rm + 0.20 * nose
        nose_mid = 0.35 * (0.5 * (le + re)) + 0.65 * nose
        r_px = max(2, int(round(0.12 * eye_d)))

        for pt in (cheek_l, cheek_r, nose_mid):
            px = int(round(np.clip(pt[0], 0, w - 1)))
            py = int(round(np.clip(pt[1], 0, h - 1)))
            cv2.circle(sample_mask, (px, py), r_px, 1, -1)

        pixels = lab[sample_mask == 1]
        if len(pixels) < 10:
            fx, fy, fw, fh = [int(round(v)) for v in face.bbox]
            roi = lab[max(0, fy) : min(h, fy + fh), max(0, fx) : min(w, fx + fw)]
            pixels = roi.reshape(-1, 3)

        l_vals, a_vals, b_vals = pixels[:, 0], pixels[:, 1], pixels[:, 2]
        return AdaptiveSkinToneProfile(
            l_median=float(np.median(l_vals)),
            l_iqr=float(np.percentile(l_vals, 75) - np.percentile(l_vals, 25) + 4.0),
            a_median=float(np.median(a_vals)),
            a_iqr=float(np.percentile(a_vals, 75) - np.percentile(a_vals, 25) + 2.5),
            b_median=float(np.median(b_vals)),
            b_iqr=float(np.percentile(b_vals, 75) - np.percentile(b_vals, 25) + 2.5),
        )

    @staticmethod
    def compute_upper_body_crop_box(
        img_shape: Tuple[int, int],
        face: FaceDetectionResult,
        aspect_ratio: float = 16.0 / 9.0,
        vertical_head_scale: float = 4.5,
    ) -> Tuple[int, int, int, int]:
        """Dynamically compute an aspect-locked upper-body crop box (x1, y1, x2, y2)."""
        img_h, img_w = img_shape[:2]
        if img_h < 2 or img_w < 2:
            raise ValueError(f"Image dimensions too small for cropping: ({img_w}x{img_h})")
        fx, fy, fw, fh = face.bbox
        cx = fx + 0.5 * fw

        crop_h = int(round(np.clip(fh * vertical_head_scale, min(img_h, max(2, img_h * 0.25)), img_h * 0.65)))
        crop_h = max(2, min(img_h, crop_h))
        crop_w = int(round(crop_h * aspect_ratio))
        if crop_w > img_w:
            crop_w = img_w
            crop_h = max(2, min(img_h, int(round(crop_w / aspect_ratio))))
        if crop_h > img_h:
            crop_h = img_h
            crop_w = max(2, min(img_w, int(round(crop_h * aspect_ratio))))

        y1 = int(round(fy - 0.32 * crop_h))
        x1 = int(round(cx - 0.50 * crop_w))

        x1 = max(0, min(img_w - crop_w, x1))
        y1 = max(0, min(img_h - crop_h, y1))
        return (x1, y1, x1 + crop_w, y1 + crop_h)

    @staticmethod
    def compute_head_halo_box(
        crop_shape: Tuple[int, int],
        face_in_crop: FaceDetectionResult,
    ) -> Tuple[int, int, int, int]:
        """Dynamically compute the localized (hx1, hy1, hx2, hy2) head+hair+neck box inside a crop."""
        ch, cw = crop_shape[:2]
        if ch < 4 or cw < 4:
            raise ValueError(f"Crop shape too small for head halo box: ({cw}x{ch})")
        fx, fy, fw, fh = face_in_crop.bbox
        cx = fx + 0.5 * fw

        margin_x = min(16, cw // 8)
        margin_top = min(12, ch // 8)
        margin_bot = min(20, ch // 6)

        hx1 = int(round(np.clip(cx - 0.95 * fw, margin_x, cw - margin_x - 2)))
        hx2 = int(round(np.clip(cx + 0.95 * fw, hx1 + 2, cw - margin_x)))
        hy1 = int(round(np.clip(fy - 0.58 * fh, margin_top, ch - margin_bot - 2)))
        hy2 = int(round(np.clip(fy + fh + 0.35 * fh, hy1 + 2, ch - margin_bot)))
        return (hx1, hy1, hx2, hy2)

    def extract_pure_head_identity_reference(
        self,
        bgr_img: np.ndarray,
        target_size: Tuple[int, int] = (768, 820),
    ) -> np.ndarray:
        """Crop a tight head-and-collar ID reference portrait (face filling ~65% of frame) from any half/full-body photo.

        Ensures the image generator devotes maximum visual tokens strictly to the subject's true face shape,
        eyes, nose, and mouth instead of the original photo's body/chair/background.
        """
        h, w = bgr_img.shape[:2]
        face = self.detect(bgr_img)
        fx, fy, fw, fh = face.bbox
        cx = fx + 0.5 * fw

        # If the input image is already a tight head crop (face height >= 36% of image height), keep its framing
        if fh / float(max(1, h)) >= 0.36:
            return bgr_img.copy()

        # Compute a tight ID-portrait box around the head, hair, ears, and upper collar
        box_h = int(round(fh * 2.35))
        box_w = int(round(box_h * (target_size[0] / float(target_size[1]))))
        x1 = int(round(cx - 0.50 * box_w))
        y1 = int(round(fy - 0.58 * fh))
        x1 = max(0, min(w - box_w, x1))
        y1 = max(0, min(h - box_h, y1))
        x2 = min(w, x1 + box_w)
        y2 = min(h, y1 + box_h)

        crop = bgr_img[y1:y2, x1:x2]
        if crop.shape[1] < target_size[0]:
            crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_LANCZOS4)
        return crop
