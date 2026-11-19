from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import cv2
import numpy as np

from .detector import DynamicFaceDetector, FaceDetectionResult


@dataclass(frozen=True)
class SkinQualityMetrics:
    """Quantitative metrics verifying zero plastic gloss and natural skin micro-texture."""

    specular_glare_ratio: float
    skin_micro_texture_std: float
    p88_luminance: float
    skin_pixel_count: int


class MatteSkinProcessor:
    """Universal Halo-Free Matte Skin Specular Roll-Off & Scale-Adaptive Pore Micro-Texture Engine.

    Zero hardcoded skin colors or fixed pixel scales:
    - Samples the subject's actual CIE LAB skin distribution dynamically from the detected cheek/nose zones.
    - Adapts the face oval geometry to 3D head yaw (frontal, 3/4, or profile).
    - Scales pore micro-texture wavelengths proportionally to the detected inter-ocular distance.
    """

    def __init__(
        self,
        detector: DynamicFaceDetector,
        percentile_knee: float = 99.2,
        rolloff_range: float = 3.0,
        rolloff_scale: float = 8.0,
        micro_texture_amplitude: float = 0.0,
        organ_crisp_strength: float = 0.0,
    ) -> None:
        self.detector = detector
        self.percentile_knee = percentile_knee
        self.rolloff_range = rolloff_range
        self.rolloff_scale = rolloff_scale
        self.micro_texture_amplitude = micro_texture_amplitude
        self.organ_crisp_strength = organ_crisp_strength

    def build_anatomical_organ_mask(
        self, img_shape: Tuple[int, int], det: FaceDetectionResult
    ) -> np.ndarray:
        """Build a soft, pose-invariant mask covering strictly the eyes/eyelids/eyebrows, nostrils/nose-tip, and lips."""
        h, w = img_shape[:2]
        le, re, nose, lm, rm = det.landmarks_5pt
        eye_d = max(8.0, det.inter_ocular_distance)
        angle = det.eye_tilt_degrees

        organ_mask = np.zeros((h, w), dtype=np.float32)
        for pt_e in (le, re):
            cv2.ellipse(
                organ_mask,
                (int(round(pt_e[0])), int(round(pt_e[1]))),
                (int(round(0.32 * eye_d)), int(round(0.22 * eye_d))),
                angle,
                0,
                360,
                1.0,
                -1,
            )
            cv2.ellipse(
                organ_mask,
                (int(round(pt_e[0])), int(round(pt_e[1] - 0.25 * eye_d))),
                (int(round(0.34 * eye_d)), int(round(0.12 * eye_d))),
                angle,
                0,
                360,
                0.75,
                -1,
            )

        cv2.ellipse(
            organ_mask,
            (int(round(nose[0])), int(round(nose[1] + 0.04 * eye_d))),
            (int(round(0.20 * eye_d)), int(round(0.14 * eye_d))),
            angle,
            0,
            360,
            0.85,
            -1,
        )

        mouth_mid = 0.5 * (lm + rm)
        mouth_w = max(6.0, float(np.linalg.norm(rm - lm)))
        cv2.ellipse(
            organ_mask,
            (int(round(mouth_mid[0])), int(round(mouth_mid[1]))),
            (int(round(0.54 * mouth_w)), int(round(0.24 * mouth_w))),
            angle,
            0,
            360,
            0.90,
            -1,
        )

        ksize = max(5, int(round(0.14 * eye_d)) | 1)
        return cv2.GaussianBlur(organ_mask, (ksize, ksize), max(1.5, 0.04 * eye_d))

    def build_dynamic_skin_mask(
        self, bgr_img: np.ndarray, det: FaceDetectionResult
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build a 3D-yaw-adaptive and skin-tone-adaptive feather mask excluding eyes, eyewear, lips, and hair."""
        h, w = bgr_img.shape[:2]
        fx, fy, fw, fh = det.bbox
        le, re, nose, lm, rm = det.landmarks_5pt
        eye_d = max(8.0, det.inter_ocular_distance)
        angle = det.eye_tilt_degrees
        yaw = det.yaw_asymmetry_ratio

        face_oval = np.zeros((h, w), dtype=np.float32)
        # Shift oval center slightly opposite to nose yaw so the visible cheek on 3/4 views is fully covered
        cx = int(round(fx + (0.50 - 0.12 * yaw) * fw))
        cy = int(round(fy + 0.50 * fh))
        cv2.ellipse(
            face_oval,
            (cx, cy),
            (int(round(0.43 * fw)), int(round(0.47 * fh))),
            angle,
            0,
            360,
            1.0,
            -1,
        )

        eye_excl = np.zeros((h, w), dtype=np.float32)
        for pt_e in (le, re):
            # Exclude eye, eyelids, and eyewear lens/frame
            cv2.ellipse(
                eye_excl,
                (int(round(pt_e[0])), int(round(pt_e[1] + 0.03 * eye_d))),
                (int(round(0.36 * eye_d)), int(round(0.28 * eye_d))),
                angle,
                0,
                360,
                1.0,
                -1,
            )
            # Exclude eyebrow arch so eyebrow hairs retain crisp micro-contrast
            cv2.ellipse(
                eye_excl,
                (int(round(pt_e[0])), int(round(pt_e[1] - 0.26 * eye_d))),
                (int(round(0.38 * eye_d)), int(round(0.14 * eye_d))),
                angle,
                0,
                360,
                1.0,
                -1,
            )

        # Exclude nostril / alar base shadow so 3D nose contour never flattens or smudges
        cv2.ellipse(
            eye_excl,
            (int(round(nose[0])), int(round(nose[1] + 0.07 * eye_d))),
            (int(round(0.22 * eye_d)), int(round(0.13 * eye_d))),
            angle,
            0,
            360,
            1.0,
            -1,
        )

        mouth_mid = 0.5 * (lm + rm)
        mouth_w = max(6.0, float(np.linalg.norm(rm - lm)))
        cv2.ellipse(
            eye_excl,
            (int(round(mouth_mid[0])), int(round(mouth_mid[1]))),
            (int(round(0.58 * mouth_w)), int(round(0.28 * mouth_w))),
            angle,
            0,
            360,
            1.0,
            -1,
        )

        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        skin_profile = self.detector.sample_adaptive_skin_profile(bgr_img, det)
        skin_gate = skin_profile.build_chrominance_gate(lab)

        raw_mask = face_oval * (1.0 - eye_excl) * skin_gate
        ksize = max(5, int(round(0.12 * eye_d)) | 1)
        feathered = cv2.GaussianBlur(raw_mask, (ksize, ksize), max(1.5, 0.03 * eye_d))
        core = feathered > 0.5
        return feathered, core

    def process(
        self,
        bgr_img: np.ndarray,
        seed: int = 42,
        reference_skin_profile=None,
    ) -> Tuple[np.ndarray, SkinQualityMetrics]:
        """Apply halo-free plastic gloss suppression, optional reference skin tone/color lock, and natural skin micro-texture."""
        det = self.detector.detect(bgr_img)
        h, w = bgr_img.shape[:2]
        eye_d = max(8.0, det.inter_ocular_distance)
        skin_mask, core = self.build_dynamic_skin_mask(bgr_img, det)

        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        # Exclude background wall color from skin_mask to prevent any boundary halo
        bg_samples = np.concatenate(
            [lab[: max(4, h // 16), : max(4, w // 12)].reshape(-1, 3), lab[: max(4, h // 16), -max(4, w // 12) :].reshape(-1, 3)],
            axis=0,
        )
        bg_med = np.median(bg_samples, axis=0)
        bg_dist = np.sqrt((L - bg_med[0]) ** 2 + (A - bg_med[1]) ** 2 + (B - bg_med[2]) ** 2)
        not_bg = np.clip((bg_dist - 18.0) / 10.0, 0.0, 1.0)
        skin_mask = skin_mask * not_bg
        core = skin_mask > 0.5

        skin_pixels = L[core]
        if len(skin_pixels) > 50 and reference_skin_profile is not None:
            # Gently remove background color cast (A, B) and dark shadow shift (L) to match reference skin tone & color
            cur_l_med = float(np.median(L[core]))
            cur_a_med = float(np.median(A[core]))
            cur_b_med = float(np.median(B[core]))
            dl = float(np.clip((reference_skin_profile.l_median - cur_l_med) * 0.45, -10.0, 14.0))
            da = float(np.clip((reference_skin_profile.a_median - cur_a_med) * 0.65, -8.0, 8.0))
            db = float(np.clip((reference_skin_profile.b_median - cur_b_med) * 0.75, -10.0, 10.0))
            L = np.clip(L + skin_mask * dl, 0, 255)
            A = np.clip(A + skin_mask * da, 0, 255)
            B = np.clip(B + skin_mask * db, 0, 255)
            skin_pixels = L[core]

        p_knee = (
            float(np.percentile(skin_pixels, self.percentile_knee))
            if len(skin_pixels) > 50
            else 210.0
        )

        if len(skin_pixels) > 50:
            # Roll off ONLY extreme blown white clipping (> P99.2) in luminance ONLY.
            # NEVER shift A/B chrominance inside highlights (prevents orange/peach blotches near philtrum/eyelids).
            excess = np.maximum(0.0, L - p_knee)
            L_rolled = np.where(
                L > p_knee,
                p_knee + self.rolloff_range * np.tanh(excess / self.rolloff_scale),
                L,
            )
            L = (1.0 - skin_mask) * L + skin_mask * L_rolled

        # Fine pixel-scale optical grain only (never coarse >1.5px blotchy haze)
        scale_factor = float(np.clip(eye_d / 110.0, 0.5, 3.0))
        if self.micro_texture_amplitude > 0.0:
            rng = np.random.default_rng(seed)
            n_fine = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
            bp_fine = cv2.GaussianBlur(n_fine, (0, 0), 0.45) - cv2.GaussianBlur(
                n_fine, (0, 0), 0.95
            )
            bp_fine /= float(np.std(bp_fine) + 1e-6)
            L = np.clip(L + self.micro_texture_amplitude * bp_fine * skin_mask, 0, 255)

        # Anti-Smudge Organ Micro-Crisping: sharpen eyes, eyebrows, nostrils, and lip contour without adding skin gloss
        if self.organ_crisp_strength > 0.0:
            organ_mask = self.build_anatomical_organ_mask((h, w), det)
            l_fine = cv2.GaussianBlur(L, (0, 0), max(0.55, 0.75 * scale_factor))
            l_coarse = cv2.GaussianBlur(L, (0, 0), max(1.20, 1.95 * scale_factor))
            organ_detail = np.clip(l_fine - l_coarse, -10.0, 10.0)
            L = np.clip(L + self.organ_crisp_strength * organ_detail * organ_mask, 0, 255)

        lab_out = np.stack([L, np.clip(A, 0, 255), np.clip(B, 0, 255)], axis=-1).astype(np.uint8)
        out_bgr = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
        metrics = self.measure_metrics(out_bgr, det)
        return out_bgr, metrics

    def measure_metrics(
        self, bgr_img: np.ndarray, det: FaceDetectionResult | None = None
    ) -> SkinQualityMetrics:
        """Measure plastic specular glare ratio and natural skin micro-texture std on a portrait."""
        if det is None:
            det = self.detector.detect(bgr_img)
        _, core = self.build_dynamic_skin_mask(bgr_img, det)
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        skin_L = L[core]
        if len(skin_L) < 50:
            return SkinQualityMetrics(0.0, 2.0, 180.0, int(len(skin_L)))

        # True blown specular glare: desaturated bright hotspots (L > max(228, l_med+52) & chroma < 0.55*c_med) or clipping (> 247)
        l_med = float(np.median(skin_L))
        chroma = np.hypot(A[core] - 128.0, B[core] - 128.0)
        c_med = float(np.median(chroma)) + 1e-5
        glare_thresh = min(242.0, max(228.0, l_med + 52.0))
        glare_mask = ((skin_L > glare_thresh) & (chroma < 0.55 * c_med)) | (skin_L > 247.0)
        glare_ratio = float(np.mean(glare_mask))

        scale_factor = float(np.clip(det.inter_ocular_distance / 110.0, 0.5, 3.0))
        L_smooth = cv2.GaussianBlur(L, (0, 0), 2.2 * scale_factor)
        micro_res = (L - L_smooth)[core]
        micro_std = float(np.std(micro_res))
        p88 = float(np.percentile(skin_L, 88.0))
        return SkinQualityMetrics(
            specular_glare_ratio=glare_ratio,
            skin_micro_texture_std=micro_std,
            p88_luminance=p88,
            skin_pixel_count=int(len(skin_L)),
        )
