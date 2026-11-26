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
        self, bgr_img: np.ndarray, det: FaceDetectionResult, lab_f32: np.ndarray | None = None
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

        lab = lab_f32 if lab_f32 is not None else cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        skin_profile = self.detector.sample_adaptive_skin_profile(bgr_img, det)
        skin_gate = skin_profile.build_chrominance_gate(lab)

        raw_mask = face_oval * (1.0 - eye_excl) * skin_gate
        ksize = max(5, int(round(0.12 * eye_d)) | 1)
        if ksize > 25 and min(h, w) > 256:
            sm_w, sm_h = max(32, w // 4), max(32, h // 4)
            sm_raw = cv2.resize(raw_mask, (sm_w, sm_h), interpolation=cv2.INTER_AREA)
            sm_k = max(5, (ksize // 4) | 1)
            sm_blur = cv2.GaussianBlur(sm_raw, (sm_k, sm_k), max(1.0, 0.0075 * eye_d))
            feathered = cv2.resize(sm_blur, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
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
        h_in, w_in = bgr_img.shape[:2]
        fx, fy, fw, fh = det.bbox
        # Auto-crop large canvases tightly around face oval (1.2x face bbox; zero skin pixels exist outside face_oval!)
        rx1 = max(0, int(fx - 0.12 * fw))
        ry1 = max(0, int(fy - 0.12 * fh))
        rx2 = min(w_in, int(fx + 1.12 * fw))
        ry2 = min(h_in, int(fy + 1.12 * fh))
        if max(h_in, w_in) > 768 and fw > 8 and fh > 8 and ((rx2 - rx1) * (ry2 - ry1)) < 0.88 * (h_in * w_in):
            bgr_img = bgr_img[ry1:ry2, rx1:rx2]
            det = FaceDetectionResult(
                bbox=(fx - rx1, fy - ry1, fw, fh),
                landmarks_5pt=det.landmarks_5pt - np.array([rx1, ry1], dtype=np.float32),
                confidence=det.confidence,
            )
        # If ultra-large close-up face (>900px bbox), evaluate skin mask & micro-std at canonical 768px face height (~10x faster, scale-normalized!)
        if max(bgr_img.shape[:2]) > 960 and det.bbox[3] > 680:
            sc_m = 680.0 / float(det.bbox[3])
            mw = max(64, int(round(bgr_img.shape[1] * sc_m)))
            mh = max(64, int(round(bgr_img.shape[0] * sc_m)))
            bgr_img = cv2.resize(bgr_img, (mw, mh), interpolation=cv2.INTER_AREA)
            det = FaceDetectionResult(
                bbox=tuple(float(v) * sc_m for v in det.bbox),  # type: ignore[arg-type]
                landmarks_5pt=(det.landmarks_5pt * sc_m).astype(np.float32),
                confidence=det.confidence,
            )
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        _, core = self.build_dynamic_skin_mask(bgr_img, det, lab_f32=lab)
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

    def harmonize_intrinsic_skin_albedo_and_texture(
        self,
        canvas_bgr: np.ndarray,
        reference_face_bgr: np.ndarray,
        ambient_blend_ratio: float = 0.28,
        seed: int = 42,
    ) -> Tuple[np.ndarray, dict]:
        """v2.2 Universal Zero-Hardcoding Skin Color (`피부색감`) & Skin Texture (`피부질감`) Mastering.

        1. Measures the reference person's true CIE L*a*b* cheek skin undertone and IOD-normalized micro-pore texture
           directly from `reference_face_bgr` (`Gen-0`).
        2. Builds a seamless Face + Exposed-Neck/Collarbone + Exposed-Hand skin mask on `canvas_bgr` (excluding eyes,
           sclera, eyebrows, nostrils, lip vermilion, hair, and garments) so zero face-vs-neck seam can ever appear.
        3. Restores the subject's alive rosy/melanin undertone (`1 - ambient_blend_ratio` intrinsic `Gen-0` albedo +
           `ambient_blend_ratio` local scene lighting), eliminating corpse-grey/olive cast and `de_yellow` over-cooling.
        4. Matches fine velvety pore micro-texture (`[0.8..2.2px]` at canonical IOD) to `Gen-0` while damping blotchy
           mid-frequency AI wax mottle and rolling off shiny plastic cheekbone hotspots.
        """
        ref_det = self.detector.detect(reference_face_bgr)
        dst_det = self.detector.detect(canvas_bgr)
        _, ref_core = self.build_dynamic_skin_mask(reference_face_bgr, ref_det)
        ref_lab = cv2.cvtColor(reference_face_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        if int(np.sum(ref_core)) < 40:
            return canvas_bgr, {"applied": False, "reason": "insufficient_ref_skin"}

        ref_l_med = float(np.median(ref_lab[:, :, 0][ref_core]))
        ref_a_med = float(np.median(ref_lab[:, :, 1][ref_core]))
        ref_b_med = float(np.median(ref_lab[:, :, 2][ref_core]))

        ref_iod = max(12.0, float(ref_det.inter_ocular_distance))
        dst_iod = max(12.0, float(dst_det.inter_ocular_distance))
        ref_sig_fine = max(0.65, 1.35 * (ref_iod / 130.0))
        ref_pore_band = ref_lab[:, :, 0] - cv2.GaussianBlur(ref_lab[:, :, 0], (0, 0), ref_sig_fine)
        ref_pore_std = float(np.clip(np.std(ref_pore_band[ref_core]), 0.75, 2.80))

        # Work inside a generous upper-body ROI covering head, neck, collarbones, and hands for fast 4K execution
        H, W = canvas_bgr.shape[:2]
        fx, fy, fw, fh = dst_det.bbox
        rx1 = max(0, int(round(fx - 2.2 * fw)))
        ry1 = max(0, int(round(fy - 0.8 * fh)))
        rx2 = min(W, int(round(fx + 3.2 * fw)))
        ry2 = min(H, int(round(fy + 4.2 * fh)))

        roi = canvas_bgr[ry1:ry2, rx1:rx2].copy()
        rh, rw = roi.shape[:2]
        roi_det = self.detector.detect(roi)
        face_skin_mask, face_core = self.build_dynamic_skin_mask(roi, roi_det)
        if int(np.sum(face_core)) < 40:
            return canvas_bgr, {"applied": False, "reason": "insufficient_dst_skin"}

        roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, A, B = roi_lab[:, :, 0], roi_lab[:, :, 1], roi_lab[:, :, 2]

        cur_l_med = float(np.median(L[face_core]))
        cur_a_med = float(np.median(A[face_core]))
        cur_b_med = float(np.median(B[face_core]))

        # Extend skin mask seamlessly onto exposed neck/collarbone/hand pixels that match the current skin distribution
        rfx, rfy, rfw, rfh = roi_det.bbox
        body_skin_zone = np.zeros((rh, rw), dtype=np.float32)
        cv2.ellipse(
            body_skin_zone,
            (int(round(rfx + 0.50 * rfw)), int(round(rfy + 1.15 * rfh))),
            (max(12, int(round(1.65 * rfw))), max(16, int(round(2.10 * rfh)))),
            0.0,
            0,
            360,
            1.0,
            -1,
        )
        organ_excl = self.build_anatomical_organ_mask((rh, rw), roi_det)
        skin_dist = np.sqrt(
            ((L - cur_l_med) / 28.0) ** 2 + ((A - cur_a_med) / 7.5) ** 2 + ((B - cur_b_med) / 9.5) ** 2
        )
        chrom_sim = np.clip(1.35 - skin_dist, 0.0, 1.0) * body_skin_zone * (1.0 - np.clip(organ_excl * 1.15, 0.0, 1.0))
        full_skin_mask = np.maximum(face_skin_mask, chrom_sim * 0.88)
        k_feather = max(9, int(round(0.16 * dst_iod)) | 1)
        full_skin_mask = cv2.GaussianBlur(full_skin_mask, (k_feather, k_feather), max(2.2, 0.05 * dst_iod))
        # Taper ROI borders to 0 so ROI paste is 100% invisible
        b_pad = max(8, int(round(0.12 * dst_iod)))
        border_w = np.ones((rh, rw), dtype=np.float32)
        for bi in range(min(b_pad, rh // 4, rw // 4)):
            w_val = float(bi) / float(b_pad)
            border_w[bi, :] = np.minimum(border_w[bi, :], w_val)
            border_w[rh - 1 - bi, :] = np.minimum(border_w[rh - 1 - bi, :], w_val)
            border_w[:, bi] = np.minimum(border_w[:, bi], w_val)
            border_w[:, rw - 1 - bi] = np.minimum(border_w[:, rw - 1 - bi], w_val)
        full_skin_mask *= border_w

        # 1. Illuminant-aware Skin Color (`피부색감`) restoration
        keep_w = float(np.clip(1.0 - ambient_blend_ratio, 0.50, 0.85))
        da = float(np.clip((ref_a_med - cur_a_med) * keep_w, -7.5, 8.5))
        db = float(np.clip((ref_b_med - cur_b_med) * keep_w, -8.5, 9.5))
        dl = float(np.clip((ref_l_med - cur_l_med) * 0.26, -6.0, 9.5))

        L_new = np.clip(L + full_skin_mask * dl, 0.0, 255.0)
        A_new = np.clip(A + full_skin_mask * da, 0.0, 255.0)
        B_new = np.clip(B + full_skin_mask * db, 0.0, 255.0)

        # 2. Subsurface Smoothness + Scale-Matched Velvety Pore Micro-Texture (`피부질감`)
        dst_sig_fine = max(0.75, 1.35 * (dst_iod / 130.0))
        dst_sig_mid = max(2.20, 3.80 * (dst_iod / 130.0))
        l_blur_fine = cv2.GaussianBlur(L_new, (0, 0), dst_sig_fine)
        l_blur_mid = cv2.GaussianBlur(L_new, (0, 0), dst_sig_mid)
        pore_layer = L_new - l_blur_fine
        mottle_layer = l_blur_fine - l_blur_mid

        cur_pore_std = float(np.std(pore_layer[face_core])) + 1e-5
        target_pore_gain = float(np.clip(ref_pore_std / cur_pore_std, 0.78, 1.32))
        # Gently calm blotchy AI mid-frequency wax mottle by 18% while matching exact fine pore amplitude
        L_tex = l_blur_mid + 0.82 * mottle_layer + target_pore_gain * pore_layer

        # If upscaled low-pixel candidate lacks fine micro-grain, add ultra-subtle optical velvety pore grain
        if cur_pore_std < ref_pore_std * 0.82:
            rng = np.random.default_rng(seed)
            n_raw = rng.normal(0.0, 1.0, (rh, rw)).astype(np.float32)
            n_bp = cv2.GaussianBlur(n_raw, (0, 0), 0.45 * (dst_iod / 130.0 + 0.5)) - cv2.GaussianBlur(
                n_raw, (0, 0), 1.15 * (dst_iod / 130.0 + 0.5)
            )
            n_bp /= float(np.std(n_bp) + 1e-6)
            synth_amp = float(np.clip((ref_pore_std - cur_pore_std) * 0.45, 0.0, 0.85))
            L_tex = L_tex + synth_amp * n_bp

        # Velvety specular highlight roll-off (> P98.8) so shiny plastic cheekbone patches become soft natural sheen
        p_knee = float(np.percentile(L_tex[face_core], 98.8))
        excess = np.maximum(0.0, L_tex - p_knee)
        L_tex = np.where(L_tex > p_knee, p_knee + 3.2 * np.tanh(excess / 8.0), L_tex)

        L_final = np.clip((1.0 - full_skin_mask) * L + full_skin_mask * L_tex, 0.0, 255.0)
        roi_out = cv2.cvtColor(
            np.stack([L_final, A_new, B_new], axis=-1).astype(np.uint8),
            cv2.COLOR_LAB2BGR,
        )
        out_canvas = canvas_bgr.copy()
        out_canvas[ry1:ry2, rx1:rx2] = roi_out
        return out_canvas, {
            "applied": True,
            "ref_lab_med": [round(ref_l_med, 1), round(ref_a_med - 128.0, 2), round(ref_b_med - 128.0, 2)],
            "applied_delta_lab": [round(dl, 2), round(da, 2), round(db, 2)],
            "ref_pore_std": round(ref_pore_std, 3),
            "pore_gain": round(target_pore_gain, 3),
        }
