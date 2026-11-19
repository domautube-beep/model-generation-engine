from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np

from .config import EngineConfig
from .detector import DynamicFaceDetector


class Seamless5KCompositor:
    """Universal zero-hardcoding outpainting and sub-pixel compositor for any target resolution & aspect ratio.

    Automatically detects where the subject is positioned via YuNet to determine optimal left/right
    (or top/bottom) padding ratios when pad_left_ratio is None.
    """

    def __init__(self, config: EngineConfig, detector: DynamicFaceDetector) -> None:
        self.config = config
        self.detector = detector

    def _resolve_pad_left_ratio(self, resized_bgr: np.ndarray) -> float:
        """Auto-detect subject horizontal center to place negative space on the side with more room."""
        if self.config.canvas.pad_left_ratio is not None:
            return float(np.clip(self.config.canvas.pad_left_ratio, 0.0, 1.0))
        try:
            face = self.detector.detect(resized_bgr)
            cx, _ = face.center
            norm_x = cx / float(max(1, resized_bgr.shape[1]))
            # If subject is right-of-center (norm_x > 0.52), pad more on the left (0.65);
            # if subject is left-of-center (norm_x < 0.48), pad more on the right (0.35); otherwise center (0.50).
            if norm_x > 0.52:
                return 0.65
            elif norm_x < 0.48:
                return 0.35
            return 0.50
        except Exception:
            return 0.65

    def build_ultrawide_canvas(self, upscaled_scene_bgr: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Convert an upscaled scene into any target canvas resolution (e.g. 5K 5040x2160 21:9, 4K 16:9, 9:16).

        Returns (canvas_bgr, pad_left_px, scaled_scene_width_px).
        """
        spec = self.config.canvas
        uh, uw = upscaled_scene_bgr.shape[:2]
        scale = float(spec.target_height) / float(uh)
        new_w = int(round(uw * scale))
        resized = cv2.resize(
            upscaled_scene_bgr, (new_w, spec.target_height), interpolation=cv2.INTER_LANCZOS4
        )

        if new_w >= spec.target_width:
            try:
                face = self.detector.detect(resized)
                cx, _ = face.center
                x0 = int(round(cx - 0.5 * spec.target_width))
            except Exception:
                x0 = (new_w - spec.target_width) // 2
            x0 = max(0, min(new_w - spec.target_width, x0))
            return resized[:, x0 : x0 + spec.target_width].copy(), 0, spec.target_width

        pad_left_ratio = self._resolve_pad_left_ratio(resized)
        pad_total = spec.target_width - new_w
        pad_left = int(round(pad_total * pad_left_ratio))
        pad_right = pad_total - pad_left

        strip_w = min(40, max(8, new_w // 60))
        left_strip = np.median(resized[:, :strip_w, :], axis=1, keepdims=True).astype(np.uint8)
        right_strip = np.median(resized[:, -strip_w:, :], axis=1, keepdims=True).astype(np.uint8)
        left_pad = np.repeat(left_strip, pad_left, axis=1)
        right_pad = np.repeat(right_strip, pad_right, axis=1)

        if spec.wall_grain_sigma > 0:
            rng = np.random.default_rng(42)
            if pad_left > 0:
                left_pad = np.clip(
                    left_pad.astype(np.float32)
                    + rng.normal(0, spec.wall_grain_sigma, left_pad.shape).astype(np.float32),
                    0,
                    255,
                ).astype(np.uint8)
            if pad_right > 0:
                right_pad = np.clip(
                    right_pad.astype(np.float32)
                    + rng.normal(0, spec.wall_grain_sigma, right_pad.shape).astype(np.float32),
                    0,
                    255,
                ).astype(np.uint8)

        canvas = np.concatenate([left_pad, resized, right_pad], axis=1)

        blend_w = min(240, new_w // 3, max(spec.seam_blend_width_px, pad_left // 2, pad_right // 2))
        for i in range(blend_w):
            t = 0.5 - 0.5 * np.cos(np.pi * (i / float(max(1, blend_w))))
            if pad_left > 0 and i < new_w:
                canvas[:, pad_left + i] = (
                    (1.0 - t) * left_strip[:, 0, :].astype(np.float32)
                    + t * resized[:, i, :].astype(np.float32)
                ).clip(0, 255).astype(np.uint8)
            xr_idx = new_w - blend_w + i
            if pad_right > 0 and 0 <= xr_idx < new_w:
                canvas[:, pad_left + xr_idx] = (
                    (1.0 - t) * resized[:, xr_idx, :].astype(np.float32)
                    + t * right_strip[:, 0, :].astype(np.float32)
                ).clip(0, 255).astype(np.uint8)

        return canvas, pad_left, new_w

    def integrate_refined_crop(
        self,
        canvas_5k_bgr: np.ndarray,
        restored_crop_bgr: np.ndarray,
        crop_box_5k: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, Tuple[int, int], float, Tuple[int, int, int, int]]:
        """Align and seamlessly blend a restored upper-body crop into the 5K canvas."""
        x1, y1, x2, y2 = crop_box_5k
        bw, bh = x2 - x1, y2 - y1
        if bw < 32 or bh < 32:
            raise ValueError(f"crop_box_5k dimensions too small for integration: ({bw}x{bh})")
        orig_patch = canvas_5k_bgr[y1:y2, x1:x2].astype(np.float32)
        ref_patch = cv2.resize(
            restored_crop_bgr, (bw, bh), interpolation=cv2.INTER_LANCZOS4
        ).astype(np.float32)

        face_in_crop = self.detector.detect(ref_patch.astype(np.uint8))
        hx1, hy1, hx2, hy2 = self.detector.compute_head_halo_box((bh, bw), face_in_crop)

        ring_mask = np.ones((bh, bw), dtype=bool)
        margin_y = min(max(8, int(bh * 0.10)), bh // 3)
        margin_x = min(max(8, int(bw * 0.12)), bw // 3)
        ring_mask[margin_y : bh - margin_y, margin_x : bw - margin_x] = False

        best_err = 1e18
        best_dx, best_dy = 0, 0
        border_guard = min(10, bh // 6, bw // 6)
        for dy in range(-6, 7):
            for dx in range(-6, 7):
                shifted = np.roll(np.roll(ref_patch, dy, axis=0), dx, axis=1)
                valid = ring_mask.copy()
                if border_guard > 0:
                    valid[:border_guard, :] = False
                    valid[-border_guard:, :] = False
                    valid[:, :border_guard] = False
                    valid[:, -border_guard:] = False
                if not np.any(valid):
                    valid = ring_mask
                err = float(np.mean(np.abs(shifted[valid] - orig_patch[valid])))
                if err < best_err:
                    best_err = err
                    best_dx, best_dy = dx, dy

        ref_aligned = np.roll(np.roll(ref_patch, best_dy, axis=0), best_dx, axis=1)

        wide_head_mask = np.zeros((bh, bw), dtype=np.float32)
        pad_x = min(24, bw // 8)
        pad_top = min(16, bh // 8)
        pad_bot = min(32, bh // 6)
        wx1 = int(np.clip(hx1 - int((hx2 - hx1) * 0.18), pad_x, bw - pad_x - 2))
        wx2 = int(np.clip(hx2 + int((hx2 - hx1) * 0.18), wx1 + 2, bw - pad_x))
        wy1 = int(np.clip(hy1 - int((hy2 - hy1) * 0.12), pad_top, bh - pad_bot - 2))
        wy2 = int(np.clip(hy2, wy1 + 2, bh - pad_bot))
        cv2.rectangle(wide_head_mask, (wx1, wy1), (wx2, wy2), 1.0, -1)
        wide_head_mask = cv2.GaussianBlur(wide_head_mask, (41, 41), 11.0)[..., None]

        bg_halo = np.zeros((bh, bw), dtype=bool)
        halo_y1 = max(0, wy1 - min(20, bh // 10))
        halo_y2 = min(bh, wy1 + min(40, max(4, (wy2 - wy1) // 3)))
        if halo_y2 > halo_y1 and wx2 > wx1:
            bg_halo[halo_y1:halo_y2, wx1:wx2] = True
            fx, fy, fw, fh = [int(round(v)) for v in face_in_crop.bbox]
            bg_halo[max(0, fy) : min(bh, fy + fh), max(0, fx) : min(bw, fx + fw)] = False

        if np.any(bg_halo):
            for c in range(3):
                diff_c = float(np.median(orig_patch[:, :, c][bg_halo] - ref_aligned[:, :, c][bg_halo]))
                if np.isfinite(diff_c):
                    ref_aligned[:, :, c] = np.clip(ref_aligned[:, :, c] + diff_c, 0, 255)

        blended = (wide_head_mask * ref_aligned + (1.0 - wide_head_mask) * orig_patch).clip(
            0, 255
        ).astype(np.uint8)

        out_canvas = canvas_5k_bgr.copy()
        out_canvas[y1:y2, x1:x2] = blended
        return out_canvas, (best_dx, best_dy), best_err, (hx1, hy1, hx2, hy2)

    def integrate_tonematched_crop(
        self,
        canvas_5k_bgr: np.ndarray,
        refined_crop_bgr: np.ndarray,
        crop_box_5k: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, Tuple[int, int], float, Tuple[int, int, int, int]]:
        """Seamlessly integrate a Tone-Matched 3D In-Place refined 16:9 upper-body crop into the 5K canvas.

        Unlike narrow head-only boxes or 2D oval face pasting, this preserves 100% of the refined 3D head,
        eyewear temple arms, neck, collar, and shoulder proportions inside the crop interior while applying
        an edge-aware cosine feather + low-frequency border color match strictly on the outer perimeter.
        """
        ch, cw = canvas_5k_bgr.shape[:2]
        x1, y1, x2, y2 = crop_box_5k
        bw, bh = x2 - x1, y2 - y1
        if bw < 32 or bh < 32:
            raise ValueError(f"crop_box_5k dimensions too small for integration: ({bw}x{bh})")

        orig_patch = canvas_5k_bgr[y1:y2, x1:x2].astype(np.float32)
        ref_patch = cv2.resize(
            refined_crop_bgr, (bw, bh), interpolation=cv2.INTER_LANCZOS4
        ).astype(np.float32)

        face_in_crop = self.detector.detect(ref_patch.astype(np.uint8))
        hx1, hy1, hx2, hy2 = self.detector.compute_head_halo_box((bh, bw), face_in_crop)

        ring_mask = np.zeros((bh, bw), dtype=bool)
        margin_x = min(max(12, int(bw * 0.08)), bw // 4)
        margin_y = min(max(12, int(bh * 0.08)), bh // 4)
        if x1 > 0:
            ring_mask[:, :margin_x] = True
        if x2 < cw:
            ring_mask[:, -margin_x:] = True
        if y1 > 0:
            ring_mask[:margin_y, :] = True
        if y2 < ch:
            ring_mask[-margin_y:, :] = True

        best_err = 1e18
        best_dx, best_dy = 0, 0
        border_guard = min(8, bh // 8, bw // 8)
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                shifted = np.roll(np.roll(ref_patch, dy, axis=0), dx, axis=1)
                valid = ring_mask.copy()
                if border_guard > 0:
                    valid[:border_guard, :] = False
                    valid[-border_guard:, :] = False
                    valid[:, :border_guard] = False
                    valid[:, -border_guard:] = False
                if not np.any(valid):
                    valid = ring_mask
                err = float(np.mean(np.abs(shifted[valid] - orig_patch[valid])))
                if err < best_err:
                    best_err = err
                    best_dx, best_dy = dx, dy

        ref_aligned = np.roll(np.roll(ref_patch, best_dy, axis=0), best_dx, axis=1)
        # Restore rolled wrap-around strips from orig_patch so edges never wrap opposite borders
        if best_dy > 0:
            ref_aligned[:best_dy, :] = orig_patch[:best_dy, :]
        elif best_dy < 0:
            ref_aligned[best_dy:, :] = orig_patch[best_dy:, :]
        if best_dx > 0:
            ref_aligned[:, :best_dx] = orig_patch[:, :best_dx]
        elif best_dx < 0:
            ref_aligned[:, best_dx:] = orig_patch[:, best_dx:]

        ramp_l = min(96, bw // 10) if x1 > 0 else 0
        ramp_r = min(96, bw // 10) if x2 < cw else 0
        ramp_t = min(64, bh // 10) if y1 > 0 else 0
        ramp_b = min(96, bh // 8) if y2 < ch else 0

        wx = np.ones(bw, dtype=np.float32)
        if ramp_l > 0:
            t = np.linspace(0.0, 1.0, ramp_l, dtype=np.float32)
            wx[:ramp_l] = 0.5 - 0.5 * np.cos(np.pi * t)
        if ramp_r > 0:
            t = np.linspace(1.0, 0.0, ramp_r, dtype=np.float32)
            wx[-ramp_r:] = np.minimum(wx[-ramp_r:], 0.5 - 0.5 * np.cos(np.pi * t))

        wy = np.ones(bh, dtype=np.float32)
        if ramp_t > 0:
            t = np.linspace(0.0, 1.0, ramp_t, dtype=np.float32)
            wy[:ramp_t] = 0.5 - 0.5 * np.cos(np.pi * t)
        if ramp_b > 0:
            t = np.linspace(1.0, 0.0, ramp_b, dtype=np.float32)
            wy[-ramp_b:] = np.minimum(wy[-ramp_b:], 0.5 - 0.5 * np.cos(np.pi * t))

        w_mask = wy[:, None] * wx[None, :]

        # Low-frequency ambient wall/border offset match confined strictly to the outer perimeter
        low_diff = cv2.GaussianBlur(orig_patch - ref_aligned, (0, 0), 36.0)
        perimeter_weight = (1.0 - w_mask)[..., None]
        ref_harmonized = np.clip(ref_aligned + perimeter_weight * low_diff, 0.0, 255.0)

        # Sharp-edge anti-ghosting guard strictly inside the outer feather zone (0 < w_mask < 0.98)
        local_l1 = np.mean(np.abs(ref_harmonized - orig_patch), axis=2)
        in_ramp = (w_mask > 0.0) & (w_mask < 0.98)
        edge_suppress = np.zeros_like(w_mask)
        edge_suppress[in_ramp] = np.clip((local_l1[in_ramp] - 18.0) / 28.0, 0.0, 1.0)
        w_eff = w_mask * (1.0 - edge_suppress * (1.0 - w_mask))
        w_eff = cv2.GaussianBlur(w_eff, (15, 15), 3.5)[..., None]

        blended = np.clip(w_eff * ref_harmonized + (1.0 - w_eff) * orig_patch, 0.0, 255.0).astype(
            np.uint8
        )
        seam_err = float(np.mean(np.abs(blended.astype(np.float32)[ring_mask] - orig_patch[ring_mask])))

        out_canvas = canvas_5k_bgr.copy()
        out_canvas[y1:y2, x1:x2] = blended
        return out_canvas, (best_dx, best_dy), seam_err, (hx1, hy1, hx2, hy2)
