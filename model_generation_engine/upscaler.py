from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
import onnxruntime as ort

from .config import EngineConfig
from .detector import DynamicFaceDetector

FFHQ_TEMPLATE_512 = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19360],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)


class SuperResolutionEngine:
    """Universal 4x Real-ESRGAN (4xNomos8kSC) + YuNet 5-Point Aligned CodeFormer Focus Engine."""

    def __init__(self, config: EngineConfig, detector: DynamicFaceDetector) -> None:
        self.config = config
        self.detector = detector
        self.esrgan_bin = config.resolve_model_path("realesrgan-ncnn-vulkan")
        self.models_dir = self.esrgan_bin.parent
        config.resolve_model_path(f"{config.upscale_model_name}.param")
        config.resolve_model_path(f"{config.upscale_model_name}.bin")
        self.codeformer_path = config.resolve_model_path("codeformer.onnx")
        self._cf_session = ort.InferenceSession(
            str(self.codeformer_path), providers=["CPUExecutionProvider"]
        )

    def upscale_image_file(self, input_path: Path, output_path: Path) -> Path:
        """Run Real-ESRGAN (4xNomos8kSC) GPU super-resolution on an image file."""
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Upscale input file not found: {input_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(self.esrgan_bin),
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "-m",
            str(self.models_dir),
            "-n",
            self.config.upscale_model_name,
            "-s",
            str(self.config.upscale_factor),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not output_path.exists():
            raise RuntimeError(f"realesrgan-ncnn-vulkan failed to produce {output_path}")
        return output_path

    @staticmethod
    def validate_similarity_transform(M: np.ndarray, max_tilt_deg: float = 25.0) -> None:
        """Validate 2x3 affine matrix M to guarantee similarity transform without skew, shear, or reflection."""
        if M is None or M.shape != (2, 3) or not np.all(np.isfinite(M)):
            raise RuntimeError("Invalid or non-finite affine transform matrix M.")
        scale_x = float(np.hypot(M[0, 0], M[0, 1]))
        scale_y = float(np.hypot(M[1, 0], M[1, 1]))
        if scale_x < 0.05 or scale_y < 0.05 or scale_x > 25.0 or scale_y > 25.0:
            raise RuntimeError(f"Degenerate affine scale detected: sx={scale_x:.3f}, sy={scale_y:.3f}")
        if abs(scale_x - scale_y) / max(scale_x, scale_y, 1e-6) > 0.08:
            raise RuntimeError("Non-uniform affine skew detected in M; aborting to prevent face distortion.")
        dot = float(M[0, 0] * M[1, 0] + M[0, 1] * M[1, 1])
        cos_shear = abs(dot) / max(scale_x * scale_y, 1e-6)
        if cos_shear > 0.05:
            raise RuntimeError(f"Affine shear detected in M (cos_shear={cos_shear:.3f}); aborting.")
        det = float(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])
        if det <= 0:
            raise RuntimeError("Reflection (negative determinant) detected in M; aborting.")
        rot_deg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        if abs(rot_deg) > max_tilt_deg:
            raise RuntimeError(f"Abnormal rotation ({rot_deg:.1f} deg) in M exceeds max ({max_tilt_deg} deg).")

    def restore_face_focus_5pt(self, patch_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Restore crisp eye, eyewear, hair, and skin focus using 5-point YuNet + CodeFormer."""
        bh, bw = patch_bgr.shape[:2]
        det = self.detector.detect(patch_bgr)
        pts5 = det.landmarks_5pt

        if abs(det.eye_tilt_degrees) > 25.0:
            raise RuntimeError(
                f"Detected abnormal eye tilt ({det.eye_tilt_degrees:.1f} deg); aborting CodeFormer to prevent distortion."
            )

        M, _ = cv2.estimateAffinePartial2D(pts5, FFHQ_TEMPLATE_512, method=cv2.LMEDS)
        self.validate_similarity_transform(M)

        M_inv = cv2.invertAffineTransform(M)
        face_512 = cv2.warpAffine(
            patch_bgr, M, (512, 512), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
        )

        rgb = cv2.cvtColor(face_512, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None, ...]
        w_arr = np.array(self.config.codeformer_fidelity_w, dtype=np.float64)
        out = self._cf_session.run(["y"], {"x": inp, "w": w_arr})[0]
        cf_rgb = ((out[0].transpose(1, 2, 0) + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        cf_512 = cv2.cvtColor(cf_rgb, cv2.COLOR_RGB2BGR)

        lab_orig = cv2.cvtColor(face_512, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_cf = cv2.cvtColor(cf_512, cv2.COLOR_BGR2LAB).astype(np.float32)
        for c in (0, 1, 2):
            diff = np.mean(lab_orig[140:400, 140:372, c]) - np.mean(lab_cf[140:400, 140:372, c])
            lab_cf[:, :, c] = np.clip(lab_cf[:, :, c] + diff, 0, 255)
        enhanced_512 = cv2.cvtColor(lab_cf.astype(np.uint8), cv2.COLOR_LAB2BGR)

        if self.config.unsharp_amount > 0:
            blur = cv2.GaussianBlur(enhanced_512, (0, 0), self.config.unsharp_sigma)
            amt = self.config.unsharp_amount
            enhanced_512 = cv2.addWeighted(enhanced_512, 1.0 + amt, blur, -amt, 0)

        mask_512 = np.zeros((512, 512), dtype=np.float32)
        cv2.ellipse(mask_512, (256, 255), (175, 215), 0, 0, 360, 1.0, -1)
        mask_512 = cv2.GaussianBlur(mask_512, (45, 45), 14.0)

        warped_back = cv2.warpAffine(enhanced_512, M_inv, (bw, bh), flags=cv2.INTER_LANCZOS4)
        mask_back = cv2.warpAffine(mask_512, M_inv, (bw, bh), flags=cv2.INTER_LINEAR)[..., None]

        restored = (
            mask_back * warped_back.astype(np.float32)
            + (1.0 - mask_back) * patch_bgr.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)
        return restored, mask_back
