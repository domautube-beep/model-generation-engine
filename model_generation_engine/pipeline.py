from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from PIL import Image

from .apple_vision import AppleVisionAnalyzer
from .config import EngineConfig
from .detector import DynamicFaceDetector
from .face_reattach import FaceLockCompositor, HeadRedrawCompositor
from .skin_processor import MatteSkinProcessor
from .upscaler import SuperResolutionEngine
from .compositor import Seamless5KCompositor
from .quality_gate import QualityGateAuditor, QualityReport


class ModelGenerationPipeline:
    """Universal Zero-Hardcoding 5K Model Generation & Super-Resolution Pipeline."""

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()
        yunet_path = self.config.resolve_model_path("yunet.onnx")
        self.detector = DynamicFaceDetector(yunet_path)
        self.skin_processor = MatteSkinProcessor(self.detector)
        self.upscaler = SuperResolutionEngine(self.config, self.detector)
        self.compositor = Seamless5KCompositor(self.config, self.detector)
        self.auditor = QualityGateAuditor(self.config, self.detector)
        bin_dir = (self.config.vault_dir / "models" / "apple_vision_bin") if self.config.vault_dir else None
        self.apple_vision = AppleVisionAnalyzer(bin_dir=bin_dir)
        self.facelock = FaceLockCompositor(self.apple_vision)
        self.head_redraw = HeadRedrawCompositor(self.apple_vision)

    def refine_matte_skin_texture(self, bgr_img: np.ndarray, seed: int = 42) -> np.ndarray:
        """Eliminate plastic/oily facial specular gloss and preserve scale-adaptive natural skin micro-texture."""
        out_bgr, _ = self.skin_processor.process(bgr_img, seed=seed)
        return out_bgr

    def run_single_pass_5k(
        self,
        pure_scratch_scene_path: Path,
        output_hero_path: Path,
        output_closeup_path: Optional[Path] = None,
        seed: int = 42,
        reference_face_path: Optional[Path] = None,
        reattach_mode: str = "none",
        redrawn_head_path: Optional[Path] = None,
    ) -> Dict[str, object]:
        """Execute Single-Pass Pure From-Scratch 5K Pipeline with optional Face Re-Attachment (`facelock`, `head_redraw`, `hybrid`) & 9-Gate Audit."""
        ref_bgr: Optional[np.ndarray] = None
        ref_skin_profile = None
        if reference_face_path is not None and Path(reference_face_path).exists():
            ref_bgr = cv2.imread(str(reference_face_path))
            if ref_bgr is not None:
                ref_det = self.detector.detect(ref_bgr)
                ref_skin_profile = self.detector.sample_adaptive_skin_profile(ref_bgr, ref_det)

        with tempfile.TemporaryDirectory(prefix="single_pass_5k_") as tmp_dir:
            tmp = Path(tmp_dir)
            up_path = self.upscaler.upscale_image_file(Path(pure_scratch_scene_path), tmp / "up_scene.jpg")
            up_bgr = cv2.imread(str(up_path))
            canvas_5k, _, _ = self.compositor.build_ultrawide_canvas(up_bgr)

            reattach_info: Dict[str, object] = {}
            if reattach_mode in ("head_redraw", "hybrid") and redrawn_head_path is not None and Path(redrawn_head_path).exists():
                up_head_path = self.upscaler.upscale_image_file(Path(redrawn_head_path), tmp / "up_head.jpg")
                head_pil = Image.open(up_head_path).convert("RGB")
                canvas_pil = Image.fromarray(cv2.cvtColor(canvas_5k, cv2.COLOR_BGR2RGB))
                stitched_pil, hr_meta = self.head_redraw.blend_redrawn_head(canvas_pil, head_pil)
                canvas_5k = cv2.cvtColor(np.asarray(stitched_pil), cv2.COLOR_RGB2BGR)
                reattach_info["head_redraw"] = hr_meta

            if reattach_mode in ("facelock", "hybrid") and ref_bgr is not None:
                ref_pil = Image.fromarray(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB))
                canvas_pil = Image.fromarray(cv2.cvtColor(canvas_5k, cv2.COLOR_BGR2RGB))
                locked_pil, fl_meta = self.facelock.apply_facelock(ref_pil, canvas_pil)
                canvas_5k = cv2.cvtColor(np.asarray(locked_pil), cv2.COLOR_RGB2BGR)
                reattach_info["facelock"] = fl_meta

            face_5k = self.detector.detect(canvas_5k)
            crop_box_5k = self.detector.compute_upper_body_crop_box(
                canvas_5k.shape[:2], face_5k, aspect_ratio=self.config.canvas.crop_aspect_ratio
            )
            X1, Y1, X2, Y2 = crop_box_5k
            vx1, vy1 = max(0, X1 - 120), max(0, Y1 - 60)
            vx2, vy2 = min(canvas_5k.shape[1], X2 + 120), min(canvas_5k.shape[0], Y2 + 160)

            roi = canvas_5k[vy1:vy2, vx1:vx2].copy()
            roi_face = self.detector.detect(roi)
            min_org_grad, _ = self.auditor.measure_per_organ_sharpness(roi, roi_face)
            if (
                self.config.auto_restore_distant_face_iod_px > 0.0
                and roi_face.inter_ocular_distance < self.config.auto_restore_distant_face_iod_px
            ) or min_org_grad < self.config.min_organ_gradient_energy:
                roi, _ = self.upscaler.restore_face_focus_5pt(roi)

            roi_matte, skin_metrics = self.skin_processor.process(
                roi, seed=seed, reference_skin_profile=ref_skin_profile
            )
            canvas_5k[vy1:vy2, vx1:vx2] = roi_matte

            report = self.auditor.audit(
                canvas_5k, crop_box_5k, seam_l1_diff=0.0, reference_face_bgr=ref_bgr
            )
            if not report.passed:
                raise RuntimeError(f"QualityGateAuditor rejected output: {report.message}")

            output_hero_path = Path(output_hero_path)
            output_hero_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_hero_path), canvas_5k, [int(cv2.IMWRITE_JPEG_QUALITY), 97])
            if output_closeup_path:
                cv2.imwrite(str(output_closeup_path), roi_matte, [int(cv2.IMWRITE_JPEG_QUALITY), 97])

            return {
                "output_hero_path": str(output_hero_path),
                "crop_box_5k": crop_box_5k,
                "skin_metrics": skin_metrics,
                "reattach_info": reattach_info,
                "quality_report": report,
            }

    def extract_dynamic_upper_body_crop(
        self, fullbody_scene_path: Path, output_crop_path: Path
    ) -> Tuple[Path, Tuple[int, int, int, int]]:
        """Dynamically detect the subject on a base canvas and extract an aspect-locked upper-body crop."""
        scene_bgr = cv2.imread(str(fullbody_scene_path))
        if scene_bgr is None:
            raise FileNotFoundError(f"Cannot read full-body scene at {fullbody_scene_path}")

        canvas_5k, _, _ = self.compositor.build_ultrawide_canvas(scene_bgr)
        h5, w5 = canvas_5k.shape[:2]
        w_base, h_base = w5 // 2, h5 // 2
        base_canvas = cv2.resize(canvas_5k, (w_base, h_base), interpolation=cv2.INTER_LANCZOS4)

        face_det = self.detector.detect(base_canvas)
        crop_box_base = self.detector.compute_upper_body_crop_box(
            (h_base, w_base), face_det, aspect_ratio=self.config.canvas.crop_aspect_ratio
        )
        x1, y1, x2, y2 = crop_box_base
        crop_bgr = base_canvas[y1:y2, x1:x2]
        render_w, render_h = self.config.canvas.crop_render_size
        crop_resized = cv2.resize(crop_bgr, (render_w, render_h), interpolation=cv2.INTER_LANCZOS4)

        output_crop_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_crop_path), crop_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        return output_crop_path, crop_box_base

    def run_5k_synthesis(
        self,
        fullbody_scene_path: Path,
        refined_crop_path: Path,
        output_hero_path: Path,
        output_closeup_path: Optional[Path] = None,
        output_macro_path: Optional[Path] = None,
        crop_box_base_override: Optional[Tuple[int, int, int, int]] = None,
        cached_up_scene_path: Optional[Path] = None,
        cached_up_crop_path: Optional[Path] = None,
    ) -> Dict[str, object]:
        """2-Stage distant full-body synthesis with dynamic crop alignment and 6-Gate Audit."""
        with tempfile.TemporaryDirectory(prefix="model_5k_") as tmp_dir:
            tmp = Path(tmp_dir)
            if cached_up_scene_path and Path(cached_up_scene_path).exists():
                up_scene_path = Path(cached_up_scene_path)
            else:
                up_scene_path = self.upscaler.upscale_image_file(
                    Path(fullbody_scene_path), tmp / "up_scene.jpg"
                )

            if cached_up_crop_path and Path(cached_up_crop_path).exists():
                up_crop_path = Path(cached_up_crop_path)
            else:
                up_crop_path = self.upscaler.upscale_image_file(
                    Path(refined_crop_path), tmp / "up_crop.jpg"
                )

            up_scene_bgr = cv2.imread(str(up_scene_path))
            up_crop_bgr = cv2.imread(str(up_crop_path))

            canvas_5k, _, _ = self.compositor.build_ultrawide_canvas(up_scene_bgr)

            if crop_box_base_override is not None:
                x1, y1, x2, y2 = crop_box_base_override
                crop_box_5k = (x1 * 2, y1 * 2, x2 * 2, y2 * 2)
            else:
                face_5k = self.detector.detect(canvas_5k)
                crop_box_5k = self.detector.compute_upper_body_crop_box(
                    canvas_5k.shape[:2], face_5k, aspect_ratio=self.config.canvas.crop_aspect_ratio
                )

            bw = crop_box_5k[2] - crop_box_5k[0]
            bh = crop_box_5k[3] - crop_box_5k[1]
            patch_bgr = cv2.resize(up_crop_bgr, (bw, bh), interpolation=cv2.INTER_LANCZOS4)

            restored_patch_bgr, _ = self.upscaler.restore_face_focus_5pt(patch_bgr)

            final_canvas_5k, shift_xy, border_err, head_box_in_crop = (
                self.compositor.integrate_refined_crop(canvas_5k, restored_patch_bgr, crop_box_5k)
            )

            report: QualityReport = self.auditor.audit(final_canvas_5k, crop_box_5k, border_err)
            if not report.passed:
                raise RuntimeError(f"QualityGateAuditor rejected output: {report.message}")

            output_hero_path = Path(output_hero_path)
            output_hero_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_hero_path), final_canvas_5k, [int(cv2.IMWRITE_JPEG_QUALITY), 97])

            X1, Y1, X2, Y2 = crop_box_5k
            HX1, HY1, HX2, HY2 = head_box_in_crop
            if output_closeup_path:
                vx1, vy1 = max(0, X1 - 120), max(0, Y1 - 60)
                vx2, vy2 = min(final_canvas_5k.shape[1], X2 + 120), min(
                    final_canvas_5k.shape[0], Y2 + 160
                )
                cv2.imwrite(
                    str(output_closeup_path),
                    final_canvas_5k[vy1:vy2, vx1:vx2],
                    [int(cv2.IMWRITE_JPEG_QUALITY), 97],
                )

            if output_macro_path:
                fx1, fy1 = max(0, X1 + HX1 - 40), max(0, Y1 + HY1 - 40)
                fx2, fy2 = min(final_canvas_5k.shape[1], X1 + HX2 + 40), min(
                    final_canvas_5k.shape[0], Y1 + HY2 + 40
                )
                cv2.imwrite(
                    str(output_macro_path),
                    final_canvas_5k[fy1:fy2, fx1:fx2],
                    [int(cv2.IMWRITE_JPEG_QUALITY), 97],
                )

            return {
                "output_hero_path": str(output_hero_path),
                "crop_box_5k": crop_box_5k,
                "alignment_shift": shift_xy,
                "quality_report": report,
            }


# Backward-compatible alias
PortraitIdentityPipeline = ModelGenerationPipeline
