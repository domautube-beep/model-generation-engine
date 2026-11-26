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
from .identity_turnaround import RealPersonTurnaroundEngine
from .scene_radiometry import PostGenCohesionMasterer, SingleLensRadiometricSpec
from .skin_processor import MatteSkinProcessor
from .upscaler import SuperResolutionEngine
from .quality_gate import QualityGateAuditor, QualityReport


class ModelGenerationPipeline:
    """Universal Zero-Hardcoding 5K Model Generation & Super-Resolution Pipeline."""

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()
        yunet_path = self.config.resolve_model_path("yunet.onnx")
        self.detector = DynamicFaceDetector(yunet_path)
        self.skin_processor = MatteSkinProcessor(self.detector)
        self.upscaler = SuperResolutionEngine(self.config, self.detector)
        self.auditor = QualityGateAuditor(self.config, self.detector)
        bin_dir = (self.config.vault_dir / "models" / "apple_vision_bin") if self.config.vault_dir else None
        self.apple_vision = AppleVisionAnalyzer(bin_dir=bin_dir)
        self.turnaround = RealPersonTurnaroundEngine(
            self.config, self.detector, self.apple_vision, self.upscaler
        )
        self._compositor = None
        self._facelock = None
        self._head_redraw = None
        self._relit_integrator = None
        self._proportion_calibrator = None
        self._batch_composer = None

    @property
    def compositor(self):
        if self._compositor is None:
            from .compositor import Seamless5KCompositor
            self._compositor = Seamless5KCompositor(self.config, self.detector)
        return self._compositor

    @property
    def facelock(self):
        if self._facelock is None:
            from .face_reattach import FaceLockCompositor
            self._facelock = FaceLockCompositor(self.apple_vision)
        return self._facelock

    @property
    def head_redraw(self):
        if self._head_redraw is None:
            from .face_reattach import HeadRedrawCompositor
            self._head_redraw = HeadRedrawCompositor(self.apple_vision)
        return self._head_redraw

    @property
    def relit_integrator(self):
        if self._relit_integrator is None:
            from .face_reattach import RelitOpticalFaceIntegrator
            self._relit_integrator = RelitOpticalFaceIntegrator(self.apple_vision)
        return self._relit_integrator

    @property
    def proportion_calibrator(self):
        if self._proportion_calibrator is None:
            from .proportion_calibrator import MicroProportionCalibrator
            self._proportion_calibrator = MicroProportionCalibrator(self.apple_vision)
        return self._proportion_calibrator

    @property
    def batch_composer(self):
        if self._batch_composer is None:
            from .multi_asset_composer import MultiSceneBatchComposer
            self._batch_composer = MultiSceneBatchComposer(self.turnaround)
        return self._batch_composer

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
        cohesion_mode: str = "off",
        reference_scene_path: Optional[Path] = None,
        radiometric_spec: Optional[SingleLensRadiometricSpec] = None,
        angle_bank: Optional[Dict[str, Path]] = None,
        calibrate_micro_proportions: bool = False,  # Default False: pure unwarped fresh Gen-0 single-pass (opt-in TPS only)
    ) -> Dict[str, object]:
        """Execute Single-Pass Pure From-Scratch 5K Pipeline with Post-Gen Optical Cohesion Mastering & 9-Gate Audit."""
        ref_bgr: Optional[np.ndarray] = None
        if reference_face_path is not None and Path(reference_face_path).exists():
            ref_bgr = cv2.imread(str(reference_face_path))

        ref_scene_bgr: Optional[np.ndarray] = None
        if reference_scene_path is not None and Path(reference_scene_path).exists():
            ref_scene_bgr = cv2.imread(str(reference_scene_path))

        with tempfile.TemporaryDirectory(prefix="single_pass_5k_") as tmp_dir:
            tmp = Path(tmp_dir)
            if getattr(self.config, "upscale_mode", "optical_lanczos") != "esrgan":
                raw_bgr = cv2.imread(str(pure_scratch_scene_path))
                if raw_bgr is None:
                    raise FileNotFoundError(f"Cannot read scene image: {pure_scratch_scene_path}")
                up_bgr = self.upscaler.upscale_bgr_in_memory(
                    raw_bgr, target_min_height=self.config.canvas.target_height
                )
            else:
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

            if reattach_mode == "relit_optical" and reference_face_path is not None and Path(reference_face_path).exists():
                src_face_p = Path(reference_face_path)
                chosen_lbl = "gen0_origin"
                turn_meta: Dict[str, float] = {}
                if angle_bank:
                    src_face_p, chosen_lbl, turn_meta = self.relit_integrator.select_angle_matched_highres_source(
                        canvas_5k, Path(reference_face_path), angle_bank
                    )
                chosen_bgr = cv2.imread(str(src_face_p))
                if chosen_bgr is not None:
                    if max(chosen_bgr.shape[:2]) < 1800 and self.upscaler.upscayl_bin is not None:
                        chosen_bgr, _ = self.upscaler.upscale_with_upscayl(chosen_bgr, style="photo", scale=4)
                    canvas_5k, ro_meta = self.relit_integrator.integrate_relit_face(chosen_bgr, canvas_5k)
                    ro_meta["chosenSourceLabel"] = chosen_lbl
                    ro_meta.update(turn_meta)
                    reattach_info["relit_optical"] = ro_meta
            elif calibrate_micro_proportions and reattach_mode == "none" and ref_bgr is not None:
                try:
                    canvas_5k, cal_meta = self.proportion_calibrator.calibrate_to_gen0(canvas_5k, ref_bgr)
                    reattach_info["micro_proportion_calibration"] = cal_meta
                except Exception:
                    pass

            face_5k = self.detector.detect(canvas_5k)
            crop_box_5k = self.detector.compute_upper_body_crop_box(
                canvas_5k.shape[:2], face_5k, aspect_ratio=self.config.canvas.crop_aspect_ratio
            )
            X1, Y1, X2, Y2 = crop_box_5k
            vx1, vy1 = max(0, X1 - 120), max(0, Y1 - 60)
            vx2, vy2 = min(canvas_5k.shape[1], X2 + 120), min(canvas_5k.shape[0], Y2 + 160)

            roi = canvas_5k[vy1:vy2, vx1:vx2].copy()
            zero_touch = bool(getattr(self.config, "zero_touch_4k_passthrough", True)) and reattach_mode == "none"
            skin_metrics: Dict[str, object] = {"zero_touch_4k_passthrough": zero_touch}
            if not zero_touch:
                roi_face = self.detector.detect(roi)
                min_org_grad, _ = self.auditor.measure_per_organ_sharpness(roi, roi_face)
                if reattach_mode == "none" and (
                    (
                        self.config.auto_restore_distant_face_iod_px > 0.0
                        and roi_face.inter_ocular_distance < self.config.auto_restore_distant_face_iod_px
                    )
                    or min_org_grad < self.config.min_organ_gradient_energy
                ):
                    roi, _ = self.upscaler.restore_face_focus_5pt(roi)

                # CRITICAL: NEVER pass studio `reference_skin_profile` here — keep scene lighting 100% intact!
                roi_matte, skin_metrics = self.skin_processor.process(
                    roi, seed=seed, reference_skin_profile=None
                )
                canvas_5k[vy1:vy2, vx1:vx2] = roi_matte

            # Post-Generation Single-Camera Optical & Radiometric Mastering (OFF by default in v1.0.4 Zero-Touch 4K!)
            cohesion_meta: Dict[str, object] = {"mode": cohesion_mode, "applied": False}
            if cohesion_mode not in ("off", "") and not (zero_touch and cohesion_mode == "auto"):
                canvas_5k, cohesion_meta = PostGenCohesionMasterer.master(
                    canvas_5k,
                    face_det=face_5k,
                    reference_scene_bgr=ref_scene_bgr,
                    target_spec=radiometric_spec,
                    mode=cohesion_mode,
                    seed=seed,
                )

            final_roi = canvas_5k[vy1:vy2, vx1:vx2].copy()
            report = self.auditor.audit(
                canvas_5k, crop_box_5k, seam_l1_diff=0.0, reference_face_bgr=ref_bgr
            )
            if not report.passed:
                raise RuntimeError(f"QualityGateAuditor rejected output: {report.message}")

            output_hero_path = Path(output_hero_path)
            output_hero_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_hero_path), canvas_5k, [int(cv2.IMWRITE_JPEG_QUALITY), 97])
            if output_closeup_path:
                cv2.imwrite(str(output_closeup_path), final_roi, [int(cv2.IMWRITE_JPEG_QUALITY), 97])

            return {
                "output_hero_path": str(output_hero_path),
                "crop_box_5k": crop_box_5k,
                "skin_metrics": skin_metrics,
                "cohesion_meta": cohesion_meta,
                "reattach_info": reattach_info,
                "quality_report": report,
            }

    def remaster_scene_cohesion(
        self,
        input_scene_path: Path,
        output_hero_path: Path,
        output_closeup_path: Optional[Path] = None,
        reference_scene_path: Optional[Path] = None,
        mode: str = "auto",
        seed: int = 42,
    ) -> Dict[str, object]:
        """Instant (<0.3s) zero-regeneration optical mood/tone/contrast/color remastering on any existing output."""
        src = cv2.imread(str(input_scene_path))
        if src is None:
            raise FileNotFoundError(f"Cannot read input scene: {input_scene_path}")
        ref_scene = (
            cv2.imread(str(reference_scene_path))
            if (reference_scene_path is not None and Path(reference_scene_path).exists())
            else None
        )
        face_det = None
        try:
            face_det = self.detector.detect(src)
        except Exception:
            pass

        mastered, meta = PostGenCohesionMasterer.master(
            src,
            face_det=face_det,
            reference_scene_bgr=ref_scene,
            mode=mode,
            seed=seed,
        )
        output_hero_path = Path(output_hero_path)
        output_hero_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_hero_path), mastered, [int(cv2.IMWRITE_JPEG_QUALITY), 97])

        if output_closeup_path and face_det is not None:
            crop_box = self.detector.compute_upper_body_crop_box(
                mastered.shape[:2], face_det, aspect_ratio=self.config.canvas.crop_aspect_ratio
            )
            X1, Y1, X2, Y2 = crop_box
            vx1, vy1 = max(0, X1 - 120), max(0, Y1 - 60)
            vx2, vy2 = min(mastered.shape[1], X2 + 120), min(mastered.shape[0], Y2 + 160)
            cv2.imwrite(str(output_closeup_path), mastered[vy1:vy2, vx1:vx2], [int(cv2.IMWRITE_JPEG_QUALITY), 97])

        return {"output_hero_path": str(output_hero_path), "cohesion_meta": meta}

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

    def run_tonematched_5k_synthesis(
        self,
        fullbody_scene_path: Path,
        refined_crop_path: Path,
        output_hero_path: Path,
        output_closeup_path: Optional[Path] = None,
        crop_box_base_override: Optional[Tuple[int, int, int, int]] = None,
        reference_face_path: Optional[Path] = None,
        seed: int = 42,
    ) -> Dict[str, object]:
        """Tone-Matched 3D In-Place Upper-Body 5K Synthesis (Zero 2D Oval Pixel Pasting, Zero CodeFormer Distortion).

        1. Upscales both `fullbody_scene_path` and `refined_crop_path` 4x via `realesrgan-ncnn-vulkan` (`4xNomos8kSC`).
        2. Builds the auto-balanced 5K (`5040x2160`) 21:9 canvas and integrates the full 3D refined upper-body crop
           via `integrate_tonematched_crop` (preserving continuous eyewear temples, 3D head-shoulder proportions, and collar).
        3. Applies scene-harmonized P88 specular glare suppression (`reference_skin_profile=None`) so natural scene lighting,
           shadow falloff, color temperature, and overall tone & manner remain 100% cohesive.
        4. Verifies all 9 automated Quality Gates including `SFace` 128-D Identity & `Apple Vision` 3D Face Proportions.
        """
        ref_bgr: Optional[np.ndarray] = None
        if reference_face_path is not None and Path(reference_face_path).exists():
            ref_bgr = cv2.imread(str(reference_face_path))

        with tempfile.TemporaryDirectory(prefix="tonematched_5k_") as tmp_dir:
            tmp = Path(tmp_dir)
            up_scene_path = self.upscaler.upscale_image_file(
                Path(fullbody_scene_path), tmp / "up_scene.jpg"
            )
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

            final_canvas_5k, shift_xy, seam_err, _ = self.compositor.integrate_tonematched_crop(
                canvas_5k, up_crop_bgr, crop_box_5k
            )

            X1, Y1, X2, Y2 = crop_box_5k
            vx1, vy1 = max(0, X1 - 120), max(0, Y1 - 60)
            vx2, vy2 = min(final_canvas_5k.shape[1], X2 + 120), min(
                final_canvas_5k.shape[0], Y2 + 160
            )
            roi = final_canvas_5k[vy1:vy2, vx1:vx2].copy()
            roi_matte, skin_metrics = self.skin_processor.process(
                roi, seed=seed, reference_skin_profile=None
            )
            final_canvas_5k[vy1:vy2, vx1:vx2] = roi_matte

            report: QualityReport = self.auditor.audit(
                final_canvas_5k, crop_box_5k, seam_err, reference_face_bgr=ref_bgr
            )
            if not report.passed:
                raise RuntimeError(f"QualityGateAuditor rejected output: {report.message}")

            output_hero_path = Path(output_hero_path)
            output_hero_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_hero_path), final_canvas_5k, [int(cv2.IMWRITE_JPEG_QUALITY), 97])
            if output_closeup_path:
                cv2.imwrite(
                    str(output_closeup_path), roi_matte, [int(cv2.IMWRITE_JPEG_QUALITY), 97]
                )

            return {
                "output_hero_path": str(output_hero_path),
                "crop_box_5k": crop_box_5k,
                "alignment_shift": shift_xy,
                "seam_border_l1_diff": seam_err,
                "skin_metrics": skin_metrics,
                "quality_report": report,
            }


# Backward-compatible alias
PortraitIdentityPipeline = ModelGenerationPipeline
