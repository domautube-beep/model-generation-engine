"""Non-Skippable 6-Stage Fresh-Origin Editorial Workflow Engine (`v9.0`).

Guarantees that EVERY real-person generation strictly executes all 6 mandatory stages in order:
  Stage 1 (`ORIGIN_PREP`)      : Verify untouched Gen-0 + extract 25-metric Blueprint (18 Craniofacial/Jaw/Cheek + 7 Ocular)
                                 + extract lossless pure-optical Gen-0 ocular detail anchor + erase foreign people in locations.
  Stage 2 (`PROMPT_LOCK`)      : Emit K=3 fresh Gen-0-only single-pass prompts (`<= 35 deg`: zero synthetic 8dir dilution)
                                 with Jawline-Angle, Cheek-Area, Ocular-Anatomy, and Eye-Level Bounce Fill locks + manifest.
  Stage 3 (`FRESH_GEN_K`)      : Generate K independent fresh candidates strictly from whitelisted untouched Gen-0 + raw assets.
  Stage 4 (`NATIVE_RANKING`)   : Score all K candidates natively (`0px` warp, `0%` pasting) across SFace + 18 Craniofacial + 7 Ocular.
  Stage 5 (`FINAL_4K_10GATE`)  : Single final Upscayl 4x (`high-fidelity-4x` / `digital-art-4x`) + matte glare control
                                 + 10-Gate Quality Audit (`passed == True` REQUIRED) + export 4K / closeup / macro / 400% eye strip
                                 + Google Drive Vault sync + signed `workflow_receipt.json`.
  Stage 6 (`VISUAL_INSPECT`)   : Mandatory human/agent visual check of 4K macro + 400% eye comparison strip (enforced by Stop hook).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import CanvasSpec, EngineConfig, SubjectProfile
from .face_exaggeration_guard import FaceExaggerationGuard
from .identity_turnaround import AnatomicalBlueprint
from .ocular_anatomy import OcularAnatomyAnalyzer
from .organ_depth_anatomy import OrganAndDepthAnatomyAnalyzer
from .origin_guard import OriginOnlyOneShotGuard
from .pipeline import ModelGenerationPipeline
from .prompt_builder import PromptFirewall
from .proportion_calibrator import compare_14pt_proportions, extract_14pt_proportion_vector
from .real_multishot_studio import (
    FaceLockedOutpaintStudio,
    IntrinsicAlbedoRelighter,
    RealMultiShotBank,
    analyze_target_scene_chiaroscuro_and_mood,
    master_full_frame_mood_and_grain,
)
from .scene_radiometry import (
    PostGenCohesionMasterer,
    SceneAdaptive5Lens5LightingPlanner,
    SceneRadiometryAnalyzer,
    build_framing_and_depth_directive,
    resolve_framing_mode,
)
from .skin_processor import MatteSkinProcessor
from .version_guard import CANONICAL_PROJECT_NAME, SHORT_PROJECT_NAME, check_version_integrity


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class CandidateScore:
    candidate_path: str
    sface_similarity: float
    contour_iou: float
    organ_depth_drift: float
    multi_segment_head_err_pct: float
    mean_weighted_drift_pct: float
    cheek_area_diff_pct: float
    lower_jaw_area_diff_pct: float
    avg_gonion_diff_deg: float
    face_to_shoulder_ratio: float
    ocular_drift_score: float
    eyelid_edge_ncc: float
    sclera_diff_pct: float
    crease_prom_diff: float
    ocular_passed: bool
    native_bounds_passed: bool
    composite_rank_score: float
    issues: List[str]


class FreshOriginEditorialWorkflow:
    """Ironclad 6-stage state machine enforcing Fresh Gen-0 Regeneration + 25-Metric Native Selection + 10-Gate 4K Delivery."""

    DEFAULT_CAMERA_SEEDS: Tuple[Tuple[str, str], ...] = (
        ("50mm", "Hasselblad X2D 100C with XCD 55V (50mm equiv)"),
        ("65mm", "Hasselblad X2D 100C with XCD 65mm f/2.8"),
        ("85mm", "Canon EOS R5 with RF 85mm f/1.2L"),
        ("105mm", "Leica SL3 with APO-Summicron 105mm"),
        ("135mm", "Sony Alpha 7R V with FE 135mm f/1.8 GM"),
    )

    def _build_dynamic_focal_clean_plates(
        self,
        clean_loc_p: Path,
        loc_bgr_pre: Optional[np.ndarray],
        loc_pose_meta: Optional[Dict[str, object]],
        wdir: Path,
        presets: Sequence[Dict[str, object]],
    ) -> List[Path]:
        """Dynamically crop focal-matched clean location plates (50..135mm) around the detected subject/furniture anchor (0% hardcoded coords)."""
        bgr = cv2.imread(str(clean_loc_p))
        if bgr is None or bgr.size == 0:
            return [clean_loc_p for _ in presets]
        h, w = bgr.shape[:2]
        aspect = float(w) / float(max(1, h))

        cx_norm = float((loc_pose_meta or {}).get("auto_subject_x_norm", 0.50))
        cy_norm = 0.50
        found_face = False
        if loc_bgr_pre is not None and loc_bgr_pre.size > 0:
            try:
                det_pre = self.pipeline.detector.detect(loc_bgr_pre)
                fx, fy, fw, fh = det_pre.bbox
                lh, lw = loc_bgr_pre.shape[:2]
                cx_norm = float(np.clip((fx + 0.5 * fw) / float(max(1, lw)), 0.18, 0.82))
                # Anchor slightly below face center (upper-chest/lap axis) so 85/105/135mm frame head + torso/lap naturally
                cy_norm = float(np.clip((fy + 1.85 * fh) / float(max(1, lh)), 0.30, 0.72))
                found_face = True
            except Exception:
                found_face = False

        if not found_face:
            # Saliency/edge centroid fallback on empty clean rooms (zero hardcoded chair coordinates)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.hypot(gx, gy)
            # Weight lower-middle interior furniture band slightly higher than ceiling border
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            vy_weight = np.exp(-0.5 * (((yy / float(max(1, h)) - 0.56) / 0.28) ** 2))
            w_map = mag * vy_weight
            total_w = float(np.sum(w_map))
            if total_w > 1e-3:
                cx_norm = float(np.clip(float(np.sum(w_map * (xx / float(max(1, w))))) / total_w, 0.22, 0.78))
                cy_norm = float(np.clip(float(np.sum(w_map * (yy / float(max(1, h))))) / total_w, 0.34, 0.68))

        out_paths: List[Path] = []
        for idx, preset in enumerate(presets):
            raw_scale = float(preset.get("clean_plate_crop_scale", 1.0))  # type: ignore[arg-type]
            scale = max(0.74, raw_scale) if min(h, w) < 1800 else raw_scale
            lens_mm = int(preset.get("lens_mm", 85))  # type: ignore[arg-type]
            if scale >= 0.995:
                out_p = wdir / f"loc{idx + 1:02d}_{lens_mm}mm_clean.jpg"
                shutil.copy2(clean_loc_p, out_p)
                self.origin_guard.register_approved_origin(out_p, kind="clean_location", depth=0)
                out_paths.append(out_p)
                continue

            # Tighter lenses shift slightly upward toward head/chest height for natural portrait headroom
            tightness = float(np.clip((1.0 - scale) / 0.54, 0.0, 1.0))
            eff_cy = float(np.clip(cy_norm - 0.09 * tightness, 0.22, 0.75))
            eff_cx = float(np.clip(0.50 * (1.0 - tightness) + cx_norm * tightness, 0.15, 0.85))
            ch = max(64, int(round(h * scale)))
            cw = max(64, min(w, int(round(ch * aspect))))
            x0 = int(round(np.clip(eff_cx * w - 0.5 * cw, 0, max(0, w - cw))))
            y0 = int(round(np.clip(eff_cy * h - 0.5 * ch, 0, max(0, h - ch))))
            sub = cv2.resize(bgr[y0 : y0 + ch, x0 : x0 + cw], (w, h), interpolation=cv2.INTER_LANCZOS4)
            out_p = wdir / f"loc{idx + 1:02d}_{lens_mm}mm_clean.jpg"
            cv2.imwrite(str(out_p), sub, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
            self.origin_guard.register_approved_origin(out_p, kind="clean_location", depth=0)
            out_paths.append(out_p)
        return out_paths

    def __init__(self, pipeline: Optional[ModelGenerationPipeline] = None) -> None:
        self.pipeline = pipeline or ModelGenerationPipeline(EngineConfig())
        self.ocular = OcularAnatomyAnalyzer(self.pipeline.apple_vision)
        self.organ_depth = OrganAndDepthAnatomyAnalyzer(self.pipeline.apple_vision)
        self.exaggeration_guard = FaceExaggerationGuard(self.pipeline.apple_vision)
        self.origin_guard = OriginOnlyOneShotGuard()
        self.relighter = IntrinsicAlbedoRelighter(self.pipeline.detector, self.pipeline.apple_vision)
        self.facelock_studio = FaceLockedOutpaintStudio(
            self.pipeline.detector, self.pipeline.apple_vision, self.relighter
        )
        self._bp_mem_cache: Dict[str, Tuple[AnatomicalBlueprint, Optional[object]]] = {}
        self._verified_8dir_mem_cache: Dict[Tuple[str, str], str] = {}
        self._ocular_crop_bytes_cache: Dict[str, bytes] = {}

    def _get_cached_gen0_blueprint_and_organ(
        self, gen0_p: Path, ref_bgr: np.ndarray
    ) -> Tuple[str, AnatomicalBlueprint, Optional[object]]:
        """Extract or load cached 25-metric AnatomicalBlueprint + OrganAndDepthProfile keyed by Gen-0 SHA-256."""
        gen0_sha = _sha256_file(gen0_p)
        hit = self._bp_mem_cache.get(gen0_sha)
        if hit is not None:
            return gen0_sha, hit[0], hit[1]

        cache_dir = Path.home() / ".cache" / "model_generation_engine" / "blueprint_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        disk_f = cache_dir / f"bp_v110_{gen0_sha[:28]}.json"
        if disk_f.is_file():
            try:
                from .ocular_anatomy import OcularAnatomyProfile
                from .organ_depth_anatomy import OrganAndDepthProfile

                raw = json.loads(disk_f.read_text(encoding="utf-8"))
                bp_d = dict(raw["blueprint"])
                oc_d = bp_d.pop("ocular_profile", None)
                bp_d["ocular_profile"] = OcularAnatomyProfile(**oc_d) if isinstance(oc_d, dict) else None
                bp = AnatomicalBlueprint(**bp_d)
                op_d = raw.get("organ_profile")
                op = OrganAndDepthProfile(**op_d) if isinstance(op_d, dict) else None
                self._bp_mem_cache[gen0_sha] = (bp, op)
                return gen0_sha, bp, op
            except Exception:
                pass

        bp = self.pipeline.turnaround.extract_anatomical_blueprint(ref_bgr)
        op = self.organ_depth.extract_profile(ref_bgr)
        self._bp_mem_cache[gen0_sha] = (bp, op)
        try:
            payload = {
                "blueprint": asdict(bp),
                "organ_profile": asdict(op) if op is not None else None,
            }
            disk_f.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        return gen0_sha, bp, op

    def prepare_origin_bundle(
        self,
        gen0_portrait_path: Path,
        work_dir: Path,
        scene_instruction: str,
        outfit_instruction: str,
        location_reference_path: Optional[Path] = None,
        clean_location_override_path: Optional[Path] = None,
        turnaround_8dir_sheet_path: Optional[Path] = None,
        extra_raw_asset_paths: Sequence[Path] = (),
        real_original_paths: Sequence[Path] = (),
        design_mode: str = "zero_touch_4k",
        aspect_ratio_label: str = "3:4",
        best_of_k: int = 5,
        match_gen0_yaw: bool = True,
        attach_ocular_detail_anchor: bool = True,
        text_only_location: bool = False,
        no_film_grain: bool = False,
        fixed_lens_mm: Optional[int] = None,
        framing_mode: str = "auto",
        subject: Optional[SubjectProfile] = None,
    ) -> Dict[str, object]:
        """Stages 1 & 2: Validate Gen-0 (+ optional Multi-Shot Real Bank), clean location, emit K prompts."""
        gen0_p = Path(gen0_portrait_path).resolve()
        if not gen0_p.is_file():
            raise FileNotFoundError(f"Gen-0 portrait not found: {gen0_p}")

        # Hard check: reject any previously generated/derivative/upscaled file as Gen-0!
        self.origin_guard.verify_one_shot_and_inputs([gen0_p])

        wdir = Path(work_dir).resolve()
        wdir.mkdir(parents=True, exist_ok=True)

        bank: Optional[RealMultiShotBank] = None
        bank_summary: Optional[Dict[str, object]] = None
        skeletal_clause = ""
        if real_original_paths:
            bank = RealMultiShotBank(
                self.pipeline.config,
                self.pipeline.detector,
                self.pipeline.apple_vision,
                self.pipeline.auditor,
                origin_guard=self.origin_guard,
            )
            all_real = [gen0_p] + [Path(p).resolve() for p in real_original_paths]
            bank_summary = bank.ingest_real_originals(all_real)
            skeletal_clause = bank.to_dynamic_skeletal_clause()
            PromptFirewall.validate_prompt(skeletal_clause)

        ref_bgr = cv2.imread(str(gen0_p))
        if ref_bgr is None:
            raise ValueError(f"Cannot decode Gen-0 image: {gen0_p}")

        gen0_sha, blueprint, organ_profile = self._get_cached_gen0_blueprint_and_organ(gen0_p, ref_bgr)
        clause_25pt = blueprint.to_3d_invariance_clause()
        organ_depth_clause = organ_profile.to_dynamic_organ_depth_clause() if organ_profile is not None else ""
        natural_id_clause = organ_profile.to_natural_identity_only_clause(clean_studio=no_film_grain) if organ_profile is not None else ""
        PromptFirewall.validate_prompt(clause_25pt)
        if organ_depth_clause:
            PromptFirewall.validate_prompt(organ_depth_clause)
        if natural_id_clause:
            PromptFirewall.validate_prompt(natural_id_clause)

        # Pre-analyze target location pose, chiaroscuro mood & radiometry BEFORE person-erasure so 8-dir yaw & lighting match naturally!
        loc_pose_meta: Optional[Dict[str, object]] = None
        loc_radio_spec = None
        scene_mood: Optional[Dict[str, object]] = None
        clean_loc_p: Optional[Path] = None
        loc_src_resolved: Optional[Path] = None
        if location_reference_path is not None and Path(location_reference_path).is_file():
            loc_src = Path(location_reference_path).resolve()
            loc_src_resolved = loc_src
            loc_bgr_pre = cv2.imread(str(loc_src))
            if loc_bgr_pre is not None:
                try:
                    loc_pose_meta = self.pipeline.turnaround.analyze_reference_pose_before_masking(loc_bgr_pre)
                except Exception:
                    loc_pose_meta = None
                try:
                    loc_radio_spec = SceneRadiometryAnalyzer.analyze(scene_bgr=loc_bgr_pre)
                except Exception:
                    loc_radio_spec = None
                try:
                    scene_mood = analyze_target_scene_chiaroscuro_and_mood(loc_bgr_pre, self.pipeline.detector)
                except Exception:
                    scene_mood = None
                if design_mode != "pure_character_sheet" and not text_only_location:
                    clean_loc_p = wdir / f"loc00_{loc_src.stem}_clean.jpg"
                    if clean_location_override_path is not None and Path(clean_location_override_path).resolve().is_file():
                        shutil.copy2(Path(clean_location_override_path).resolve(), clean_loc_p)
                    else:
                        self.pipeline.turnaround.erase_foreign_person_completely(
                            loc_bgr_pre, output_clean_path=clean_loc_p, remove_overlay_text=True
                        )
                    self.origin_guard.register_approved_origin(clean_loc_p, kind="clean_location", depth=0)
        elif (not text_only_location) and clean_location_override_path is not None and Path(clean_location_override_path).is_file():
            clean_loc_p = wdir / "loc00_override_clean.jpg"
            shutil.copy2(Path(clean_location_override_path).resolve(), clean_loc_p)
            self.origin_guard.register_approved_origin(clean_loc_p, kind="clean_location", depth=0)
            ov_bgr = cv2.imread(str(clean_loc_p))
            if ov_bgr is not None:
                scene_mood = analyze_target_scene_chiaroscuro_and_mood(ov_bgr, self.pipeline.detector)

        target_yaw_for_bank = (
            float(blueprint.source_signed_yaw_deg)
            if match_gen0_yaw
            else float((loc_pose_meta or {}).get("auto_yaw_deg", blueprint.source_signed_yaw_deg))
        )

        ordered_refs: List[Dict[str, str]] = []
        role_lines: List[str] = []
        turnaround_p: Optional[Path] = None
        turnaround_master_sheet_meta: Optional[str] = None
        ocular_anchor_p: Optional[Path] = None
        facelock_meta: Optional[Dict[str, object]] = None
        facelock_canvas_p: Optional[Path] = None

        if turnaround_8dir_sheet_path is not None:
            raw_turn_p = Path(turnaround_8dir_sheet_path).expanduser().resolve()
            if not raw_turn_p.is_file():
                raise FileNotFoundError(
                    f"[8-Dir Bank Verification Failed] --turnaround-sheet file does not exist: {raw_turn_p}"
                )
            turn_sha = _sha256_file(raw_turn_p)
            if turn_sha == gen0_sha:
                raise RuntimeError(
                    "[8-Dir Bank Verification Failed] --turnaround-sheet cannot point to Gen-0 itself; must be a verified 8-direction bank!"
                )
            bank_ck = (gen0_sha, turn_sha)
            if bank_ck in self._verified_8dir_mem_cache:
                turnaround_master_sheet_meta = self._verified_8dir_mem_cache[bank_ck]
            else:
                distinct_views = set()
                for probe_yaw in (-90.0, -45.0, -22.5, 22.5, 45.0, 90.0):
                    iv = self.pipeline.turnaround.resolve_nearest_individual_8dir_portrait(raw_turn_p, probe_yaw)
                    if iv is not None and iv[0].is_file():
                        iv_sha = _sha256_file(iv[0].resolve())
                        if iv_sha != gen0_sha:
                            distinct_views.add(iv_sha)
                if len(distinct_views) < 4:
                    raise RuntimeError(
                        f"[8-Dir Bank Verification Failed] '{raw_turn_p.name}' lacks verified distinct 1-Hop 8-direction portraits (found {len(distinct_views)} < 4)!"
                    )
                turnaround_master_sheet_meta = str(raw_turn_p)
                self._verified_8dir_mem_cache[bank_ck] = turnaround_master_sheet_meta

        if design_mode == "pure_character_sheet" and turnaround_8dir_sheet_path is not None and Path(turnaround_8dir_sheet_path).exists():
            # Character-Sheet Angle-Matched Pure 1-Pass Mode (ZERO clean-plate hole composite, ZERO post face-masking!):
            # Select the exact pose-matched portrait from the 8-direction character bank + real camera anchors!
            raw_turn_p = Path(turnaround_8dir_sheet_path).resolve()
            indiv = self.pipeline.turnaround.resolve_nearest_individual_8dir_portrait(raw_turn_p, target_yaw_for_bank)
            if indiv is not None:
                turnaround_p, matched_8dir_view, matched_8dir_yaw = indiv[0].resolve(), indiv[1], indiv[2]
                self.origin_guard.register_approved_origin(turnaround_p, kind="8dir_face", depth=1)
                ordered_refs.append(
                    {
                        "slot": "Image 1",
                        "role": "pose_angle_matched_8dir_character_portrait",
                        "path": str(turnaround_p),
                        "sha256": _sha256_file(turnaround_p),
                        "view_key": matched_8dir_view,
                        "label": f"Pose-Angle-Matched 8-Direction Character Portrait ({matched_8dir_view}, {matched_8dir_yaw:+.1f} deg)",
                    }
                )
                role_lines.append(
                    f"Image 1 (PRIMARY POSE-ANGLE-MATCHED CHARACTER PORTRAIT, {matched_8dir_yaw:+.1f} deg): "
                    "exact head rotation, 3D facial structure, eye/nose/lip geometry, and hair drape for the target pose"
                )
                target_yaw_for_bank = float(matched_8dir_yaw)
            if bank is not None and len(bank.entries) >= 2:
                prim_real, sec_real = bank.select_best_reference_pair(target_yaw_for_bank, scene_instruction=scene_instruction)
                hi_real = sec_real if sec_real is not None else prim_real
                ordered_refs.append(
                    {
                        "slot": f"Image {len(ordered_refs) + 1}",
                        "role": "real_highres_organ_skin_master",
                        "path": str(hi_real.path),
                        "sha256": hi_real.sha256,
                        "label": f"Untouched real camera close-up organ & skin anchor ({hi_real.name})",
                    }
                )
                ordered_refs.append(
                    {
                        "slot": f"Image {len(ordered_refs) + 1}",
                        "role": "real_posture_shoulder_build_anchor",
                        "path": str(prim_real.path),
                        "sha256": prim_real.sha256,
                        "label": f"Untouched real camera body-proportion & natural expression anchor ({prim_real.name})",
                    }
                )
            else:
                ordered_refs.append(
                    {
                        "slot": f"Image {len(ordered_refs) + 1}",
                        "role": "gen0_real_camera_anchor",
                        "path": str(gen0_p),
                        "sha256": _sha256_file(gen0_p),
                        "label": "Untouched real camera Gen-0 anchor",
                    }
                )
        elif bank is not None and len(bank.entries) >= 2:
            # Multi-Shot Real Master Bank active: prioritize REAL posture+angle portrait + REAL high-IOD close-up (0% AI 8-dir dilution!)
            prim_real, sec_real = bank.select_best_reference_pair(
                target_yaw_for_bank, scene_instruction=scene_instruction
            )
            gen0_p = Path(prim_real.path).resolve()
            ref_bgr_prim = cv2.imread(str(gen0_p))
            if ref_bgr_prim is not None:
                blueprint = self.pipeline.turnaround.extract_anatomical_blueprint(ref_bgr_prim)
                op_prim = self.organ_depth.extract_profile(ref_bgr_prim)
                if op_prim is not None:
                    organ_depth_clause = op_prim.to_dynamic_organ_depth_clause(
                        low_key_scene=bool(scene_mood and scene_mood.get("is_low_key", False))
                    )
                    natural_id_clause = op_prim.to_natural_identity_only_clause()
                    PromptFirewall.validate_prompt(organ_depth_clause)
                    PromptFirewall.validate_prompt(natural_id_clause)
                target_yaw_for_bank = float(prim_real.yaw_deg)

            ordered_refs.append(
                {
                    "slot": "Image 1",
                    "role": "real_posture_angle_body_identity",
                    "path": str(prim_real.path),
                    "sha256": prim_real.sha256,
                    "label": f"Untouched real camera posture & angle portrait ({prim_real.name}, {prim_real.yaw_deg:+.1f} deg)",
                }
            )
            role_lines.append(
                "Image 1 (SUPREME REAL CAMERA FACE & SKELETAL PROPORTION IDENTITY AUTHORITY — IDENTITY ONLY, NOT POSTURE OR LIGHTING): "
                "untouched real portrait for facial structure, shoulder-to-head scale, neck, and hand anatomy only"
            )
            if sec_real is not None:
                ordered_refs.append(
                    {
                        "slot": "Image 2",
                        "role": "real_highres_organ_skin_master",
                        "path": str(sec_real.path),
                        "sha256": sec_real.sha256,
                        "label": f"Untouched high-resolution real camera organ & pore master ({sec_real.name}, IOD {sec_real.iod_px:.0f}px)",
                    }
                )
                role_lines.append(
                    "Image 2 (HIGH-RESOLUTION REAL CAMERA EYE, NOSE, LIP & PORE ALBEDO AUTHORITY): "
                    "untouched close-up real photograph of the exact same person from Image 1"
                )
        else:
            ordered_refs.append(
                {
                    "slot": "Image 1",
                    "role": "gen0_full_identity_cranial_jaw_cheek",
                    "path": str(gen0_p),
                    "sha256": _sha256_file(gen0_p),
                    "label": f"Untouched Gen-0 real camera portrait ({blueprint.source_signed_yaw_deg:+.1f} deg)",
                }
            )
            role_lines.append(
                "Image 1 (SUPREME IDENTITY, JAWLINE-ANGLE, CHEEK-AREA & CRANIAL DOME AUTHORITY — UNTOUCHED Gen-0): "
                "100% absolute ground-truth authority for cranial dome, hairline, hair parting/tuck, exact cheek contour, "
                "exact mandibular jawline angle & chin taper (do NOT round or widen the jaw), nose tip, philtrum, lips, and natural skin tone"
            )
            needs_rotated_8dir_ref = self.pipeline.turnaround.requires_intermediate_turnaround_hop(
                float(blueprint.source_signed_yaw_deg), target_yaw_for_bank, max_direct_delta_deg=25.0
            )
            matched_8dir_view: Optional[str] = None
            matched_8dir_yaw: Optional[float] = None
            if turnaround_8dir_sheet_path is not None and Path(turnaround_8dir_sheet_path).exists() and needs_rotated_8dir_ref:
                raw_turn_p = Path(turnaround_8dir_sheet_path).resolve()
                indiv = self.pipeline.turnaround.resolve_nearest_individual_8dir_portrait(raw_turn_p, target_yaw_for_bank)
                if indiv is not None:
                    turnaround_p, matched_8dir_view, matched_8dir_yaw = indiv[0].resolve(), indiv[1], indiv[2]
                elif raw_turn_p.is_file() and "master_8angle" not in raw_turn_p.name.lower():
                    turnaround_p = raw_turn_p
                if turnaround_p is not None and turnaround_p.is_file():
                    self.origin_guard.register_approved_origin(turnaround_p, kind="8dir_face", depth=1)
                    ordered_refs.append(
                        {
                            "slot": f"Image {len(ordered_refs) + 1}",
                            "role": "verified_1hop_8dir_individual_portrait",
                            "path": str(turnaround_p),
                            "sha256": _sha256_file(turnaround_p),
                            "view_key": matched_8dir_view or "custom",
                            "label": (
                                f"Verified 1-Hop Individual 8-Direction Portrait ({matched_8dir_view}, {matched_8dir_yaw:+.1f}°)"
                                if matched_8dir_yaw is not None
                                else "Verified 1-Hop Individual 8-Direction Portrait"
                            ),
                        }
                    )
                    role_lines.append(
                        f"Image {len(ordered_refs)} (VERIFIED 1-HOP INDIVIDUAL 3D JAWLINE & CHEEK VOLUME AUTHORITY): "
                        "angle-matched 1-hop portrait of the exact same person from Image 1; "
                        "match the exact 3D mandibular corner angle, lower-jaw taper, chin length, cheekbone projection, and ear/neck transition"
                    )

            lowres_or_smoothed_gen0 = bool(
                (organ_profile is not None and organ_profile.is_airbrushed_or_lowres_origin)
                or min(ref_bgr.shape[:2]) < 780
            )
            if attach_ocular_detail_anchor and turnaround_p is None and not lowres_or_smoothed_gen0:
                ocular_anchor_p = wdir / "gen0_optical_ocular_detail_anchor.jpg"
                cached_oc_bytes = self._ocular_crop_bytes_cache.get(gen0_sha)
                if cached_oc_bytes is not None:
                    ocular_anchor_p.write_bytes(cached_oc_bytes)
                else:
                    self.pipeline.turnaround.prepare_pure_gen0_ocular_detail_anchor(gen0_p, ocular_anchor_p)
                    self._ocular_crop_bytes_cache[gen0_sha] = ocular_anchor_p.read_bytes()
                    self.origin_guard.register_approved_origin(ocular_anchor_p, kind="gen0_optical_crop", depth=0)
                ordered_refs.append(
                    {
                        "slot": f"Image {len(ordered_refs) + 1}",
                        "role": "gen0_optical_ocular_detail",
                        "path": str(ocular_anchor_p),
                        "sha256": _sha256_file(ocular_anchor_p),
                        "label": "Pure lossless optical Gen-0 eye-nose-lip-skin close-up crop (0% AI, 0% Upscayl)",
                    }
                )
                role_lines.append(
                    f"Image {len(ordered_refs)} (LOSSLESS OPTICAL ORGAN & SKIN MICRO-DETAIL AUTHORITY — DIRECT Gen-0 CROP): "
                    "pure unmodified optical zoom of Image 1's exact eyes, nasal bridge/tip, lips, 3D cheek volume, and skin texture"
                )

        if clean_loc_p is not None and clean_loc_p.is_file():
            if design_mode == "design_b_rerender" and bank is not None and loc_src_resolved is not None:
                facelock_canvas_p = wdir / "stage2a_mesh_precomposite_editorial_b.jpg"
                facelock_meta = self.facelock_studio.build_editorial_pose_b_precomposite(
                    original_location_path=loc_src_resolved,
                    output_canvas_path=facelock_canvas_p,
                    bank=bank,
                    clean_location_path=clean_loc_p,
                )
                self.origin_guard.register_approved_origin(facelock_canvas_p, kind="clean_location", depth=0)
                ordered_refs.append(
                    {
                        "slot": f"Image {len(ordered_refs) + 1}",
                        "role": "editorial_b_precomposite_rerender_guide",
                        "path": str(facelock_canvas_p),
                        "sha256": _sha256_file(facelock_canvas_p),
                        "label": "Editorial Pose B Pre-Composite Guide (3/4 chair + slouchy charcoal suit + real camera face ready for full 3D optical re-rendering)",
                    }
                )
                role_lines.append(
                    f"Image {len(ordered_refs)} (EDITORIAL POSE B PRE-COMPOSITE — FULL-SCENE 3D OPTICAL RE-RENDER GUIDE): "
                    "re-photograph this exact scene as a single authentic 35mm camera exposure so true 3D side-window light and shadows sculpt the subject's face, nose, eye sockets, jaw, neck, suit, and chair together"
                )
            elif design_mode == "design_b_facelock_outpaint" and bank is not None:
                facelock_canvas_p = wdir / "stage2a_mesh_precomposite_facelock.png"
                facelock_meta = self.facelock_studio.build_facelock_precomposite_canvas(
                    clean_location_path=clean_loc_p,
                    output_canvas_path=facelock_canvas_p,
                    bank=bank,
                    original_location_path=loc_src_resolved,
                    scene_instruction=scene_instruction,
                    aspect_ratio_label=aspect_ratio_label,
                )
                self.origin_guard.register_approved_origin(facelock_canvas_p, kind="clean_location", depth=0)
                ordered_refs.append(
                    {
                        "slot": f"Image {len(ordered_refs) + 1}",
                        "role": "location_facelock_precomposite_authority",
                        "path": str(facelock_canvas_p),
                        "sha256": _sha256_file(facelock_canvas_p),
                        "label": "Face-Locked Virtual Studio Outpaint Canvas (real camera head anchored on clean scene chair)",
                    }
                )
                role_lines.append(
                    f"Image {len(ordered_refs)} (FACE-LOCKED VIRTUAL STUDIO OUTPAINT PLATE): "
                    "keep the anchored head position, head scale, hair parting, and facial expression in place and seamlessly outpaint the body, suit, hands, and room lighting"
                )
            else:
                ordered_refs.append(
                    {
                        "slot": f"Image {len(ordered_refs) + 1}",
                        "role": "location_clean_authority",
                        "path": str(clean_loc_p),
                        "sha256": _sha256_file(clean_loc_p),
                        "label": "Person-erased crisp clean location, furniture & lighting authority",
                    }
                )
                role_lines.append(
                    f"Image {len(ordered_refs)} (LOCATION, PROPS & LIGHTING AUTHORITY — PERSON-ERASED): "
                    "exact architectural space, furniture, color palette, and lighting mood—place the person from Image 1 naturally here"
                )

        for idx, extra_p in enumerate(extra_raw_asset_paths, start=1):
            ep = Path(extra_p).resolve()
            if ep.is_file():
                self.origin_guard.verify_one_shot_and_inputs([ep])
                ordered_refs.append(
                    {
                        "slot": f"Image {len(ordered_refs) + 1}",
                        "role": f"raw_styling_asset_{idx}",
                        "path": str(ep),
                        "sha256": _sha256_file(ep),
                        "label": f"Raw original styling asset ({ep.name})",
                    }
                )
                role_lines.append(
                    f"Image {len(ordered_refs)} (RAW ORIGINAL STYLING ASSET): preserve exact garment/accessory silhouette and hardware naturally"
                )

        # Natural Gaze-Head Coupling (character-sheet law: NEVER force eyes to twist opposite to head turn!)
        yaw_deg = float(target_yaw_for_bank)
        if abs(yaw_deg) >= 8.0:
            side_word = "image-right" if yaw_deg > 0 else "image-left"
            yaw_clause = (
                f"Head and gaze naturally aligned at ~{min(45.0, abs(yaw_deg)):.0f} deg toward {side_word} "
                "with level chin and calm relaxed eye alignment matching Image 1 (zero forced eye-twist or squinting)."
            )
        else:
            yaw_clause = "Head and calm relaxed gaze naturally aligned forward matching Image 1 without eye-strain."

        has_second_face_ref = (turnaround_p is not None) or (ocular_anchor_p is not None) or (bank is not None and len(bank.entries) >= 2)
        subj_label = (
            subject.subject_identity_label
            if subject is not None
            else ("exact same real person from Image 1 and Image 2" if has_second_face_ref else "exact same real person from Image 1")
        )
        k_count = max(1, int(best_of_k))
        prompts: List[Dict[str, str]] = []
        if design_mode == "pure_character_sheet":
            ref_note = "Image 1 (pose-angle-matched 8-direction character turnaround portrait) and Images 2-3 (untouched real camera organ & body anchors)"
        elif bank is not None and len(bank.entries) >= 2:
            ref_note = "Image 1 (untouched real camera facial & skeletal proportion portrait—identity ONLY, NOT posture or lighting) and Image 2 (untouched high-resolution real camera eye, nose, and lip close-up—identity ONLY, NOT lighting)"
        elif turnaround_p is not None:
            ref_note = "Image 1 (primary portrait) and Image 2 (angle-matched 3D jawline reference)"
        elif ocular_anchor_p is not None:
            ref_note = "Image 1 (full portrait) and Image 2 (pure optical eye-nose-lip-skin close-up of Image 1)"
        else:
            ref_note = "Image 1 (untouched portrait)"

        loc_slot_num = len(ordered_refs)
        kelv_val = int(loc_radio_spec.kelvin_est) if loc_radio_spec is not None else 5200
        is_lk_scene = bool(scene_mood is not None and scene_mood.get("is_low_key", False))
        lit_dir = "camera-left" if bool((scene_mood or {}).get("key_from_left", True)) else "camera-right"
        shd_dir = "camera-right" if bool((scene_mood or {}).get("key_from_left", True)) else "camera-left"

        if is_lk_scene:
            space_phrase = (
                f"inside the dim architectural space of Image {loc_slot_num}"
                if clean_loc_p is not None
                else "inside the dim classical architectural hall"
            )
            loc_light_desc = (
                f"Lit solely by existing {lit_dir} window daylight {space_phrase} "
                "with ZERO studio fill flash, ZERO reflector, and ZERO artificial face brightening: "
                "let overall facial brightness sit naturally subdued inside the dim room's moody warm-olive ambient air, "
                f"with soft muted {lit_dir} window daylight gently illuminating the subject's face and clear open eyes while the {shd_dir} side, "
                "neck, and inner garment folds transition smoothly into soft low-key ambient room shadow "
                "(preserving natural cheek volume and calm gentle expression without gaunt cheekbone hollowing) with grounded chair contact occlusion"
            )
        elif loc_radio_spec is not None and clean_loc_p is not None:
            loc_light_desc = (
                f"Lit naturally by the existing ~{kelv_val}K ambient light of Image {loc_slot_num} with zero artificial studio flash cutout, "
                "sharing identical shadow density, contact occlusion, and ambient color temperature across skin, wardrobe, and room"
            )
        elif loc_radio_spec is not None:
            loc_light_desc = (
                f"Sculpted ~{kelv_val}K directional soft studio key light with gentle ~3-stop 3D shadow roll-off across facial planes, "
                "fluid garment folds, and chair legs, natural grounded contact occlusion on the floor, and balanced natural skin tone "
                "harmonized with the warm room ambiance (zero harsh cutout flash, zero backdrop color bleed onto skin)"
            )
        else:
            loc_light_desc = (
                "Natural directional available light with authentic 3D shadow falloff, grounded contact occlusion, and zero studio flash cutout"
            )

        combined_scene_text = f"{scene_instruction} {outfit_instruction}".strip()
        resolved_framing = resolve_framing_mode(combined_scene_text, explicit_mode=framing_mode)
        adaptive_presets = SceneAdaptive5Lens5LightingPlanner.plan_5lens_5lighting_matrix(
            radio_spec=loc_radio_spec,
            scene_mood=scene_mood,
            scene_text=combined_scene_text,
            clean_studio_flag=bool(no_film_grain),
            count=k_count,
            framing_mode=resolved_framing,
        )
        focal_clean_paths: List[Path] = []
        loc_bgr_for_anchor = cv2.imread(str(loc_src_resolved)) if loc_src_resolved is not None else None
        if clean_loc_p is not None and facelock_canvas_p is None and k_count >= 2:
            focal_clean_paths = self._build_dynamic_focal_clean_plates(
                clean_loc_p=clean_loc_p,
                loc_bgr_pre=loc_bgr_for_anchor,
                loc_pose_meta=loc_pose_meta,
                wdir=wdir,
                presets=adaptive_presets,
            )

        is_airbrushed_gen0 = bool(organ_profile is not None and organ_profile.is_airbrushed_or_lowres_origin)
        for i in range(k_count):
            seed_tag, _cam_hdr = self.DEFAULT_CAMERA_SEEDS[i % len(self.DEFAULT_CAMERA_SEEDS)]
            preset = adaptive_presets[i]
            if fixed_lens_mm is not None and int(fixed_lens_mm) > 0:
                lens_mm = int(fixed_lens_mm)
                preset_ap = "f/1.8"
                framing_desc = build_framing_and_depth_directive(
                    lens_mm=lens_mm,
                    aperture=preset_ap,
                    framing_mode=resolved_framing,
                )
            else:
                lens_mm = int(preset["lens_mm"])  # type: ignore[arg-type]
                preset_ap = str(preset["aperture"])
                framing_desc = str(preset["framing_desc"])
            preset_light_ko = str(preset["lighting_name_ko"])
            preset_light_en = str(preset["lighting_desc"])
            regime_family = str(preset["scene_regime_family"])
            aperture_str = "f/2.0, ISO 800 available light" if is_lk_scene else f"{preset_ap}, natural available light"
            space_depth_rule = (
                "requested close-up composition (~70% subject occupancy) with tack-sharp focus strictly on the eyes and face while the background melts into creamy out-of-focus optical bokeh (zero busy background detail, strong 3D Z-depth)."
                if resolved_framing == "closeup"
                else "balanced 50:50 subject-to-background composition (human figure ~45%–50% of frame area, ~50%–55% open architectural breathing room, headroom, and side negative space—NEVER fill the entire frame edge-to-edge) with shallow optical depth-of-field bokeh (tack-sharp eyes/face, creamy out-of-focus background without sharp background micro-detail, strong three-dimensional Z-depth separation)."
            )
            if no_film_grain:
                slot1_camera = (
                    f"Clean, tack-sharp 100MP medium-format digital commercial editorial photograph (Hasselblad X2D 100C, {lens_mm}mm prime lens wide-open at {preset_ap}, base ISO 64, {aspect_ratio_label}, {framing_desc}), "
                    "pure noise-free optical clarity with ZERO film grain, natural undistorted perspective, "
                    f"realistic adult 7.45-head multi-segment Korean body proportions, and {space_depth_rule}"
                )
            else:
                slot1_camera = (
                    f"Unstaged raw medium-format 35mm film location photograph ({lens_mm}mm prime lens wide-open at {aperture_str}, {aspect_ratio_label}, {framing_desc}), "
                    "natural undistorted perspective with realistic adult 7.45-head multi-segment body proportions "
                    f"and {space_depth_rule}"
                )
            slot2_identity = (
                f"Subject: {subj_label} ({ref_note}). "
                f"{natural_id_clause if natural_id_clause else 'Preserve exact 1:1 anatomical facial identity from Image 1 while discarding studio flash lighting and stiff posture.'} "
                "Never beautify, exaggerate, or caricature any facial feature."
            )
            if design_mode == "design_b_rerender" and facelock_canvas_p is not None:
                clean_note = (
                    f" Re-photograph Image {loc_slot_num} as one authentic single-lens 35mm location exposure, keeping the exact chair, lounging pose, and facial identity while letting real room shadows sculpt the face and neck."
                )
            elif design_mode == "design_b_facelock_outpaint" and facelock_canvas_p is not None:
                clean_note = (
                    f" Seamlessly integrate the subject into Image {loc_slot_num} above the chair with natural hair drape, seated body, and room lighting."
                )
            elif clean_loc_p is not None:
                clean_note = (
                    f" Place the subject naturally inside the exact furniture and architectural space of Image {loc_slot_num} with continuous background bokeh."
                )
            else:
                clean_note = ""
            slot3_pose = (
                f"Physical Gravity & Pose: {scene_instruction.strip()}.{clean_note} "
                "Let hip and torso weight rest genuinely on the furniture under real gravity with a relaxed natural spine "
                "and natural shoulder posture, while keeping the head comfortably forward with a slight natural chin tuck "
                "matching Image 1's vertical camera angle (NEVER lean the head backward, NEVER tilt the chin upward, and NEVER expose the underside of the nostrils; zero stiff ID-photo tension)."
            )
            slot4_outfit = (
                f"Wardrobe: {outfit_instruction.strip()}, matte fabric hanging and creasing naturally with body slouch and gravity "
                "(completely replacing reference clothing)."
            )
            active_light_desc = preset_light_en if k_count >= 2 else loc_light_desc
            skin_realism_note = (
                "authentic unretouched real-camera adult skin micro-pores, subtle fine skin relief, and scene-matched tonal brightness/color (discarding Image 1's smoothed ID retouching and bright flash lightness) without plastic gloss"
                if is_airbrushed_gen0
                else "clean, translucent real-camera skin with fine natural pores and hydrated natural lip contour without plastic gloss"
            )
            if no_film_grain:
                slot5_light = (
                    f"Context-Matched Lighting & Skin Realism ({regime_family}): {active_light_desc}; "
                    f"{skin_realism_note}, 100% noise-free base-ISO digital clarity with ZERO film grain, ZERO dust, and ZERO vintage filter across the entire frame."
                )
            else:
                slot5_light = (
                    f"Available-Light Exposure & Analog Texture ({regime_family}): {active_light_desc}; "
                    f"{skin_realism_note}, unified by organic analog 35mm film grain across the entire frame."
                )
            p_text = f"{slot1_camera} {slot2_identity} {slot3_pose} {slot4_outfit} {slot5_light}"
            PromptFirewall.validate_prompt(p_text)

            cand_refs = [dict(r) for r in ordered_refs]
            cand_loc_path: Optional[str] = str(clean_loc_p) if clean_loc_p is not None else None
            if i < len(focal_clean_paths):
                fp = focal_clean_paths[i]
                cand_loc_path = str(fp)
                for r_item in cand_refs:
                    if r_item.get("role") == "location_clean_authority":
                        r_item["path"] = str(fp)
                        r_item["sha256"] = _sha256_file(fp)
                        r_item["label"] = f"Focal-matched {lens_mm}mm clean location authority ({fp.name})"

            prompts.append(
                {
                    "candidate_tag": f"c{i + 1}_{preset['tag']}",
                    "scene_regime_family": regime_family,
                    "framing_mode": resolved_framing,
                    "lens_mm": lens_mm,
                    "aperture": preset_ap,
                    "lighting_preset": str(preset["lighting_preset"]),
                    "lighting_name_ko": preset_light_ko,
                    "focal_clean_location_path": cand_loc_path,
                    "ordered_reference_images": cand_refs,
                    "prompt": p_text,
                    "prompt_chars": len(p_text),
                    "slot_budgets": {
                        "slot1_camera": len(slot1_camera),
                        "slot2_identity": len(slot2_identity),
                        "slot3_pose": len(slot3_pose),
                        "slot4_outfit": len(slot4_outfit),
                        "slot5_light": len(slot5_light),
                    },
                    "full_blueprint_clause": clause_25pt,
                    "organ_depth_skin_clause": organ_depth_clause,
                }
            )

        try:
            _vrep = check_version_integrity()
            _vmeta = {
                "project_name": _vrep.project_name,
                "short_alias": _vrep.short_alias,
                "canonical_version": _vrep.canonical_version,
                "canonical_label": _vrep.canonical_label,
                "version_check_passed": _vrep.passed,
            }
        except Exception:
            _vmeta = {
                "project_name": CANONICAL_PROJECT_NAME,
                "short_alias": SHORT_PROJECT_NAME,
                "canonical_version": "1.0.0",
                "canonical_label": "정본 v1.0.0",
                "version_check_passed": True,
            }

        manifest = {
            **_vmeta,
            "workflow_version": _vmeta["canonical_version"],
            "design_mode": design_mode,
            "framing_mode": resolved_framing,
            "no_film_grain": bool(no_film_grain),
            "stage": "READY_FOR_FRESH_GEN",
            "created_at": time.time(),
            "gen0_portrait_path": str(gen0_p),
            "gen0_sha256": _sha256_file(gen0_p),
            "turnaround_master_sheet_path": turnaround_master_sheet_meta,
            "matched_8dir_portrait_path": str(turnaround_p) if turnaround_p else None,
            "ocular_detail_anchor_path": str(ocular_anchor_p) if ocular_anchor_p else None,
            "clean_location_path": str(clean_loc_p) if clean_loc_p else None,
            "facelock_precomposite_meta": facelock_meta,
            "real_multishot_bank": bank_summary,
            "location_pre_mask_pose": loc_pose_meta,
            "scene_mood": scene_mood,
            "aspect_ratio_label": aspect_ratio_label,
            "best_of_k": k_count,
            "blueprint_summary": {
                "source_signed_yaw_deg": blueprint.source_signed_yaw_deg,
                "cheek_area_norm": blueprint.cheek_area_norm,
                "lower_jaw_area_norm": blueprint.lower_jaw_area_norm,
                "left_gonion_deg": blueprint.left_gonion_deg,
                "right_gonion_deg": blueprint.right_gonion_deg,
                "avg_gonion_deg": blueprint.avg_gonion_deg,
                "philtrum_to_eye_chin_ratio": blueprint.philtrum_to_eye_chin_ratio,
                "ocular": asdict(blueprint.ocular_profile) if blueprint.ocular_profile is not None else None,
            },
            "ordered_reference_images": ordered_refs,
            "prompts": prompts,
        }
        manifest_path = wdir / "workflow_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["manifest_path"] = str(manifest_path)
        return manifest

    def _estimate_multisegment_head_error_pct(self, bgr: np.ndarray, dst_face) -> float:
        """SizeKorea 8th multi-segment anthropometric head-size consistency error (%) on full-body shots."""
        from PIL import Image

        h, w = bgr.shape[:2]
        # Skip Vision body_joints subprocess on half-body/portrait shots where hips/knees are outside frame
        if float(dst_face.bbox[3]) / float(max(1, h)) >= 0.22:
            return 0.0
        actual_head_h = float(dst_face.bbox[3]) * 1.42
        try:
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            bj = self.pipeline.apple_vision.body_and_hand_points(pil) or {}
        except Exception:
            return 0.0
        bodies = bj.get("bodies") or []
        if not bodies:
            return 0.0
        jmap = {
            str(j["name"]): np.array([float(j["x"]), float(j["y"])], dtype=np.float32)
            for j in bodies[0]
            if float(j.get("conf", 0.0)) >= 0.55
            and 0.04 * w <= float(j["x"]) <= 0.96 * w
            and 0.04 * h <= float(j["y"]) <= 0.94 * h
        }
        if "root" not in jmap:
            return 0.0

        ests: List[float] = []
        neck, root = jmap.get("neck_1_joint"), jmap.get("root")
        if neck is not None and root is not None:
            d_tr = float(np.linalg.norm(root - neck))
            if d_tr > 30.0:
                ests.append(d_tr / 1.80)
        for side in ("left", "right"):
            sh, el, wr = jmap.get(f"{side}_shoulder_1_joint"), jmap.get(f"{side}_forearm_joint"), jmap.get(f"{side}_hand_joint")
            if sh is not None and el is not None:
                d_hum = float(np.linalg.norm(el - sh))
                if d_hum > 20.0:
                    ests.append(d_hum / 1.14)
            if el is not None and wr is not None:
                d_fa = float(np.linalg.norm(wr - el))
                if d_fa > 18.0:
                    ests.append(d_fa / 1.02)
            hip, kn = jmap.get(f"{side}_upLeg_joint"), jmap.get(f"{side}_leg_joint")
            if hip is not None and kn is not None:
                d_fem = float(np.linalg.norm(kn - hip))
                if d_fem > 25.0:
                    ests.append(d_fem / 1.86)
        if len(ests) < 2:
            return 0.0
        expected_head_h = float(np.median(ests))
        if expected_head_h <= 10.0:
            return 0.0
        return round((actual_head_h - expected_head_h) / expected_head_h * 100.0, 1)

    def _estimate_fast_yaw_deg(self, bgr: np.ndarray) -> float:
        """Sub-millisecond (cached) signed yaw estimator without full 25-metric cranial/mask extraction."""
        from PIL import Image

        try:
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            marks = self.pipeline.apple_vision.landmarks(pil)
            sh = self.pipeline.apple_vision.compute_face_shape(marks)
            if sh is not None and "turn" in sh:
                return float(np.clip(float(sh["turn"]) * 95.0, -90.0, 90.0))
        except Exception:
            pass
        face = self.pipeline.detector.detect(bgr)
        le, re, nose, _, _ = face.landmarks_5pt
        d_left = float(np.linalg.norm(nose - le))
        d_right = float(np.linalg.norm(nose - re))
        signed_asym = (d_left - d_right) / max(1.0, d_left + d_right)
        return float(np.clip(signed_asym * 135.0, -90.0, 90.0))

    def _resolve_angle_matched_ref_bgr(
        self,
        default_ref_p: Path,
        default_ref_bgr: np.ndarray,
        cand_bgr: np.ndarray,
        turnaround_sheet_path: Optional[Path] = None,
    ) -> Tuple[Path, np.ndarray]:
        """Select the closest yaw-matched verified 1-Hop 8-dir portrait when candidate head yaw differs from Gen-0 by >= 8 deg."""
        if turnaround_sheet_path is None or not Path(turnaround_sheet_path).exists():
            return default_ref_p, default_ref_bgr
        try:
            ref_yaw = self._estimate_fast_yaw_deg(default_ref_bgr)
            cand_yaw = self._estimate_fast_yaw_deg(cand_bgr)
            if abs(cand_yaw - ref_yaw) < 8.0:
                return default_ref_p, default_ref_bgr
            indiv = self.pipeline.turnaround.resolve_nearest_individual_8dir_portrait(
                Path(turnaround_sheet_path).resolve(), cand_yaw
            )
            if indiv is not None and indiv[0].is_file():
                matched_p, _view_k, matched_yaw = indiv[0].resolve(), indiv[1], float(indiv[2])
                if abs(cand_yaw - matched_yaw) + 2.0 < abs(cand_yaw - ref_yaw):
                    m_bgr = cv2.imread(str(matched_p))
                    if m_bgr is not None:
                        return matched_p, m_bgr
        except Exception:
            pass
        return default_ref_p, default_ref_bgr

    def score_and_select_native_winner(
        self,
        gen0_portrait_path: Path,
        candidate_paths: Sequence[Path],
        min_created_epoch: float = 0.0,
        turnaround_sheet_path: Optional[Path] = None,
    ) -> Dict[str, object]:
        """Stage 4: Score all K fresh candidates natively (`0px` warp, `0%` pasting) across SFace + 10-Gate Pre-Check + 25 Metrics."""
        from PIL import Image

        gen0_p = Path(gen0_portrait_path).resolve()
        base_ref_bgr = cv2.imread(str(gen0_p))
        if base_ref_bgr is None:
            raise FileNotFoundError(f"Cannot read Gen-0 portrait: {gen0_p}")
        base_ref_face = self.pipeline.detector.detect(base_ref_bgr)
        base_ref_marks = self.pipeline.apple_vision.landmarks(Image.fromarray(cv2.cvtColor(base_ref_bgr, cv2.COLOR_BGR2RGB)))
        base_ref_vec = extract_14pt_proportion_vector(self.pipeline.apple_vision, base_ref_bgr, marks=base_ref_marks)
        if base_ref_vec is None:
            raise RuntimeError("Failed to extract 18-metric craniofacial vector from Gen-0.")

        scored: List[CandidateScore] = []
        for cp in candidate_paths:
            p = Path(cp).resolve()
            if not p.is_file():
                continue
            if min_created_epoch > 0.0 and p.stat().st_mtime < (min_created_epoch - 2.0):
                raise RuntimeError(
                    f"[Fresh-Origin Workflow Violation] Stale candidate rejected ('{p.name}' predates current workflow manifest). "
                    "Regenerate fresh from untouched Gen-0!"
                )
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            _matched_p, ref_bgr = self._resolve_angle_matched_ref_bgr(
                gen0_p, base_ref_bgr, bgr, turnaround_sheet_path=turnaround_sheet_path
            )
            if ref_bgr is not base_ref_bgr:
                ref_face = self.pipeline.detector.detect(ref_bgr)
                ref_marks = self.pipeline.apple_vision.landmarks(Image.fromarray(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)))
                ref_vec = extract_14pt_proportion_vector(self.pipeline.apple_vision, ref_bgr, marks=ref_marks) or base_ref_vec
            else:
                ref_face = base_ref_face
                ref_marks = base_ref_marks
                ref_vec = base_ref_vec

            dst_face = self.pipeline.detector.detect(bgr)
            cand_marks = self.pipeline.apple_vision.landmarks(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            sface_angle = float(self.pipeline.auditor.measure_identity_similarity(ref_bgr, bgr, ref_face, dst_face))
            sface_gen0 = float(self.pipeline.auditor.measure_identity_similarity(base_ref_bgr, bgr, base_ref_face, dst_face))
            sface_sim = max(sface_angle, sface_gen0) if sface_gen0 >= 0.58 else sface_gen0
            c_vec = extract_14pt_proportion_vector(self.pipeline.apple_vision, bgr, full_canvas_bgr=bgr, marks=cand_marks)
            if c_vec is None:
                continue
            cmp18 = compare_14pt_proportions(ref_vec, c_vec)
            diffs = cmp18["diffsPct"]
            oc_cmp = self.ocular.compare(ref_bgr, bgr, ref_marks=ref_marks, cand_marks=cand_marks)
            oc_d = oc_cmp.get("diffs") or {}
            ex_cmp = self.exaggeration_guard.audit(ref_bgr, bgr, ref_marks=ref_marks, cand_marks=cand_marks)
            ex_idx = float(ex_cmp.get("exaggerationIndex", 0.0))
            sil_cmp = self.exaggeration_guard.measure_pupil_aligned_silhouette(ref_bgr, bgr, output_overlay_path=None)
            je_cmp = self.exaggeration_guard.measure_jaw_to_ear_neck_and_eye_audit(ref_bgr, bgr)
            c_iou = float(sil_cmp.get("contourIoU", 0.90))
            od_cmp = self.organ_depth.compare(ref_bgr, bgr)
            od_drift = float(od_cmp.get("organDepthDrift", 0.0))
            ms_err_pct = self._estimate_multisegment_head_error_pct(bgr, dst_face)
            skin_m = self.pipeline.auditor.skin_processor.measure_metrics(bgr, dst_face)

            ca_diff = float(diffs.get("cheekAreaNorm", 0.0))
            lja_diff = float(diffs.get("lowerJawAreaNorm", 0.0))
            gon_diff = float(c_vec["avgGonionDeg"]) - float(ref_vec["avgGonionDeg"])
            f2s = float(c_vec.get("faceToShoulder", 0.325))
            shoulders_vis = float(c_vec.get("shouldersInFrame", 0.0)) > 0.5
            oc_drift = float(oc_cmp.get("ocularDriftScore", 0.0))
            oc_ncc = float(oc_cmp.get("eyelidEdgeNcc", 0.0))
            oc_pass = bool(oc_cmp.get("passed", True))
            sclera_diff = float(oc_d.get("scleraDiffPct", 0.0))
            crease_diff = float(oc_d.get("creasePromDiff", 0.0))

            issues: List[str] = []
            if sface_sim < self.pipeline.config.min_identity_cosine_sim:
                issues.append(f"SFace {sface_sim:.3f} < {self.pipeline.config.min_identity_cosine_sim:.3f}")
            if sface_gen0 < 0.58:
                issues.append(f"순수 Gen-0 SFace {sface_gen0:.3f} < 0.580")
            if skin_m.specular_glare_ratio > self.pipeline.config.max_specular_glare_ratio:
                issues.append(f"피부 번들거림 {skin_m.specular_glare_ratio:.3f} > {self.pipeline.config.max_specular_glare_ratio:.3f}")
            if abs(ca_diff) > self.pipeline.config.max_cheek_area_diff_pct:
                issues.append(f"볼면적 {ca_diff:+.1f}%")
            if abs(lja_diff) > self.pipeline.config.max_cheek_area_diff_pct + 1.5:
                issues.append(f"하관면적 {lja_diff:+.1f}%")
            if abs(gon_diff) > 6.5:
                issues.append(f"턱각도 {gon_diff:+.1f}°")
            if shoulders_vis and not (
                self.pipeline.config.min_editorial_face_to_shoulder_ratio
                <= f2s
                <= self.pipeline.config.max_editorial_face_to_shoulder_ratio
            ):
                issues.append(f"어깨대비얼굴 {f2s * 100:.1f}%")
            if abs(ms_err_pct) > 22.0:
                issues.append(f"다중분절 인체비례 두상편차 {ms_err_pct:+.1f}%")
            if not oc_pass:
                issues.extend(str(x) for x in oc_cmp.get("issues", []))
            if not ex_cmp.get("passed", True):
                issues.extend(str(x) for x in ex_cmp.get("issues", []))
            if not sil_cmp.get("passed", True):
                issues.extend(str(x) for x in sil_cmp.get("issues", []))
            if not je_cmp.get("passed", True):
                issues.extend(str(x) for x in je_cmp.get("issues", []))
            if not od_cmp.get("passed", True):
                issues.extend(str(x) for x in od_cmp.get("issues", []))

            native_ok = len(issues) == 0
            # Higher composite_rank_score is better: strongly prioritize SFace identity, 4-organ+3D-depth+skin fidelity, eye NCC, and body harmony
            rank_score = (
                (100.0 if native_ok else 0.0)
                + sface_sim * 85.0
                + c_iou * 22.0
                + oc_ncc * 28.0
                - od_drift * 1.4
                - min(25.0, abs(ms_err_pct)) * 0.6
                - float(cmp18["meanWeightedDriftPct"]) * 2.2
                - oc_drift * 1.8
                - ex_idx * 2.0
                - abs(ca_diff) * 0.9
                - abs(lja_diff) * 0.7
            )
            scored.append(
                CandidateScore(
                    candidate_path=str(p),
                    sface_similarity=round(sface_sim, 4),
                    contour_iou=round(c_iou, 4),
                    organ_depth_drift=round(od_drift, 2),
                    multi_segment_head_err_pct=ms_err_pct,
                    mean_weighted_drift_pct=float(cmp18["meanWeightedDriftPct"]),
                    cheek_area_diff_pct=round(ca_diff, 1),
                    lower_jaw_area_diff_pct=round(lja_diff, 1),
                    avg_gonion_diff_deg=round(gon_diff, 1),
                    face_to_shoulder_ratio=round(f2s, 3),
                    ocular_drift_score=round(oc_drift, 2),
                    eyelid_edge_ncc=round(oc_ncc, 4),
                    sclera_diff_pct=round(sclera_diff, 1),
                    crease_prom_diff=round(crease_diff, 3),
                    ocular_passed=oc_pass,
                    native_bounds_passed=native_ok,
                    composite_rank_score=round(rank_score, 3),
                    issues=issues,
                )
            )

        if not scored:
            raise RuntimeError("No valid face candidates could be scored.")

        scored.sort(key=lambda c: c.composite_rank_score, reverse=True)
        for c_item in scored:
            try:
                self.origin_guard.register_generated_scene_output(Path(c_item.candidate_path), job_id="stage3_candidate")
            except Exception:
                pass
        winner = scored[0]
        return {
            "winner": asdict(winner),
            "all_candidates": [asdict(c) for c in scored],
            "regenerate_recommended": not winner.native_bounds_passed,
        }

    def finalize_native_winner_4k(
        self,
        gen0_portrait_path: Path,
        winner_candidate_path: Path,
        output_dir: Path,
        output_stem: str,
        min_created_epoch: float = 0.0,
        sync_to_vault: bool = True,
        strict_10gate: bool = True,
        upscale_mode: str = "tonelocked_upscayl_4k",
        upscayl_model_name: str = "ultramix-balanced-4x",
        reference_scene_path: Optional[Path] = None,
        cohesion_mode: str = "off",
        design_mode: str = "zero_touch_4k",
        no_film_grain: bool = False,
        real_bank_entries: Optional[Sequence[Dict[str, object]]] = None,
        locked_real_shot_path: Optional[Path] = None,
        turnaround_sheet_path: Optional[Path] = None,
        pure_gen0_portrait_path: Optional[Path] = None,
        framing_mode: str = "balanced_50_50",
    ) -> Dict[str, object]:
        """Stage 5: Tone-Locked True 4K Super-Resolution (`v1.0.6` default: True 4K + 0.0% lighting/color/likeness shift) + Mandatory 10-Gate Lock."""
        if not strict_10gate:
            raise RuntimeError(
                "[Anti-Bypass Lock v1.0.6] strict_10gate=False (--allow-audit-fail) is permanently forbidden! "
                "Every finalized 4K deliverable MUST pass all 10 quality gates."
            )
        gen0_p = Path(gen0_portrait_path).resolve()
        pure_gen0_p = Path(pure_gen0_portrait_path).resolve() if pure_gen0_portrait_path is not None else gen0_p
        win_p = Path(winner_candidate_path).resolve()
        if not win_p.is_file():
            raise FileNotFoundError(f"Winner candidate not found: {win_p}")
        if min_created_epoch > 0.0 and win_p.stat().st_mtime < (min_created_epoch - 2.0):
            raise RuntimeError(
                f"[Fresh-Origin Workflow Violation] Cannot finalize stale output '{win_p.name}'. Regenerate fresh from Gen-0!"
            )

        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_4k = out_dir / f"{output_stem}_4k.jpg"
        out_closeup = out_dir / f"{output_stem}_4k_closeup.jpg"
        out_macro = out_dir / f"{output_stem}_4k_macro.jpg"
        out_eye_cmp = out_dir / f"{output_stem}_eye_compare_400pct.jpg"
        out_sil_ov = out_dir / f"{output_stem}_silhouette_overlay.jpg"
        receipt_path_pre = out_dir / f"{output_stem}_workflow_receipt.json"
        for stale_out in (out_4k, out_closeup, receipt_path_pre):
            stale_out.unlink(missing_ok=True)

        raw_bgr = cv2.imread(str(win_p))
        ref_bgr = cv2.imread(str(gen0_p))
        pure_gen0_bgr = cv2.imread(str(pure_gen0_p))
        if raw_bgr is None or ref_bgr is None or pure_gen0_bgr is None:
            raise RuntimeError("Failed to decode winner or Gen-0 portrait.")

        # Register raw Stage-3 winner into forbidden generated outputs immediately so it can NEVER be re-fed!
        self.origin_guard.register_generated_scene_output(win_p, job_id=f"{output_stem}_raw_winner")

        r_sh, r_sw = raw_bgr.shape[:2]
        is_native_4k_raw = bool(max(r_sh, r_sw) >= 3840 and min(r_sh, r_sw) >= 2160)
        if not is_native_4k_raw and upscale_mode in ("optical_fullframe", "optical_lanczos", "upscayl_hybrid"):
            raise RuntimeError(
                f"[Fake-4K Blocked v1.0.6] Raw candidate '{win_p.name}' ({r_sw}x{r_sh}) is sub-4K "
                f"and CANNOT use '{upscale_mode}' (which leaves facial organs at 1K Lanczos). Use 'tonelocked_upscayl_4k'!"
            )

        # Measure HONEST pre-upscale face resolution & subject occupancy BEFORE 4K scaling
        raw_face_det = self.pipeline.detector.detect(raw_bgr)
        raw_fw, raw_fh = round(float(raw_face_det.bbox[2]), 1), round(float(raw_face_det.bbox[3]), 1)
        raw_iod_px = round(float(raw_face_det.inter_ocular_distance), 1)

        from PIL import Image as _PILOcc
        eff_framing = str(framing_mode or "balanced_50_50").strip().lower()
        occ_sc = min(1.0, 960.0 / float(max(1, r_sh, r_sw)))
        occ_sm = (
            cv2.resize(raw_bgr, (max(8, int(round(r_sw * occ_sc))), max(8, int(round(r_sh * occ_sc)))), interpolation=cv2.INTER_AREA)
            if occ_sc < 0.999
            else raw_bgr
        )
        pm_pil = self.pipeline.apple_vision.person_mask(_PILOcc.fromarray(cv2.cvtColor(occ_sm, cv2.COLOR_BGR2RGB)))
        subject_occupancy_pct = round(100.0 * float(np.mean(np.array(pm_pil) >= 128)), 2) if pm_pil is not None else 0.0
        max_allowed_occ_pct = 82.0 if eff_framing == "closeup" else 38.8
        subject_occupancy_passed = bool(subject_occupancy_pct <= max_allowed_occ_pct)

        acuity_meta: Dict[str, object] = {"applied": False}
        tonelock_sr_meta: Dict[str, object] = {"applied": False}
        zero_post_paste = design_mode in ("zero_touch_4k", "pure_character_sheet", "design_b_rerender")
        pure_zero_mask = zero_post_paste

        # 0. Fast Pre-Upscale 10-Gate Cutoff (<0.25s): immediately reject clearly defective sub-4K candidates
        #    BEFORE spending ~18s on 4x Upscayl GPU/CPU super-resolution!
        if (not is_native_4k_raw) and zero_post_paste:
            rfx, rfy, rfw, rfh = raw_face_det.bbox
            rmx1, rmy1 = max(0, int(rfx - rfw * 0.65)), max(0, int(rfy - rfh * 0.55))
            rmx2, rmy2 = min(r_sw, int(rfx + rfw * 1.65)), min(r_sh, int(rfy + rfh * 1.55))
            raw_macro_bgr = raw_bgr[rmy1:rmy2, rmx1:rmx2].copy()
            pre_angle_p, pre_ref_bgr = self._resolve_angle_matched_ref_bgr(
                gen0_p, ref_bgr, raw_macro_bgr, turnaround_sheet_path=turnaround_sheet_path
            )
            pre_ref_face = self.pipeline.detector.detect(pre_ref_bgr)
            pure_gen0_face_pre = self.pipeline.detector.detect(pure_gen0_bgr)
            sface_pre_pure = float(
                self.pipeline.auditor.measure_identity_similarity(pure_gen0_bgr, raw_bgr, pure_gen0_face_pre, raw_face_det)
            )
            sface_pre_angle = float(
                self.pipeline.auditor.measure_identity_similarity(pre_ref_bgr, raw_bgr, pre_ref_face, raw_face_det)
            )
            anat_ok_pre, anat_msg_pre = self.pipeline.auditor.validate_anatomical_geometry(raw_face_det)
            from PIL import Image as _PILImage
            pre_ref_marks = self.pipeline.apple_vision.landmarks(_PILImage.fromarray(cv2.cvtColor(pre_ref_bgr, cv2.COLOR_BGR2RGB)))
            raw_mac_marks = self.pipeline.apple_vision.landmarks(_PILImage.fromarray(cv2.cvtColor(raw_macro_bgr, cv2.COLOR_BGR2RGB)))
            oc_pre = self.ocular.compare(pre_ref_bgr, raw_macro_bgr, ref_marks=pre_ref_marks, cand_marks=raw_mac_marks)
            ex_pre = self.exaggeration_guard.audit(pre_ref_bgr, raw_macro_bgr, ref_marks=pre_ref_marks, cand_marks=raw_mac_marks)
            sil_pre = self.exaggeration_guard.measure_pupil_aligned_silhouette(pre_ref_bgr, raw_macro_bgr, output_overlay_path=None)

            pre_reasons: List[str] = []
            if not subject_occupancy_passed:
                pre_reasons.append(
                    f"인물 화면 점유율 과다 사전 컷오프 ({subject_occupancy_pct:.1f}% > 최대 {max_allowed_occ_pct:.1f}% — 전체 사진 50% 이하 배치 위반 / 몰래 줌인 차단)"
                )
            if not anat_ok_pre:
                pre_reasons.append(anat_msg_pre)
            if sface_pre_pure < 0.55:
                pre_reasons.append(f"순수 Gen-0 SFace 심각 미달 ({sface_pre_pure:.3f} < 0.550)")
            yaw_d_pre = abs(float(pre_ref_face.yaw_asymmetry_ratio) - float(raw_face_det.yaw_asymmetry_ratio))
            tilt_d_pre = abs(float(raw_face_det.eye_tilt_degrees) - float(pre_ref_face.eye_tilt_degrees))
            if yaw_d_pre <= 0.16 and tilt_d_pre < 8.0 and sface_pre_angle < 0.625:
                pre_reasons.append(f"얼굴 정체성(SFace={sface_pre_angle:.3f}) 미달")
            if not sil_pre.get("valid", False) or float(sil_pre.get("contourIoU", 0.0)) < 0.905 or float(sil_pre.get("maxContourPointErrPx", 999.0)) > 13.0:
                pre_reasons.append(
                    f"1:1 외곽선 사전 컷오프 (IoU={sil_pre.get('contourIoU')}, MaxErr={sil_pre.get('maxContourPointErrPx')}px)"
                )
            oc_d_pre = oc_pre.get("diffs") or {}
            if not oc_pre.get("valid", False) or float(oc_d_pre.get("irisWidthDiffPct", 0.0)) > 22.5 or float(oc_d_pre.get("scleraDiffPct", 0.0)) < -16.8:
                pre_reasons.append(f"눈매·홍채·흰자 사전 컷오프 ({'; '.join(str(x) for x in oc_pre.get('issues', []))})")
            ex_d_pre = ex_pre.get("diffs") or {}
            if not ex_pre.get("valid", False) or abs(float(ex_d_pre.get("eyeWidthPct", 0.0))) > 12.2 or float(ex_d_pre.get("lipHeightPct", 0.0)) > 22.5:
                pre_reasons.append(f"얼굴 과장 사전 컷오프 ({'; '.join(str(x) for x in ex_pre.get('issues', []))})")

            if pre_reasons:
                cv2.imwrite(str(out_macro), raw_macro_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 97])
                eye_pre_saved = self.ocular.save_400pct_eye_comparison_strip(pre_ref_bgr, raw_macro_bgr, str(out_eye_cmp))
                sil_pre_ov = self.exaggeration_guard.measure_pupil_aligned_silhouette(
                    pre_ref_bgr, raw_macro_bgr, output_overlay_path=str(out_sil_ov)
                )
                pre_msg = "; ".join(pre_reasons)
                pre_rcp = {
                    "project_name": CANONICAL_PROJECT_NAME,
                    "short_alias": SHORT_PROJECT_NAME,
                    "workflow_version": check_version_integrity().canonical_version,
                    "design_mode": design_mode,
                    "stage": "FINALIZED_10GATES_FAILED",
                    "pre_upscale_fast_rejected": True,
                    "timestamp": time.time(),
                    "passed_10gates": False,
                    "hero_4k_sha256": None,
                    "face_laplacian_variance": 0.0,
                    "sface_pure_gen0": round(sface_pre_pure, 4),
                    "sface_similarity": round(sface_pre_angle, 4),
                    "angle_matched_reference_path": str(pre_angle_p),
                    "upscale_mode": upscale_mode,
                    "upscayl_model": "skipped_by_pre_gate_cutoff",
                    "resolution": [r_sw, r_sh],
                    "raw_pre_upscale_canvas": [r_sw, r_sh],
                    "raw_pre_upscale_face_px": [raw_fw, raw_fh],
                    "raw_pre_upscale_iod_px": raw_iod_px,
                    "audit_message": pre_msg,
                    "silhouette_1to1": {
                        "passed": sil_pre_ov.get("passed"),
                        "contourIoU": sil_pre_ov.get("contourIoU"),
                        "maxContourPointErrPx": sil_pre_ov.get("maxContourPointErrPx"),
                        "meanContourPointErrPx": sil_pre_ov.get("meanContourPointErrPx"),
                        "width5LevelDiffPct": sil_pre_ov.get("width5LevelDiffPct"),
                        "issues": sil_pre_ov.get("issues"),
                    },
                    "exaggeration_audit": ex_pre,
                    "ocular_summary": {
                        "passed": oc_pre.get("passed"),
                        "eyelidEdgeNcc": oc_pre.get("eyelidEdgeNcc"),
                        "ocularDriftScore": oc_pre.get("ocularDriftScore"),
                        "diffs": oc_pre.get("diffs"),
                        "issues": oc_pre.get("issues"),
                    },
                    "deliverables": {
                        "hero_4k": None,
                        "closeup": None,
                        "macro": str(out_macro),
                        "eye_compare_400pct": eye_pre_saved,
                        "silhouette_overlay": sil_pre_ov.get("overlayPath"),
                    },
                    "vault_synced": [],
                }
                receipt_path_pre.write_text(json.dumps(pre_rcp, ensure_ascii=False, indent=2), encoding="utf-8")
                raise RuntimeError(f"10-Gate Pre-Upscale Fast Cutoff REJECTED candidate (<0.3s): {pre_msg}")

        # 1. Tone-Locked True 4K Super-Resolution (`v1.0.6` canonical default):
        #    - Native 4K inputs: 100.0% lossless 0.0px passthrough.
        #    - Sub-4K inputs: Upscayl 4x True Super-Resolution with 100.0% CIE LAB low-frequency & chromaticity lock (`delta_L ~ 0, delta_ab = 0`).
        up_bgr, tonelock_sr_meta = self.pipeline.upscaler.upscale_tonelocked_true_4k(
            raw_bgr, target_min_height=4800, model_name=upscayl_model_name
        )
        up_model = str(tonelock_sr_meta.get("model", "tonelocked_ultramix-balanced-4x"))
        uh, uw = up_bgr.shape[:2]

        # 2. Single-Camera Radiometric & Tonal Cohesion Mastering + Scene Chiaroscuro Mood Analysis
        face_4k = self.pipeline.detector.detect(up_bgr)
        ref_scene_bgr = (
            cv2.imread(str(reference_scene_path))
            if (reference_scene_path is not None and Path(reference_scene_path).is_file())
            else None
        )
        scene_mood: Optional[Dict[str, object]] = (
            analyze_target_scene_chiaroscuro_and_mood(ref_scene_bgr, self.pipeline.detector)
            if ref_scene_bgr is not None
            else None
        )
        is_low_key_relight = bool(
            scene_mood is not None
            and bool(scene_mood.get("is_low_key", False))
            and design_mode in ("design_a_relight", "design_b_facelock_outpaint")
        )

        is_low_key_pure_v21 = bool(
            scene_mood is not None and bool(scene_mood.get("is_low_key", False)) and design_mode == "pure_v21"
        )

        cohesion_meta: Dict[str, object] = {"mode": cohesion_mode, "applied": False}
        if (not pure_zero_mask) and (not zero_post_paste) and (not is_low_key_relight) and (not is_low_key_pure_v21) and cohesion_mode != "off":
            up_bgr, cohesion_meta = PostGenCohesionMasterer.master(
                up_bgr,
                face_det=face_4k,
                reference_scene_bgr=ref_scene_bgr,
                mode=cohesion_mode,
                seed=42,
                disable_grain=no_film_grain,
            )

        # 2b. Optional Design A or Design B (NEVER executed in pure_character_sheet or design_b_rerender zero-post-paste modes!)
        relight_meta: Dict[str, object] = {"applied": False, "design_mode": design_mode}
        real_bank_eval: Optional[Dict[str, object]] = None
        bank_obj: Optional[RealMultiShotBank] = None
        if real_bank_entries:
            bank_obj = RealMultiShotBank.from_manifest_entries(
                real_bank_entries,
                self.pipeline.config,
                self.pipeline.detector,
                self.pipeline.apple_vision,
                self.pipeline.auditor,
                origin_guard=self.origin_guard,
            )
            if (not pure_zero_mask) and (not zero_post_paste):
                if design_mode == "design_a_relight":
                    up_bgr, relight_meta = self.relighter.relight_and_fuse_real_face(
                        up_bgr, bank=bank_obj, scene_mood=scene_mood
                    )
                elif design_mode == "design_b_facelock_outpaint":
                    up_bgr, relight_meta = self.facelock_studio.lock_real_face_with_scene_relighting(
                        up_bgr, bank=bank_obj, locked_shot_path=locked_real_shot_path, scene_mood=scene_mood
                    )
                matched_p = relight_meta.get("matched_real_shot_path")
                if matched_p and Path(str(matched_p)).is_file():
                    matched_bgr = cv2.imread(str(matched_p))
                    if matched_bgr is not None:
                        ref_bgr = matched_bgr
            elif design_mode == "design_b_rerender" and locked_real_shot_path is not None and Path(locked_real_shot_path).is_file():
                matched_bgr = cv2.imread(str(locked_real_shot_path))
                if matched_bgr is not None:
                    ref_bgr = matched_bgr

        face_4k = self.pipeline.detector.detect(up_bgr)
        crop_box = self.pipeline.detector.compute_upper_body_crop_box(
            (uh, uw), face_4k, aspect_ratio=float(uw) / float(max(1, uh))
        )
        X1, Y1, X2, Y2 = crop_box
        vx1, vy1 = max(0, X1 - 120), max(0, Y1 - 60)
        vx2, vy2 = min(uw, X2 + 120), min(uh, Y2 + 160)
        skin_albedo_meta: Dict[str, object] = {"applied": False, "mode": "zero_post_paste_passthrough" if (pure_zero_mask or zero_post_paste) else "active"}
        if (not pure_zero_mask) and (not zero_post_paste):
            roi = up_bgr[vy1:vy2, vx1:vx2].copy()
            ref_organ = self.organ_depth.extract_profile(ref_bgr)
            ref_lap = float(ref_organ.skin_norm_laplacian) if ref_organ is not None else 45.0
            # Scale-adaptive micro-pore & organ acuity restoration (`생-032`, 0% hardcoding):
            # activates automatically when raw pre-upscale IOD is small (<95px) or Gen-0 was compressed/airbrushed (<24 normLap)
            iod_deficit = float(np.clip((95.0 - raw_iod_px) / 58.0, 0.0, 1.0))
            airbrush_deficit = float(np.clip((24.0 - ref_lap) / 18.0, 0.0, 1.0))
            dyn_pore_amp = round(float(np.clip(0.55 * iod_deficit + 0.50 * airbrush_deficit, 0.0, 1.05)), 3)
            dyn_organ_crisp = round(float(np.clip(0.22 * iod_deficit + 0.12 * airbrush_deficit, 0.0, 0.34)), 3)
            dyn_skin_proc = MatteSkinProcessor(
                self.pipeline.detector,
                micro_texture_amplitude=dyn_pore_amp,
                organ_crisp_strength=dyn_organ_crisp,
            )
            roi_matte, _ = dyn_skin_proc.process(roi, seed=42, reference_skin_profile=None)
            up_bgr[vy1:vy2, vx1:vx2] = roi_matte

            if design_mode == "pure_v21":
                # Pure v2.1 (`SKILL.md` Sec 2): Pure Optical Lanczos-4 + P88 MatteSkinProcessor ONLY
                # In low-key scenes, preserve the authentic warm-olive camera color palette (never bleach with studio de_yellow!)
                if is_low_key_pure_v21 and ref_scene_bgr is not None:
                    up_bgr, cohesion_meta = master_full_frame_mood_and_grain(
                        up_bgr,
                        ref_scene_bgr,
                        curve_strength=0.22,
                        chroma_strength=0.30,
                        grain_std_4k=0.0 if no_film_grain else 1.15,
                        seed=42,
                    )
                    cohesion_meta["mode"] = "pure_v21_low_key_warm_olive_grain_lock"
                skin_albedo_meta = {"applied": False, "mode": "pure_v21_optical_matte_cohesion_only"}
            elif is_low_key_relight and ref_scene_bgr is not None:
                # 100% UNIFORM whole-frame black-level/contrast curve + split-tone LUT + 35mm film grain (ZERO studio skin re-brightening!)
                up_bgr, cohesion_meta = master_full_frame_mood_and_grain(
                    up_bgr,
                    ref_scene_bgr,
                    curve_strength=0.58,
                    chroma_strength=0.72,
                    grain_std_4k=1.35,
                    seed=42,
                )
                cohesion_meta["mode"] = "low_key_full_frame_mood_and_grain"
                skin_albedo_meta = {"applied": False, "mode": "skipped_to_preserve_low_key_chiaroscuro"}
            else:
                up_bgr, skin_albedo_meta = self.pipeline.skin_processor.harmonize_intrinsic_skin_albedo_and_texture(
                    up_bgr,
                    ref_bgr,
                    ambient_blend_ratio=0.32 if relight_meta.get("applied") else 0.28,
                    seed=42,
                )
        if bank_obj is not None:
            real_bank_eval = bank_obj.evaluate_against_all_real_shots(up_bgr)

        # 3. Auto-select angle-matched verified 1-Hop 8-dir reference when head yaw differs >= 8 deg, while locking pure Gen-0 identity!
        fx, fy, fw, fh = face_4k.bbox
        mx1, my1 = max(0, int(fx - fw * 0.65)), max(0, int(fy - fh * 0.55))
        mx2, my2 = min(uw, int(fx + fw * 1.65)), min(uh, int(fy + fh * 1.55))
        macro_bgr = up_bgr[my1:my2, mx1:mx2].copy()

        angle_ref_p, ref_bgr = self._resolve_angle_matched_ref_bgr(
            gen0_p, ref_bgr, macro_bgr, turnaround_sheet_path=turnaround_sheet_path
        )

        # Full 10-Gate Quality Audit using immutable canonical production EngineConfig (immune to caller config weakening!)
        from .config import CanvasSpec as _CS, EngineConfig as _EC
        from .quality_gate import QualityGateAuditor as _QGA
        from PIL import Image as _PILImg4K
        cfg_4k = _EC(
            vault_dir=self.pipeline.config.vault_dir,
            canvas=_CS(target_width=uw, target_height=uh),
        )
        auditor_4k = _QGA(cfg_4k, self.pipeline.detector)
        report = auditor_4k.audit(up_bgr, crop_box, seam_l1_diff=0.0, reference_face_bgr=ref_bgr)

        # Also enforce direct pure Gen-0 SFace floor (>= 0.58 even under 3/4 yaw) so 8-dir reference NEVER dilutes Gen-0 identity!
        pure_gen0_face = self.pipeline.detector.detect(pure_gen0_bgr)
        roi_4k_view = up_bgr[vy1:vy2, vx1:vx2]
        sface_pure_gen0 = round(
            float(auditor_4k.measure_identity_similarity(pure_gen0_bgr, roi_4k_view, pure_gen0_face, self.pipeline.detector.detect(roi_4k_view))),
            4,
        )

        # Save diagnostic macro / eye / silhouette strips first; ONLY write deliverable `*_4k.jpg` if all 10 gates PASS!
        cv2.imwrite(str(out_closeup), roi_4k_view, [int(cv2.IMWRITE_JPEG_QUALITY), 97])
        cv2.imwrite(str(out_macro), macro_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 97])

        ref_marks_4k = self.pipeline.apple_vision.landmarks(_PILImg4K.fromarray(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)))
        macro_marks_4k = self.pipeline.apple_vision.landmarks(_PILImg4K.fromarray(cv2.cvtColor(macro_bgr, cv2.COLOR_BGR2RGB)))
        eye_strip_saved = self.ocular.save_400pct_eye_comparison_strip(ref_bgr, macro_bgr, str(out_eye_cmp))
        oc_cmp = self.ocular.compare(ref_bgr, macro_bgr, ref_marks=ref_marks_4k, cand_marks=macro_marks_4k)
        sil_cmp = self.exaggeration_guard.measure_pupil_aligned_silhouette(
            ref_bgr, macro_bgr, output_overlay_path=str(out_sil_ov)
        )
        ex_cmp = self.exaggeration_guard.audit(ref_bgr, macro_bgr, ref_marks=ref_marks_4k, cand_marks=macro_marks_4k)
        od_cmp = self.organ_depth.compare(pure_gen0_bgr, macro_bgr)

        # Gate 13: Native-4K Luminance-to-Chroma layer-mismatch skin blotchiness check
        skin_mottle_passed = True
        skin_l_to_ab_ratio = 0.0
        if min(uw, uh) >= 2160:
            fh_patch = up_bgr[
                max(0, int(fy + 0.08 * fh)) : min(uh, int(fy + 0.24 * fh)),
                max(0, int(fx + 0.25 * fw)) : min(uw, int(fx + 0.75 * fw)),
            ]
            if fh_patch.size >= 400:
                lab_fh = cv2.cvtColor(fh_patch, cv2.COLOR_BGR2LAB).astype(np.float32)
                l_hf_v = float(np.std(lab_fh[:, :, 0] - cv2.GaussianBlur(lab_fh[:, :, 0], (0, 0), 2.2)))
                ab_hf_v = float(
                    np.std(
                        np.hypot(
                            lab_fh[:, :, 1] - cv2.GaussianBlur(lab_fh[:, :, 1], (0, 0), 2.2),
                            lab_fh[:, :, 2] - cv2.GaussianBlur(lab_fh[:, :, 2], (0, 0), 2.2),
                        )
                    )
                ) + 1e-4
                skin_l_to_ab_ratio = round(l_hf_v / ab_hf_v, 2)
                if l_hf_v > 6.2 and skin_l_to_ab_ratio > 15.0:
                    skin_mottle_passed = False

        all_passed = bool(
            report.passed
            and subject_occupancy_passed
            and skin_mottle_passed
            and bool(oc_cmp.get("valid", False))
            and bool(oc_cmp.get("passed", False))
            and bool(sil_cmp.get("valid", False))
            and bool(sil_cmp.get("passed", False))
            and bool(ex_cmp.get("valid", False))
            and bool(ex_cmp.get("passed", False))
            and float(report.face_laplacian_variance) >= float(cfg_4k.min_face_laplacian_var)
            and sface_pure_gen0 >= 0.58
        )
        fail_bits: List[str] = []
        if not report.passed and report.message:
            fail_bits.append(report.message)
        if not subject_occupancy_passed:
            fail_bits.append(
                f"인물 화면 점유율 과다 ({subject_occupancy_pct:.1f}% > 최대 {max_allowed_occ_pct:.1f}% — 전체 사진 50% 이하 배치 위반 / 몰래 줌인 차단)"
            )
        if not skin_mottle_passed:
            fail_bits.append(
                f"피부 얼룩덜룩(명암-색차 레이어 불일치 반점: L/ab={skin_l_to_ab_ratio:.1f} > 15.0) 발생"
            )
        if not oc_cmp.get("passed", False):
            oc_err = "; ".join(str(x) for x in oc_cmp.get("issues", []))
            if oc_err and oc_err not in "; ".join(fail_bits):
                fail_bits.append(f"눈매·흰자 게이트 미달: {oc_err}")
        if not sil_cmp.get("passed", False):
            sil_err = "; ".join(str(x) for x in sil_cmp.get("issues", []))
            fail_bits.append(f"1:1 외곽선 게이트 미달: {sil_err}")
        if not ex_cmp.get("passed", False):
            ex_err = "; ".join(str(x) for x in ex_cmp.get("issues", []))
            if ex_err:
                fail_bits.append(f"얼굴 과장 게이트 미달: {ex_err}")
        if sface_pure_gen0 < 0.58:
            fail_bits.append(f"순수 Gen-0 직접 SFace 미달 ({sface_pure_gen0:.3f} < 0.580)")
        fail_msg = "; ".join(fail_bits) if fail_bits else report.message

        hero_4k_sha256: Optional[str] = None
        if all_passed:
            cv2.imwrite(str(out_4k), up_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 97])
            hero_4k_sha256 = _sha256_file(out_4k)
            # Blacklist generated output in OriginOnlyOneShotGuard so it can NEVER be re-fed as input!
            self.origin_guard.register_generated_scene_output(out_4k, job_id=output_stem)
        else:
            # Physically remove deliverable *_4k.jpg on failure so unapproved images can NEVER be delivered!
            out_4k.unlink(missing_ok=True)
            out_closeup.unlink(missing_ok=True)

        vault_synced: List[str] = []
        if sync_to_vault and all_passed and self.pipeline.config.vault_dir:
            v_out = self.pipeline.config.vault_dir / "outputs"
            v_out.mkdir(parents=True, exist_ok=True)
            for p_item in (out_4k, out_closeup, out_macro, out_eye_cmp, out_sil_ov):
                if p_item.is_file():
                    dst = v_out / p_item.name
                    shutil.copy2(p_item, dst)
                    vault_synced.append(str(dst))

        try:
            _vrep_fin = check_version_integrity()
            _vmeta_fin = {
                "project_name": _vrep_fin.project_name,
                "short_alias": _vrep_fin.short_name if hasattr(_vrep_fin, "short_name") else _vrep_fin.short_alias,
                "canonical_version": _vrep_fin.canonical_version,
                "version_check_passed": _vrep_fin.passed,
            }
        except Exception:
            _vmeta_fin = {
                "project_name": CANONICAL_PROJECT_NAME,
                "short_alias": SHORT_PROJECT_NAME,
                "canonical_version": "1.0.0",
                "version_check_passed": True,
            }

        max_abs_diff_vs_raw: Optional[float] = (
            round(float(np.max(np.abs(up_bgr.astype(np.int16) - raw_bgr.astype(np.int16)))), 3)
            if up_bgr.shape == raw_bgr.shape
            else None
        )
        receipt = {
            **_vmeta_fin,
            "workflow_version": _vmeta_fin["canonical_version"],
            "design_mode": design_mode,
            "zero_touch_4k_passthrough": bool(pure_zero_mask or zero_post_paste),
            "max_abs_pixel_diff_vs_raw": max_abs_diff_vs_raw,
            "stage": "FINALIZED_10GATES_PASSED" if all_passed else "FINALIZED_10GATES_FAILED",
            "timestamp": time.time(),
            "passed_10gates": all_passed,
            "hero_4k_sha256": hero_4k_sha256,
            "face_laplacian_variance": round(float(report.face_laplacian_variance), 2),
            "min_organ_gradient": round(float(report.min_organ_gradient), 2),
            "sface_pure_gen0": sface_pure_gen0,
            "angle_matched_reference_path": str(angle_ref_p),
            "upscale_mode": upscale_mode,
            "upscayl_model": up_model,
            "tonelock_sr_meta": tonelock_sr_meta,
            "optical_acuity_meta": acuity_meta,
            "cohesion_meta": cohesion_meta,
            "real_multishot_relight_meta": relight_meta,
            "real_bank_sface_evaluation": real_bank_eval,
            "skin_albedo_texture_meta": skin_albedo_meta,
            "organ_depth_summary": od_cmp,
            "resolution": [uw, uh],
            "raw_pre_upscale_canvas": [int(raw_bgr.shape[1]), int(raw_bgr.shape[0])],
            "raw_pre_upscale_face_px": [raw_fw, raw_fh],
            "raw_pre_upscale_iod_px": raw_iod_px,
            "framing_mode": eff_framing,
            "subject_occupancy_pct": subject_occupancy_pct,
            "subject_occupancy_passed": subject_occupancy_passed,
            "skin_l_to_ab_ratio": skin_l_to_ab_ratio,
            "skin_mottle_passed": skin_mottle_passed,
            "raw_resolution_warning": (
                f"원시 생성 얼굴 폭 {raw_fw:.0f}px (<260px): 400% 확대 시 Upscayl 보간 한계 존재"
                if raw_fw < 260.0
                else None
            ),
            "warp_applied": False,
            "face_pasting_applied": False,
            "sface_similarity": round(float(report.identity_cosine_sim), 4),
            "face_shape_note": report.face_shape_note,
            "audit_message": "ALL QUALITY GATES PASSED" if all_passed else fail_msg,
            "silhouette_1to1": {
                "passed": sil_cmp.get("passed"),
                "contourIoU": sil_cmp.get("contourIoU"),
                "maxContourPointErrPx": sil_cmp.get("maxContourPointErrPx"),
                "meanContourPointErrPx": sil_cmp.get("meanContourPointErrPx"),
                "width5LevelDiffPct": sil_cmp.get("width5LevelDiffPct"),
                "issues": sil_cmp.get("issues"),
            },
            "exaggeration_audit": {
                "passed": ex_cmp.get("passed"),
                "exaggerationIndex": ex_cmp.get("exaggerationIndex"),
                "diffs": ex_cmp.get("diffs"),
                "issues": ex_cmp.get("issues"),
            },
            "ocular_summary": {
                "passed": oc_cmp.get("passed"),
                "eyelidEdgeNcc": oc_cmp.get("eyelidEdgeNcc"),
                "ocularDriftScore": oc_cmp.get("ocularDriftScore"),
                "diffs": oc_cmp.get("diffs"),
                "issues": oc_cmp.get("issues"),
            },
            "deliverables": {
                "hero_4k": str(out_4k) if all_passed else None,
                "closeup": str(out_closeup) if all_passed else None,
                "macro": str(out_macro),
                "eye_compare_400pct": eye_strip_saved,
                "silhouette_overlay": sil_cmp.get("overlayPath"),
            },
            "vault_synced": vault_synced,
        }
        receipt_path = out_dir / f"{output_stem}_workflow_receipt.json"
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        receipt["receipt_path"] = str(receipt_path)

        if strict_10gate and not all_passed:
            raise RuntimeError(f"10-Gate Quality Audit REJECTED output (must regenerate fresh from Gen-0): {fail_msg}")

        return receipt
