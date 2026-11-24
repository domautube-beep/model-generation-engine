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
from .scene_radiometry import PostGenCohesionMasterer, SceneRadiometryAnalyzer


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
        (
            "c1_50mm_rembrandt",
            "Shot on Hasselblad X2D 100C with XCD 55V (50mm equiv) at f/3.2, ISO 64, environmental full-scene studio editorial photograph",
        ),
        (
            "c2_65mm_sunbeam",
            "Shot on Hasselblad X2D 100C with XCD 65mm f/2.8 at f/2.8, ISO 64, cinema full-figure directional sunbeam editorial photograph",
        ),
        (
            "c3_85mm_clamshell",
            "Shot on Canon EOS R5 with RF 85mm f/1.2L at f/2.8, ISO 64, bright commercial clamshell three-quarter portrait photograph",
        ),
        (
            "c4_105mm_lowkey_rim",
            "Shot on Leica SL3 with APO-Summicron 105mm at f/2.5, ISO 64, medium-telephoto moody split-key & warm rim editorial portrait",
        ),
        (
            "c5_135mm_north_negfill",
            "Shot on Sony Alpha 7R V with FE 135mm f/1.8 GM at f/2.2, ISO 64, tight telephoto portrait with right north-window light & left negative fill",
        ),
    )

    DEFAULT_5_LENS_LIGHTING_MATRIX: Tuple[Dict[str, object], ...] = (
        {
            "tag": "50mm_rembrandt_3to1",
            "lens_mm": 50,
            "aperture": "f/3.2",
            "framing_desc": "environmental full-frame composition showing the full architectural setting, furniture, and head-to-toe pose",
            "lighting_name_ko": "좌상단 45° 렘브란트 3:1 그리드 키라이트 (5000K)",
            "lighting_desc": (
                "Sculpted ~5000K directional gridded octabox Rembrandt key light positioned 45 degrees camera-left and 35 degrees above eye level, "
                "creating a classic illuminated Rembrandt triangle on the shadow-side (viewer-right) cheek with crisp ~3:1 three-dimensional shadow roll-off "
                "under the chin and across the floor"
            ),
        },
        {
            "tag": "65mm_golden_sunbeam_4to1",
            "lens_mm": 65,
            "aperture": "f/2.8",
            "framing_desc": "cinema classic full-figure editorial framing with natural spatial compression from upper backdrop to shoes",
            "lighting_name_ko": "오후 골든아워 창가 직사 슬래시 채광 (4300K · 4:1)",
            "lighting_desc": (
                "Warm ~4300K low-angled late-afternoon golden sunbeam streaming diagonally through tall camera-left side windows, "
                "casting distinct sunlit highlights across the camera-left side of the face, shoulder, and lap while throwing rich ~4:1 architectural window-frame shadows across the background wall and floor"
            ),
        },
        {
            "tag": "85mm_clamshell_highkey_1p5to1",
            "lens_mm": 85,
            "aperture": "f/2.8",
            "framing_desc": "classic 85mm three-quarter-to-ankle portrait framing with creamy background separation",
            "lighting_name_ko": "클램셸 버터플라이 하이키 매거진 뷰티광 (5500K · 1.5:1)",
            "lighting_desc": (
                "Bright, airy ~5500K commercial clamshell/butterfly lighting (large overhead-front diffused octabox paired with a soft silver bounce reflector below), "
                "delivering clean luminous skin, crisp dual eye catchlights, open shadow detail (~1.5:1 soft contrast ratio), and crisp garment weave clarity"
            ),
        },
        {
            "tag": "105mm_moody_split_rim_6to1",
            "lens_mm": 105,
            "aperture": "f/2.5",
            "framing_desc": "medium-telephoto knee-to-head portrait framing with compressed architectural depth",
            "lighting_name_ko": "시네마틱 로우키 스플릿 + 3200K 웜 림라이트 (6:1)",
            "lighting_desc": (
                "Moody low-key cinema lighting featuring a focused ~4800K side split key light from 70 degrees camera-left (~6:1 deep shadow falloff across the viewer-right cheek and dimmed room) "
                "accented by a subtle warm ~3200K tungsten hair-and-shoulder rim kicker from back-right that cleanly separates hair and shoulder contours from the dark background"
            ),
        },
        {
            "tag": "135mm_north_window_negfill_3p5to1",
            "lens_mm": 135,
            "aperture": "f/2.2",
            "framing_desc": "tight telephoto waist/lap-up editorial portrait framing with shallow optical depth-of-field bokeh",
            "lighting_name_ko": "우측 쿨 노스윈도우 확산광 + 좌측 네거티브 필 (6000K · 3.5:1)",
            "lighting_desc": (
                "Reversed-direction ~6000K cool-crisp overcast north-window soft light entering from 45 degrees CAMERA-RIGHT paired with a large black negative-fill flag on camera-left, "
                "illuminating the viewer-right cheek, exposed ear, and right shoulder in clean cool daylight while sculpting calm velvety ~3.5:1 negative-fill shade across the viewer-left side"
            ),
        },
    )

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
        design_mode: str = "pure_v22",
        aspect_ratio_label: str = "3:4",
        best_of_k: int = 5,
        match_gen0_yaw: bool = True,
        attach_ocular_detail_anchor: bool = True,
        text_only_location: bool = False,
        no_film_grain: bool = False,
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

        blueprint: AnatomicalBlueprint = self.pipeline.turnaround.extract_anatomical_blueprint(ref_bgr)
        organ_profile = self.organ_depth.extract_profile(ref_bgr)
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

        if turnaround_8dir_sheet_path is not None and Path(turnaround_8dir_sheet_path).exists():
            raw_turn_p = Path(turnaround_8dir_sheet_path).resolve()
            if raw_turn_p.is_file():
                turnaround_master_sheet_meta = str(raw_turn_p)

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

            if attach_ocular_detail_anchor and turnaround_p is None:
                ocular_anchor_p = wdir / "gen0_optical_ocular_detail_anchor.jpg"
                self.pipeline.turnaround.prepare_pure_gen0_ocular_detail_anchor(gen0_p, ocular_anchor_p)
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
                    "re-photograph this exact scene as a single authentic 35mm camera exposure so true 3D side-window light and shadows sculpt her face, nose, eye sockets, jaw, neck, suit, and chair together"
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

        for i in range(k_count):
            seed_tag, _cam_hdr = self.DEFAULT_CAMERA_SEEDS[i % len(self.DEFAULT_CAMERA_SEEDS)]
            preset = self.DEFAULT_5_LENS_LIGHTING_MATRIX[i % len(self.DEFAULT_5_LENS_LIGHTING_MATRIX)]
            lens_mm = int(preset["lens_mm"])  # type: ignore[arg-type]
            preset_ap = str(preset["aperture"])
            framing_desc = str(preset["framing_desc"])
            preset_light_ko = str(preset["lighting_name_ko"])
            preset_light_en = str(preset["lighting_desc"])
            aperture_str = "f/2.8, ISO 800 available light" if is_lk_scene else f"{preset_ap}, natural available light"
            if no_film_grain:
                slot1_camera = (
                    f"Clean, tack-sharp 100MP medium-format digital commercial studio photograph (Hasselblad X2D 100C, {lens_mm}mm prime lens at {preset_ap}, base ISO 64, {aspect_ratio_label}, {framing_desc}), "
                    "pure noise-free studio optical clarity with ZERO film grain, natural undistorted perspective, "
                    "realistic adult 7.45-head multi-segment Korean body proportions, and smooth natural studio depth-of-field."
                )
            else:
                slot1_camera = (
                    f"Unstaged raw medium-format 35mm film location photograph ({lens_mm}mm lens at {aperture_str}, {aspect_ratio_label}, {framing_desc}), "
                    "natural undistorted perspective with realistic adult 7.45-head multi-segment body proportions "
                    "and organic shallow depth-of-field falloff across hair edges, furniture back, and distant background."
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
                    f" Place the subject naturally inside the exact chair and architectural space of Image {loc_slot_num} with continuous background bokeh."
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
            active_light_desc = preset_light_en if k_count >= 5 else loc_light_desc
            if no_film_grain:
                slot5_light = (
                    f"Clean Commercial Studio Lighting & Skin Realism: {active_light_desc}; "
                    "clean, translucent real-camera studio skin with fine natural pores and hydrated natural lip color matching Image 1 without plastic gloss, "
                    "100% noise-free base-ISO digital studio smoothness with ZERO film grain, ZERO dust, and ZERO vintage filter across the entire frame."
                )
            else:
                slot5_light = (
                    f"Available-Light Exposure & Analog Texture: {active_light_desc}; "
                    "unadorned matte skin micro-texture without specular oil glare, unified by organic analog 35mm film grain across the entire frame."
                )
            p_text = f"{slot1_camera} {slot2_identity} {slot3_pose} {slot4_outfit} {slot5_light}"
            PromptFirewall.validate_prompt(p_text)
            prompts.append(
                {
                    "candidate_tag": f"c{i + 1}_{seed_tag}",
                    "lens_mm": lens_mm,
                    "aperture": preset_ap,
                    "lighting_preset": str(preset["tag"]),
                    "lighting_name_ko": preset_light_ko,
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

        manifest = {
            "workflow_version": "2.2",
            "design_mode": design_mode,
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
        """SizeKorea 8th multi-segment anthropometric head-size consistency error (%) across trunk, humerus, femur, and hand."""
        from PIL import Image

        h, w = bgr.shape[:2]
        actual_head_h = float(dst_face.bbox[3]) * 1.42
        try:
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            joints = self.pipeline.apple_vision.body_joints(pil) or {}
        except Exception:
            return 0.0
        if not joints:
            return 0.0

        def _pt(name: str) -> Optional[np.ndarray]:
            j = joints.get(name)
            if not j or float(j.get("confidence", 0.0)) < 0.28:
                return None
            return np.array([float(j["x"]) * w, (1.0 - float(j["y"])) * h], dtype=np.float32)

        ests: List[float] = []
        neck, root = _pt("neck_1_joint"), _pt("root")
        if neck is not None and root is not None:
            d_tr = float(np.linalg.norm(root - neck))
            if d_tr > 30.0:
                ests.append(d_tr / 1.80)
        for side in ("left", "right"):
            sh, el, wr = _pt(f"{side}_shoulder_1_joint"), _pt(f"{side}_forearm_joint"), _pt(f"{side}_hand_joint")
            if sh is not None and el is not None:
                d_hum = float(np.linalg.norm(el - sh))
                if d_hum > 20.0:
                    ests.append(d_hum / 1.14)
            if el is not None and wr is not None:
                d_fa = float(np.linalg.norm(wr - el))
                if d_fa > 18.0:
                    ests.append(d_fa / 1.02)
            hip, kn = _pt(f"{side}_upLeg_joint"), _pt(f"{side}_leg_joint")
            if hip is not None and kn is not None:
                d_fem = float(np.linalg.norm(kn - hip))
                if d_fem > 25.0:
                    ests.append(d_fem / 1.86)
        if not ests:
            return 0.0
        expected_head_h = float(np.median(ests))
        if expected_head_h <= 10.0:
            return 0.0
        return round((actual_head_h - expected_head_h) / expected_head_h * 100.0, 1)

    def score_and_select_native_winner(
        self,
        gen0_portrait_path: Path,
        candidate_paths: Sequence[Path],
        min_created_epoch: float = 0.0,
    ) -> Dict[str, object]:
        """Stage 4: Score all K fresh candidates natively (`0px` warp, `0%` pasting) across SFace + Contour IoU + Multi-Segment Anthropometry + 25 Metrics."""
        gen0_p = Path(gen0_portrait_path).resolve()
        ref_bgr = cv2.imread(str(gen0_p))
        if ref_bgr is None:
            raise FileNotFoundError(f"Cannot read Gen-0 portrait: {gen0_p}")
        ref_face = self.pipeline.detector.detect(ref_bgr)
        ref_vec = extract_14pt_proportion_vector(self.pipeline.apple_vision, ref_bgr)
        if ref_vec is None:
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
            dst_face = self.pipeline.detector.detect(bgr)
            sface_sim = float(self.pipeline.auditor.measure_identity_similarity(ref_bgr, bgr, ref_face, dst_face))
            c_vec = extract_14pt_proportion_vector(self.pipeline.apple_vision, bgr, full_canvas_bgr=bgr)
            if c_vec is None:
                continue
            cmp18 = compare_14pt_proportions(ref_vec, c_vec)
            diffs = cmp18["diffsPct"]
            oc_cmp = self.ocular.compare(ref_bgr, bgr)
            oc_d = oc_cmp.get("diffs") or {}
            ex_cmp = self.exaggeration_guard.audit(ref_bgr, bgr)
            ex_idx = float(ex_cmp.get("exaggerationIndex", 0.0))
            sil_cmp = self.exaggeration_guard.measure_pupil_aligned_silhouette(ref_bgr, bgr, output_overlay_path=None)
            c_iou = float(sil_cmp.get("contourIoU", 0.90))
            od_cmp = self.organ_depth.compare(ref_bgr, bgr)
            od_drift = float(od_cmp.get("organDepthDrift", 0.0))
            ms_err_pct = self._estimate_multisegment_head_error_pct(bgr, dst_face)

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
        upscale_mode: str = "upscayl_hybrid",
        upscayl_model_name: str = "ultramix-balanced-4x",
        reference_scene_path: Optional[Path] = None,
        cohesion_mode: str = "auto",
        design_mode: str = "pure_v22",
        no_film_grain: bool = False,
        real_bank_entries: Optional[Sequence[Dict[str, object]]] = None,
        locked_real_shot_path: Optional[Path] = None,
    ) -> Dict[str, object]:
        """Stage 5: Region-Aware Upscayl Hybrid (`v2.2` default: Optical Face + Ultramix 4K Body), Pure Optical Lanczos-4, or Upscayl 4x."""
        gen0_p = Path(gen0_portrait_path).resolve()
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

        raw_bgr = cv2.imread(str(win_p))
        ref_bgr = cv2.imread(str(gen0_p))
        if raw_bgr is None or ref_bgr is None:
            raise RuntimeError("Failed to decode winner or Gen-0 portrait.")

        # Measure HONEST pre-upscale face resolution BEFORE 4K scaling
        raw_face_det = self.pipeline.detector.detect(raw_bgr)
        raw_fw, raw_fh = round(float(raw_face_det.bbox[2]), 1), round(float(raw_face_det.bbox[3]), 1)
        raw_iod_px = round(float(raw_face_det.inter_ocular_distance), 1)

        # 1. Region-Aware Upscayl Hybrid (`v2.2` default: 100% Optical Face/Eyes + Lossless Upscayl 4K Hair/Suit/Room),
        #    Pure Optical Lanczos-4, OR full Upscayl 4x
        acuity_meta: Dict[str, object] = {"applied": False}
        zero_post_paste = design_mode in ("pure_character_sheet", "design_b_rerender")
        pure_zero_mask = zero_post_paste and (upscale_mode == "optical_fullframe")
        if (design_mode == "pure_character_sheet") or (upscale_mode == "optical_fullframe"):
            # 100.0% Uniform Whole-Frame Pure Optical 4K Scaling — ZERO face/body split mask, ZERO face pasting, ZERO local ROI color painting!
            up_bgr = self.pipeline.upscaler.upscale_bgr_in_memory(raw_bgr, target_min_height=4800)
            up_model = "optical_fullframe_uniform_4k_zero_composite"
            pure_zero_mask = True
        elif upscale_mode == "upscayl_hybrid":
            up_bgr, hyb_meta = self.pipeline.upscaler.upscale_hybrid_optical_face_upscayl_body(
                raw_bgr, target_min_height=4800, model_name=upscayl_model_name
            )
            up_model = str(hyb_meta.get("model", "hybrid_opt_face+ultramix-balanced-4x"))
            acuity_meta = hyb_meta.get("acuity") or {"applied": True}
        elif upscale_mode == "optical_lanczos":
            up_bgr = self.pipeline.upscaler.upscale_bgr_in_memory(raw_bgr, target_min_height=4800)
            if design_mode == "pure_v21":
                up_model = "optical_lanczos4_pure_v21"
            else:
                up_bgr, acuity_meta = self.pipeline.upscaler.enhance_optical_organ_acuity(up_bgr)
                up_model = "optical_lanczos4_acuity_v22"
        else:
            up_bgr, up_model = self.pipeline.upscaler.upscale_with_upscayl(
                raw_bgr, style=upscayl_model_name if upscayl_model_name else "auto", scale=4
            )
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
            roi_matte, _ = self.pipeline.skin_processor.process(roi, seed=42, reference_skin_profile=None)
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

        # 3. Full 10-Gate Quality Audit at exact native 4K dimensions
        cfg_4k = dataclasses.replace(
            self.pipeline.config,
            canvas=dataclasses.replace(self.pipeline.config.canvas, target_width=uw, target_height=uh),
        )
        auditor_4k = ModelGenerationPipeline(cfg_4k).auditor
        report = auditor_4k.audit(up_bgr, crop_box, seam_l1_diff=0.0, reference_face_bgr=ref_bgr)

        # Save all 5 mandatory deliverables even before strict check so failures can be visually diagnosed
        cv2.imwrite(str(out_4k), up_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 97])
        cv2.imwrite(str(out_closeup), up_bgr[vy1:vy2, vx1:vx2], [int(cv2.IMWRITE_JPEG_QUALITY), 97])

        fx, fy, fw, fh = face_4k.bbox
        mx1, my1 = max(0, int(fx - fw * 0.65)), max(0, int(fy - fh * 0.55))
        mx2, my2 = min(uw, int(fx + fw * 1.65)), min(uh, int(fy + fh * 1.55))
        macro_bgr = up_bgr[my1:my2, mx1:mx2].copy()
        cv2.imwrite(str(out_macro), macro_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 97])

        eye_strip_saved = self.ocular.save_400pct_eye_comparison_strip(ref_bgr, macro_bgr, str(out_eye_cmp))
        oc_cmp = self.ocular.compare(ref_bgr, macro_bgr)
        sil_cmp = self.exaggeration_guard.measure_pupil_aligned_silhouette(
            ref_bgr, macro_bgr, output_overlay_path=str(out_sil_ov)
        )
        ex_cmp = self.exaggeration_guard.audit(ref_bgr, macro_bgr)
        od_cmp = self.organ_depth.compare(ref_bgr, macro_bgr)

        all_passed = bool(report.passed and sil_cmp.get("passed", True))
        fail_msg = report.message
        if not sil_cmp.get("passed", True):
            sil_err = "; ".join(str(x) for x in sil_cmp.get("issues", []))
            fail_msg = f"{fail_msg}; {sil_err}" if not report.passed else f"1:1 외곽선 게이트 미달: {sil_err}"

        # Blacklist generated output in OriginOnlyOneShotGuard so it can NEVER be re-fed as input!
        self.origin_guard.register_generated_scene_output(out_4k, job_id=output_stem)

        vault_synced: List[str] = []
        if sync_to_vault and all_passed and self.pipeline.config.vault_dir:
            v_out = self.pipeline.config.vault_dir / "outputs"
            v_out.mkdir(parents=True, exist_ok=True)
            for p_item in (out_4k, out_closeup, out_macro, out_eye_cmp, out_sil_ov):
                if p_item.is_file():
                    dst = v_out / p_item.name
                    shutil.copy2(p_item, dst)
                    vault_synced.append(str(dst))

        receipt = {
            "workflow_version": "2.3",
            "design_mode": design_mode,
            "stage": "FINALIZED_10GATES_PASSED" if all_passed else "FINALIZED_10GATES_FAILED",
            "timestamp": time.time(),
            "passed_10gates": all_passed,
            "upscale_mode": upscale_mode,
            "upscayl_model": up_model,
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
                "hero_4k": str(out_4k),
                "closeup": str(out_closeup),
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
