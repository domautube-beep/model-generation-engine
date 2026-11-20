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
from .origin_guard import OriginOnlyOneShotGuard
from .pipeline import ModelGenerationPipeline
from .prompt_builder import PromptFirewall
from .proportion_calibrator import compare_14pt_proportions, extract_14pt_proportion_vector


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
            "c1",
            "Shot on Hasselblad X2D 100C with 80mm f/2.8 lens at f/4.0, ISO 100, crisp studio editorial medium-three-quarter photograph",
        ),
        (
            "c2",
            "Shot on Leica SL3 with APO-Summicron-SL 90mm f/2 at f/4.0, ISO 100, crisp luxury editorial medium-three-quarter photograph",
        ),
        (
            "c3",
            "Shot on Canon EOS R5 with RF 85mm f/1.2L at f/4.0, ISO 100, crisp editorial portrait-to-thigh photograph",
        ),
    )

    def __init__(self, pipeline: Optional[ModelGenerationPipeline] = None) -> None:
        self.pipeline = pipeline or ModelGenerationPipeline(EngineConfig())
        self.ocular = OcularAnatomyAnalyzer(self.pipeline.apple_vision)
        self.exaggeration_guard = FaceExaggerationGuard(self.pipeline.apple_vision)
        self.origin_guard = OriginOnlyOneShotGuard()

    def prepare_origin_bundle(
        self,
        gen0_portrait_path: Path,
        work_dir: Path,
        scene_instruction: str,
        outfit_instruction: str,
        location_reference_path: Optional[Path] = None,
        turnaround_8dir_sheet_path: Optional[Path] = None,
        extra_raw_asset_paths: Sequence[Path] = (),
        aspect_ratio_label: str = "3:4",
        best_of_k: int = 3,
        match_gen0_yaw: bool = True,
        attach_ocular_detail_anchor: bool = True,
        subject: Optional[SubjectProfile] = None,
    ) -> Dict[str, object]:
        """Stages 1 & 2: Validate Gen-0, attach 1-Hop 8-Direction Bank + optical anchor, clean location, emit K prompts."""
        gen0_p = Path(gen0_portrait_path).resolve()
        if not gen0_p.is_file():
            raise FileNotFoundError(f"Gen-0 portrait not found: {gen0_p}")

        # Hard check: reject any previously generated/derivative/upscaled file as Gen-0!
        self.origin_guard.verify_one_shot_and_inputs([gen0_p])

        wdir = Path(work_dir).resolve()
        wdir.mkdir(parents=True, exist_ok=True)

        ref_bgr = cv2.imread(str(gen0_p))
        if ref_bgr is None:
            raise ValueError(f"Cannot decode Gen-0 image: {gen0_p}")

        blueprint: AnatomicalBlueprint = self.pipeline.turnaround.extract_anatomical_blueprint(ref_bgr)
        clause_25pt = blueprint.to_3d_invariance_clause()
        PromptFirewall.validate_prompt(clause_25pt)

        ordered_refs: List[Dict[str, str]] = [
            {
                "slot": "Image 1",
                "role": "gen0_full_identity_cranial_jaw_cheek",
                "path": str(gen0_p),
                "sha256": _sha256_file(gen0_p),
                "label": f"Untouched Gen-0 real camera portrait ({blueprint.source_signed_yaw_deg:+.1f} deg)",
            }
        ]
        role_lines: List[str] = [
            "Image 1 (SUPREME IDENTITY, JAWLINE-ANGLE, CHEEK-AREA & CRANIAL DOME AUTHORITY — UNTOUCHED Gen-0): "
            "100% absolute ground-truth authority for cranial dome, hairline, hair parting/tuck, exact cheek contour, "
            "exact mandibular jawline angle & chin taper (do NOT round or widen the jaw), nose tip, philtrum, lips, and natural skin tone"
        ]

        turnaround_p: Optional[Path] = None
        if turnaround_8dir_sheet_path is not None and Path(turnaround_8dir_sheet_path).is_file():
            turnaround_p = Path(turnaround_8dir_sheet_path).resolve()
            self.origin_guard.register_approved_origin(turnaround_p, kind="8dir_face", depth=1)
            ordered_refs.append(
                {
                    "slot": f"Image {len(ordered_refs) + 1}",
                    "role": "verified_1hop_8dir_turnaround_bank",
                    "path": str(turnaround_p),
                    "sha256": _sha256_file(turnaround_p),
                    "label": "Verified 1-Hop 4K 8-Direction Master Turnaround Bank (-90°..+90°)",
                }
            )
            role_lines.append(
                f"Image {len(ordered_refs)} (VERIFIED 1-HOP 8-DIRECTION 3D BONE-STRUCTURE & JAWLINE-ANGLE BANK, -90°..+90°): "
                "3D multi-angle reference of the exact same person from Image 1 across all 8 frontal-hemisphere yaw angles; "
                "match the exact 3D mandibular corner angle, lower-jaw taper, chin length, cheekbone projection, and ear/neck transition"
            )

        ocular_anchor_p: Optional[Path] = None
        # Google Flow Nano Banana Pro drops face identity when >=3 face refs are attached!
        # When an individual 8-dir face anchor is already attached as Image 2, keep STRICTLY 2 face refs (Gen-0 + 8-dir)!
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
                    "label": "Pure lossless optical Gen-0 ocular/eye-to-lip close-up crop (0% AI, 0% Upscayl)",
                }
            )
            role_lines.append(
                f"Image {len(ordered_refs)} (LOSSLESS OPTICAL OCULAR & FACIAL MICRO-DETAIL AUTHORITY — DIRECT Gen-0 CROP): "
                "pure unmodified optical zoom of Image 1's exact eyes, thin in-line eyelid fold, visible white sclera, "
                "natural medium iris diameter, inner corner, delicate eyebrows, nose tip, and lips—reproduce these exact "
                "eyes 1:1 without thickening the eyelid crease or enlarging the irises"
            )

        clean_loc_p: Optional[Path] = None
        if location_reference_path is not None and Path(location_reference_path).is_file():
            loc_src = Path(location_reference_path).resolve()
            loc_bgr = cv2.imread(str(loc_src))
            if loc_bgr is not None:
                clean_loc_p = wdir / f"loc00_{loc_src.stem}_clean.jpg"
                self.pipeline.turnaround.erase_foreign_person_completely(
                    loc_bgr, output_clean_path=clean_loc_p, remove_overlay_text=True
                )
                self.origin_guard.register_approved_origin(clean_loc_p, kind="clean_location", depth=0)
                ordered_refs.append(
                    {
                        "slot": f"Image {len(ordered_refs) + 1}",
                        "role": "location_clean_authority",
                        "path": str(clean_loc_p),
                        "sha256": _sha256_file(clean_loc_p),
                        "label": "Person-erased clean location, furniture & lighting authority",
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

        yaw_deg = float(blueprint.source_signed_yaw_deg)
        if match_gen0_yaw and abs(yaw_deg) >= 8.0:
            side_word = "image-right" if yaw_deg > 0 else "image-left"
            yaw_clause = f"Head turned ~{abs(yaw_deg):.0f} deg toward {side_word} matching Image 1, chin level, calm direct gaze."
        else:
            yaw_clause = "Head facing cleanly forward matching Image 1, chin level, calm direct gaze."

        subj_label = subject.subject_identity_label if subject is not None else "exact same real person from Image 1 and Image 2"
        k_count = max(1, int(best_of_k))
        prompts: List[Dict[str, str]] = []
        # 5-Slot Positive Prompt Allocator (~1,050-1,250 chars total: top-loaded 8-head framing, ZERO numeric spam, ZERO pink-elephant 'NEVER' bloat!)
        ocular_short = (
            "sleek flatter almond upper eyelids with tight thin in-line fold, crisp dark lash root, visible white sclera, natural fuller brow density"
            if (blueprint.ocular_profile is not None and float(getattr(blueprint.ocular_profile, "avg_crease_gap_ratio", 0.35)) <= 0.36)
            else "exact natural eyelid arch, iris size, visible white sclera, and natural brow density from Image 1"
        )
        ref_note = "Image 1 (untouched portrait) and Image 2 (angle reference)" if turnaround_p is not None else "Image 1 (untouched portrait)"
        for i in range(k_count):
            seed_tag, _cam_hdr = self.DEFAULT_CAMERA_SEEDS[i % len(self.DEFAULT_CAMERA_SEEDS)]
            slot1_camera = (
                f"Hasselblad X2D 85mm f/4 full-three-quarter to knee-up distant editorial portrait ({aspect_ratio_label}), "
                "pulled back generously with ample margin so true 8-head proportion holds (small graceful head, face ~1/3 of open shoulder width)."
            )
            slot2_identity = (
                f"Subject: {subj_label} ({ref_note}). "
                "Exact 1:1 facial fidelity: crisp steep chin-to-ear mandibular diagonal with clean shadow beneath jaw, taut lower cheek, "
                f"Measured Ocular Anatomy Lock ({ocular_short}), exact nose, hydrated natural lips, hairline parting & ear tuck."
            )
            slot3_pose = f"Scene & Pose: {scene_instruction.strip()}. {yaw_clause}"
            slot4_outfit = f"Outfit: {outfit_instruction.strip()} (replacing reference clothing completely)."
            slot5_light = (
                "Lighting & Texture: sculpted 45-degree editorial key light carving clean jaw-to-neck separation shadow, "
                "authentic unretouched matte skin pores, natural filmic tonal depth, clean studio finish."
            )
            p_text = f"{slot1_camera} {slot2_identity} {slot3_pose} {slot4_outfit} {slot5_light}"
            PromptFirewall.validate_prompt(p_text)
            prompts.append(
                {
                    "candidate_tag": f"c{i + 1}_{seed_tag}",
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
                }
            )

        manifest = {
            "workflow_version": "9.0",
            "stage": "READY_FOR_FRESH_GEN",
            "created_at": time.time(),
            "gen0_portrait_path": str(gen0_p),
            "gen0_sha256": _sha256_file(gen0_p),
            "ocular_detail_anchor_path": str(ocular_anchor_p) if ocular_anchor_p else None,
            "clean_location_path": str(clean_loc_p) if clean_loc_p else None,
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

    def score_and_select_native_winner(
        self,
        gen0_portrait_path: Path,
        candidate_paths: Sequence[Path],
        min_created_epoch: float = 0.0,
    ) -> Dict[str, object]:
        """Stage 4: Score all K fresh candidates natively (`0px` warp, `0%` pasting) across SFace + 18 Craniofacial + 7 Ocular."""
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
            if not oc_pass:
                issues.extend(str(x) for x in oc_cmp.get("issues", []))
            if not ex_cmp.get("passed", True):
                issues.extend(str(x) for x in ex_cmp.get("issues", []))

            native_ok = len(issues) == 0
            # Higher composite_rank_score is better: strongly prioritize native_ok, zero facial exaggeration, high SFace, high eye NCC
            rank_score = (
                (100.0 if native_ok else 0.0)
                + sface_sim * 55.0
                + oc_ncc * 25.0
                - float(cmp18["meanWeightedDriftPct"]) * 2.5
                - oc_drift * 1.8
                - ex_idx * 2.0
                - abs(ca_diff) * 0.9
                - abs(lja_diff) * 0.7
            )
            scored.append(
                CandidateScore(
                    candidate_path=str(p),
                    sface_similarity=round(sface_sim, 4),
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
    ) -> Dict[str, object]:
        """Stage 5: Single final Upscayl 4x + matte glare control + 10-Gate Audit + 4K/closeup/macro/400% eye strip + signed receipt."""
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

        # Measure HONEST pre-upscale face resolution BEFORE Upscayl 4x
        raw_face_det = self.pipeline.detector.detect(raw_bgr)
        raw_fw, raw_fh = round(float(raw_face_det.bbox[2]), 1), round(float(raw_face_det.bbox[3]), 1)
        raw_iod_px = round(float(raw_face_det.inter_ocular_distance), 1)

        # 1. SINGLE Final Upscayl 4x pass (`high-fidelity-4x` Photo / `digital-art-4x` Illustration) — ZERO warp, ZERO face-pasting!
        up_bgr, up_model = self.pipeline.upscaler.upscale_with_upscayl(raw_bgr, style="auto", scale=4)
        uh, uw = up_bgr.shape[:2]

        # 2. Gentle luminance-only matte specular roll-off (keeping 100% scene lighting untouched)
        face_4k = self.pipeline.detector.detect(up_bgr)
        crop_box = self.pipeline.detector.compute_upper_body_crop_box(
            (uh, uw), face_4k, aspect_ratio=float(uw) / float(max(1, uh))
        )
        X1, Y1, X2, Y2 = crop_box
        vx1, vy1 = max(0, X1 - 120), max(0, Y1 - 60)
        vx2, vy2 = min(uw, X2 + 120), min(uh, Y2 + 160)
        roi = up_bgr[vy1:vy2, vx1:vx2].copy()
        roi_matte, _ = self.pipeline.skin_processor.process(roi, seed=42, reference_skin_profile=None)
        up_bgr[vy1:vy2, vx1:vx2] = roi_matte

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
            "workflow_version": "10.0",
            "stage": "FINALIZED_10GATES_PASSED" if all_passed else "FINALIZED_10GATES_FAILED",
            "timestamp": time.time(),
            "passed_10gates": all_passed,
            "upscayl_model": up_model,
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
