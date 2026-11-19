"""Universal Multi-Asset & Multi-Location Scene Batch Composer (`multi_asset_composer.py`).

Supports arbitrary projects and combinations without hardcoding:
- Subject Identity: Untouched Gen-0 Canonical Front Anchor + Approved 8-Angle Bank (`0°..315°`)
- Multi-Category Styling Attachments (all optional, multiple files per slot):
  - `garments`: Clothing photos/notes (outerwear, top, bottom, suit, dress)
  - `accessories`: Eyewear, watch, tie, scarf, hat, jewelry, belt
  - `bags`: Briefcase, tote, shoulder bag, handbag, backpack
  - `shoes`: Dress shoes, loafers, sneakers, boots, heels
- Multi-Location & Style Reference Batching (`locations`):
  - Accepts 1..N location/mood/style reference images and/or text prompts.
  - Uploading N locations automatically expands into N parallel scene generation jobs (`여러 장 넣으면 생성도 여러 장`).
  - Every uploaded garment, accessory, bag, shoe, and location/style image automatically has any foreign human faces
    neutral-grey-masked via `RealPersonTurnaroundEngine.mask_foreign_faces_in_reference`.
- 3D Previz Pose Conditioning (`poses`):
  - Renders or attaches the 3D Previz Mannequin Guide (`PosePrevizSpec`) and automatically routes the exact matching
    angle from the 8-Angle Turnaround Bank (`resolve_8angle_view_from_azimuth`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

from .config import SubjectProfile
from .identity_turnaround import AnatomicalBlueprint, RealPersonTurnaroundEngine
from .previz_pose import PosePrevizSpec, Previz3DRenderer
from .prompt_builder import PromptFirewall

ITEM_CATEGORIES: Tuple[str, ...] = ("garment", "accessory", "bag", "shoe")


@dataclass
class AttachmentItem:
    """Single user-attached styling item (garment, accessory, bag, or shoe)."""

    category: str  # "garment" | "accessory" | "bag" | "shoe"
    image_path: Optional[Path] = None
    description: str = ""
    sub_role: str = ""  # e.g. "outerwear", "top", "bottom", "watch", "tote_bag", "loafers"

    def __post_init__(self) -> None:
        if self.category not in ITEM_CATEGORIES:
            raise ValueError(f"Unsupported item category '{self.category}'. Expected one of {ITEM_CATEGORIES}.")


@dataclass
class SceneLocationInput:
    """Single target location or mood/style reference (uploading N creates N generation jobs)."""

    name: str = "scene"
    image_path: Optional[Path] = None
    mood_or_place_prompt: str = ""
    lighting_note: str = ""
    scope: str = "scene_and_mood"  # "scene_and_mood" (follow space+light+tone) or "style_mood_only" (new space, same tone & light)


@dataclass
class ComposedSceneJob:
    """Fully resolved, firewall-validated single generation job ready for execution."""

    job_index: int
    job_tag: str
    matched_8angle_view: str
    head_azimuth_deg: float
    prompt: str
    ordered_reference_images: List[Dict[str, str]] = field(default_factory=list)
    previz_guide_path: Optional[str] = None
    location_name: str = ""


class MultiSceneBatchComposer:
    """Expands (Subject 8-Angle Bank × Styling Items × N Locations × M Previz Poses) into parallel jobs."""

    def __init__(self, turnaround_engine: RealPersonTurnaroundEngine) -> None:
        self.turnaround = turnaround_engine

    def _prepare_masked_copy(self, src_path: Path, work_dir: Path, prefix: str) -> Tuple[Path, int]:
        bgr = cv2.imread(str(src_path))
        if bgr is None:
            raise FileNotFoundError(f"Cannot read attachment image: {src_path}")
        out_path = work_dir / f"{prefix}_{src_path.stem}_masked.jpg"
        _, masked_count = self.turnaround.mask_foreign_faces_in_reference(bgr, output_masked_path=out_path)
        return (out_path if masked_count > 0 else src_path), masked_count

    def compose_batch(
        self,
        canonical_front_path: Path,
        subject: SubjectProfile,
        locations: Sequence[SceneLocationInput],
        poses: Sequence[PosePrevizSpec],
        angle_bank: Optional[Dict[str, Path]] = None,
        items: Sequence[AttachmentItem] = (),
        work_dir: Optional[Path] = None,
        aspect_ratio_label: str = "16:9",
        attach_previz_image: bool = True,
    ) -> List[ComposedSceneJob]:
        """Compose `len(locations) * len(poses)` independent, firewall-validated scene generation jobs."""
        if not locations:
            raise ValueError("At least one location or mood specification is required.")
        if not poses:
            poses = [PosePrevizSpec()]

        front_bgr = cv2.imread(str(canonical_front_path))
        if front_bgr is None:
            raise FileNotFoundError(f"Cannot read canonical front anchor: {canonical_front_path}")
        blueprint: AnatomicalBlueprint = self.turnaround.extract_anatomical_blueprint(front_bgr)

        stage_dir = Path(work_dir) if work_dir is not None else Path(canonical_front_path).parent / ".batch_staging"
        stage_dir.mkdir(parents=True, exist_ok=True)
        bank = angle_bank or {}

        # Pre-mask foreign faces in all attached styling items once
        prepared_items: List[Tuple[AttachmentItem, Optional[Path], int]] = []
        for idx, it in enumerate(items):
            if it.image_path is not None and Path(it.image_path).exists():
                m_path, cnt = self._prepare_masked_copy(Path(it.image_path), stage_dir, f"item{idx:02d}_{it.category}")
                prepared_items.append((it, m_path, cnt))
            else:
                prepared_items.append((it, None, 0))

        jobs: List[ComposedSceneJob] = []
        job_counter = 0

        for loc_idx, loc in enumerate(locations):
            loc_img_prepared: Optional[Path] = None
            loc_masked_cnt = 0
            if loc.image_path is not None and Path(loc.image_path).exists():
                loc_img_prepared, loc_masked_cnt = self._prepare_masked_copy(
                    Path(loc.image_path), stage_dir, f"loc{loc_idx:02d}"
                )

            for pose_idx, pose in enumerate(poses):
                job_counter += 1
                job_tag = f"scene_{loc_idx + 1:02d}_pose_{pose_idx + 1:02d}"

                # 1. Select [Untouched Gen-0 Front Anchor, Angle-Matched 8-Way View]
                id_refs, matched_view = self.turnaround.select_angle_matched_references(
                    canonical_front_path=canonical_front_path,
                    angle_bank=bank,
                    target_azimuth_deg=pose.head_azimuth_deg,
                )

                ordered_imgs: List[Dict[str, str]] = []
                role_clauses: List[str] = []

                # Image 1: Untouched Gen-0 Front Identity
                ordered_imgs.append(
                    {
                        "path": str(id_refs[0]),
                        "role": "identity_primary_front",
                        "label": "Primary untouched front identity anchor (exact face, true iris color, authentic skin texture)",
                    }
                )
                role_clauses.append(
                    "Image 1 (IDENTITY-PRIMARY-FRONT): untouched original photograph of the target subject—sole authority "
                    "for true facial identity, authentic iris color, skin texture, eyewear frame material, and hair color"
                )

                # Image 2 (optional): Angle-matched view from the 8-Angle Turnaround Bank
                if len(id_refs) > 1:
                    ordered_imgs.append(
                        {
                            "path": str(id_refs[1]),
                            "role": f"identity_angle_{matched_view}",
                            "label": f"Approved {matched_view} turnaround anchor ({pose.head_azimuth_deg:.0f} deg)",
                        }
                    )
                    role_clauses.append(
                        f"Image {len(ordered_imgs)} (IDENTITY-8ANGLE-{matched_view.upper()}): approved {matched_view} turnaround view "
                        "of the exact same subject—use strictly for 3D skull/nasal/cheek/ear/hair/eyewear-temple silhouette at this angle "
                        "(do NOT copy its grey studio background or studio suit)"
                    )

                # Styling items (Garments, Accessories, Bags, Shoes)
                styling_text_clauses: List[str] = []
                for it, m_path, m_cnt in prepared_items:
                    cat_title = {
                        "garment": "WARDROBE-GARMENT",
                        "accessory": "STYLING-ACCESSORY",
                        "bag": "CARRIED-BAG",
                        "shoe": "FOOTWEAR-SHOES",
                    }[it.category]
                    sub_note = f" ({it.sub_role})" if it.sub_role else ""
                    desc_note = f": {it.description.strip()}" if it.description.strip() else ""
                    if m_path is not None:
                        ordered_imgs.append(
                            {
                                "path": str(m_path),
                                "role": f"item_{it.category}",
                                "label": f"{cat_title}{sub_note}{desc_note}",
                            }
                        )
                        mask_hint = (
                            " (any original wearer's face is covered by a grey box; ignore the grey box and original wearer)"
                            if m_cnt > 0
                            else ""
                        )
                        role_clauses.append(
                            f"Image {len(ordered_imgs)} ({cat_title}{sub_note}): exact product design, material weave, color, seams, "
                            f"hardware, and silhouette to wear/carry{desc_note}{mask_hint}"
                        )
                    if it.description.strip():
                        styling_text_clauses.append(f"{it.category}{sub_note}: {it.description.strip()}")

                # Location / Mood Style Reference
                loc_clause = ""
                if loc_img_prepared is not None:
                    ordered_imgs.append(
                        {
                            "path": str(loc_img_prepared),
                            "role": "location_or_mood_reference",
                            "label": f"Location / Style Reference ({loc.name})",
                        }
                    )
                    mask_hint = (
                        " (any foreign person's face in this reference is covered by a grey box—do NOT render any grey box or foreign person)"
                        if loc_masked_cnt > 0
                        else ""
                    )
                    if loc.scope == "style_mood_only":
                        role_clauses.append(
                            f"Image {len(ordered_imgs)} (STYLE-MOOD-REFERENCE): extract its exact 16-point tone & manner—overall brightness, "
                            "key-light direction, shadow softness, color temperature, color grading, surface reflection, depth of field, and editorial mood—"
                            f"while directing a fresh spatial composition{mask_hint}"
                        )
                    else:
                        role_clauses.append(
                            f"Image {len(ordered_imgs)} (LOCATION-AND-TONE-REFERENCE): place the subject naturally inside this environment's "
                            "architecture, spatial depth, directional lighting, shadow falloff, color temperature, ambient color spill, and overall "
                            f"tone & manner{mask_hint}"
                        )

                if loc.mood_or_place_prompt.strip():
                    loc_clause += f" Environment & Mood Direction: {loc.mood_or_place_prompt.strip()}."
                if loc.lighting_note.strip():
                    loc_clause += f" Lighting & Atmosphere: {loc.lighting_note.strip()}."
                if not loc_clause and loc_img_prepared is None:
                    loc_clause = " Environment & Mood Direction: clean contemporary architectural editorial setting with natural directional key light."

                # 3D Previz Pose Guide Image
                previz_path: Optional[Path] = None
                if attach_previz_image:
                    previz_path = stage_dir / f"previz_{job_tag}.jpg"
                    Previz3DRenderer.render_pose_guide(pose, canvas_size=(1280, 720), output_path=previz_path)
                    ordered_imgs.append(
                        {
                            "path": str(previz_path),
                            "role": "pose_previz_guide",
                            "label": f"3D Previz Pose Rig ({pose.head_azimuth_deg:.0f} deg)",
                        }
                    )
                    role_clauses.append(
                        f"Image {len(ordered_imgs)} (3D-PREVIZ-POSE-GUIDE): structural 3D mannequin pose blueprint ONLY—match its "
                        "exact camera placement, torso angle, head orientation, arm/hand positions, and leg stance on the real human subject; "
                        "NEVER draw stick-figure lines, mannequin spheres, arrows, or dark grid backgrounds in the final photograph"
                    )

                wardrobe_summary = (
                    "; ".join(styling_text_clauses)
                    if styling_text_clauses
                    else (
                        "wear/carry every attached wardrobe, accessory, bag, and footwear item with 1:1 product fidelity"
                        if prepared_items
                        else "contemporary tailored editorial attire harmonizing with the environment"
                    )
                )

                prompt = (
                    "Editorial portrait. Shot on a medium-format 100MP camera with prime optics, tack-sharp optical micro-contrast, "
                    "deep true blacks, and zero atmospheric haze or milky veil. "
                    "CRITICAL MANDATORY RULE — REAL IDENTITY LOCK + 16-POINT SCENE TONE & MANNER HARMONY: "
                    f"Create ONE photorealistic {aspect_ratio_label} photograph of the {subject.subject_identity_label} from Image 1. "
                    "Attached Reference Roles: " + "; ".join(role_clauses) + ". "
                    f"Identity & Eyewear Continuity: Preserve {subject.facial_identity_descriptor()}. "
                    "Keep exact 1:1 facial anatomy, authentic iris color from Image 1 (zero green or purple lens-coating tint), "
                    "and if eyewear is worn, render both temple arms continuously from the outer front hinges over the ears. "
                    f"{blueprint.to_3d_invariance_clause()} "
                    f"Physical Build: {subject.body_frame_descriptor()}, maintaining balanced adult 7.5-to-8-head head-to-shoulder proportions. "
                    f"Pose & Camera Rig: {pose.to_prompt_clause()}. "
                    f"Wardrobe, Accessories, Bag & Footwear: {wardrobe_summary}. "
                    f"16-Point Tone & Manner Integration:{loc_clause} Harmonize key-light direction, shadow softness, floor/furniture contact shadows, "
                    "ambient wall color bounce, fabric surface reflectance, and depth of field so the subject, outfit, and space read as one "
                    "authentic single-camera photograph. "
                    "Skin Expression: authentic natural real skin texture with subtle pores, zero plastic/oily shine, zero dewy gloss, "
                    "zero specular glare, zero artificial wrinkles, zero text, and zero watermarks."
                )
                PromptFirewall.validate_prompt(prompt)

                jobs.append(
                    ComposedSceneJob(
                        job_index=job_counter,
                        job_tag=job_tag,
                        matched_8angle_view=matched_view,
                        head_azimuth_deg=pose.head_azimuth_deg,
                        prompt=prompt,
                        ordered_reference_images=ordered_imgs,
                        previz_guide_path=str(previz_path) if previz_path else None,
                        location_name=loc.name,
                    )
                )

        return jobs
