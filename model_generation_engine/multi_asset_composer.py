"""Universal `1원본들 → 1생성` Multi-Asset, 8-Direction Face Bank & Multi-Location Batch Composer (`multi_asset_composer.py` v3.3).

Enforces the ironclad **`1원본들 → 1생성` (One-Originals-Set to One-Generation)** contract:
1. **8-Direction Face Bank ALWAYS Attached**:
   - Attaches the subject's approved **8-Direction Face Anchors** (`Exact Angle-Matched 8-Direction Face` + `Untouched 0° Original Face`
     + `Complete 8-Direction Master Turnaround Sheet` / available 8-direction bank views) directly into the single generation call!
2. **100% Raw Original Assets ONLY (ZERO Derived Collages, ZERO Previz Wireframe JPEGs, ZERO Re-Feeding)**:
   - Every garment (`inner_top`, `mid_layer`, `outerwear`, `bottom`, `one_piece`), accessory (`watch`, `eyewear`, `jewelry`, `belt`, `hat`),
     bag (`micro_mini .. large_tote_weekender` × 11 `carry_mode`s), shoe, and location is attached as its **original asset file**
     (with foreign human faces grey-masked only if a foreign model's face was detected inside a lookbook/mood photo).
   - Validated by `OriginOnlyOneShotGuard` before execution (`Depth == 1` atomic single shot).
3. **Auto Face-Direction Inference by Default (`face_direction_mode = "auto"`), Manual Override on Demand**:
   - Automatically detects the ideal head direction (`-90°..+90°`) per location from reference pose or key-light direction unless
     the user explicitly overrides the direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

from .config import SubjectProfile
from .identity_turnaround import (
    LEGACY_VIEW_ALIASES,
    TURNAROUND_FULL_SWEEP,
    VIEW_AZIMUTH_DEG,
    VIEW_LABELS_KO,
    AnatomicalBlueprint,
    RealPersonTurnaroundEngine,
    resolve_8angle_view_from_azimuth,
)
from .origin_guard import OriginOnlyOneShotGuard
from .previz_pose import PosePrevizSpec, Previz3DRenderer
from .prompt_builder import PromptFirewall

ITEM_CATEGORIES: Tuple[str, ...] = ("garment", "accessory", "bag", "shoe")

GARMENT_SLOTS: Dict[str, str] = {
    "inner_top": "INNER BASE TOP (t-shirt / dress shirt / turtleneck worn closest to skin)",
    "mid_layer": "MIDDLE LAYER (knitwear / vest / cardigan layered over inner top)",
    "outerwear": "OUTERWEAR LAYER (jacket / blazer / coat / overshirt worn as outermost top layer)",
    "bottom": "LOWER-BODY TROUSERS / SKIRT",
    "one_piece": "FULL SUIT / DRESS / ONE-PIECE ENSEMBLE",
}

WEARING_STYLES: Dict[str, str] = {
    "auto": "",
    "open_layered": "worn unbuttoned/open at the front so the inner shirt/t-shirt/knit underneath is clearly visible",
    "closed_buttoned": "worn neatly buttoned/fastened at the front",
    "draped_shoulders": "draped effortlessly over both shoulders",
    "sleeves_rolled": "with sleeves casually rolled/pushed slightly up the forearm",
    "tucked_in": "neatly tucked into the waistband of the trousers/skirt",
    "untucked_relaxed": "worn untucked with a clean natural hem drape over the waist",
}

ACCESSORY_SLOTS: Dict[str, Tuple[str, str]] = {
    "eyewear": (
        "EYEWEAR / GLASSES",
        "1:1 cranial temple-to-temple width fit, bridge resting naturally on nasal bone, both temple arms extending continuously over ears",
    ),
    "watch_left": (
        "WRISTWATCH (LEFT WRIST)",
        "authentic horological proportion (38–42mm case diameter = ~68% of human wrist width, never oversized), strapped snugly around the left wrist bone just past the sleeve cuff",
    ),
    "watch_right": (
        "WRISTWATCH (RIGHT WRIST)",
        "authentic horological proportion (38–42mm case diameter = ~68% of human wrist width, never oversized), strapped snugly around the right wrist bone just past the sleeve cuff",
    ),
    "necklace_tie_scarf": (
        "NECKWEAR (TIE / NECKLACE / SCARF)",
        "proportioned naturally inside the collar/clavicle V-zone without exaggerated pendant or knot scale",
    ),
    "jewelry_fine": (
        "FINE JEWELRY (EARRING / RING / BRACELET)",
        "delicate real-world millimeter jewelry scale (1.0–2.5cm relative to earlobe/finger width, zero chunky toy-like inflation)",
    ),
    "belt": (
        "WAIST BELT",
        "standard 3.0–3.5cm strap width threaded cleanly through trouser belt loops with realistic buckle proportion",
    ),
    "hat": (
        "HEADWEAR / HAT",
        "fitted naturally to cranial crown circumference without squashing hair/forehead proportions",
    ),
}

BAG_SIZES: Dict[str, Tuple[str, str, float]] = {
    "micro_mini": (
        "MICRO / MINI BAG (~18–22cm width)",
        "compact mini scale—roughly 2x palm width and smaller than the subject's head height (~20cm); NEVER enlarge into a tote",
        0.55,
    ),
    "small_compact": (
        "SMALL COMPACT BAG (~24–28cm width)",
        "compact A5 everyday scale (~26cm wide, slightly narrower than torso profile)",
        0.78,
    ),
    "medium_standard": (
        "MEDIUM / BRIEFCASE BAG (~34–38cm width)",
        "balanced A4 briefcase/medium tote scale (~36cm wide, roughly matching side torso breadth; ~1.6x head height)",
        1.0,
    ),
    "large_tote_weekender": (
        "LARGE OVERSIZED TOTE / WEEKENDER (~45–52cm width)",
        "generous travel/shopper scale (~48cm wide, wider than torso profile with realistic gravity sag)",
        1.35,
    ),
}

BAG_CARRY_MODES: Dict[str, str] = {
    "hand_right": "held naturally in the RIGHT hand by its top handle(s), arm relaxed downward so the bag hangs beside the right hip/upper thigh with realistic weight tension",
    "hand_left": "held naturally in the LEFT hand by its top handle(s), arm relaxed downward so the bag hangs beside the left hip/upper thigh with realistic weight tension",
    "crook_right": "hooked gracefully over the crook of the RIGHT forearm/elbow, hand relaxed forward",
    "crook_left": "hooked gracefully over the crook of the LEFT forearm/elbow, hand relaxed forward",
    "shoulder_right": "slung vertically over the RIGHT shoulder by its strap, bag body resting cleanly against the right flank/hip",
    "shoulder_left": "slung vertically over the LEFT shoulder by its strap, bag body resting cleanly against the left flank/hip",
    "crossbody_l2r": "worn crossbody with the strap running diagonally from the LEFT shoulder across the chest so the bag rests at the RIGHT hip",
    "crossbody_r2l": "worn crossbody with the strap running diagonally from the RIGHT shoulder across the chest so the bag rests at the LEFT hip",
    "underarm_clutch": "tucked neatly under the arm against the ribcage/waist as a clutch hold",
    "backpack": "worn on the back with shoulder straps resting naturally over the shoulders",
    "placed_beside": "NOT held in the hands—placed upright as a grounded editorial prop beside the subject (on the floor next to the chair/feet or adjacent surface) with realistic contact shadow",
}

SHOE_SLOTS: Dict[str, str] = {
    "dress_shoes": "TAILORED DRESS SHOES / DERBIES / OXFORDS",
    "loafers": "PENNY / TASSEL LOAFERS",
    "sneakers": "CLEAN MINIMALIST LOW-TOP SNEAKERS",
    "boots": "ANKLE / CHELSEA BOOTS",
    "heels_sandals": "EDITORIAL HEELS / STRAPPY SANDALS",
}


@dataclass
class AttachmentItem:
    """Single user-attached original styling item."""

    category: str  # "garment" | "accessory" | "bag" | "shoe"
    image_path: Optional[Path] = None
    description: str = ""
    sub_role: str = ""
    wearing_style: str = "auto"
    scale_mode: str = "medium_standard"
    size_cm: Optional[float] = None
    carry_mode: str = "hand_right"

    def __post_init__(self) -> None:
        if self.category not in ITEM_CATEGORIES:
            raise ValueError(f"Unsupported item category '{self.category}'. Expected one of {ITEM_CATEGORIES}.")


@dataclass
class SceneLocationInput:
    """Single target location or mood/style reference (uploading N creates N parallel jobs)."""

    name: str = "scene"
    image_path: Optional[Path] = None
    raw_unmasked_path: Optional[Path] = None
    mood_or_place_prompt: str = ""
    lighting_note: str = ""
    scope: str = "scene_and_mood"
    override_yaw_deg: Optional[float] = None


@dataclass
class ComposedSceneJob:
    """Fully resolved, firewall- & origin-guard-validated single generation job (`1원본들 → 1생성`)."""

    job_index: int
    job_tag: str
    matched_8angle_view: str
    head_azimuth_deg: float
    direction_source: str  # "auto_reference_pose" | "auto_scene_light" | "manual_override"
    prompt: str
    ordered_reference_images: List[Dict[str, str]] = field(default_factory=list)
    previz_ui_preview_path: Optional[str] = None  # For UI display ONLY — NEVER passed into image generator!
    location_name: str = ""

    @property
    def previz_guide_path(self) -> Optional[str]:
        return self.previz_ui_preview_path


class MultiSceneBatchComposer:
    """Expands (Subject 8-Direction Face Bank × Original Styling Assets × N Locations × Pose) under `1원본들 → 1생성`."""

    def __init__(
        self,
        turnaround_engine: RealPersonTurnaroundEngine,
        origin_guard: Optional[OriginOnlyOneShotGuard] = None,
    ) -> None:
        self.turnaround = turnaround_engine
        self.origin_guard = origin_guard or OriginOnlyOneShotGuard()

    def _prepare_masked_copy(self, src_path: Path, work_dir: Path, prefix: str) -> Tuple[Path, int]:
        bgr = cv2.imread(str(src_path))
        if bgr is None:
            raise FileNotFoundError(f"Cannot read attachment image: {src_path}")
        out_path = work_dir / f"{prefix}_{src_path.stem}_masked.jpg"
        _, masked_count = self.turnaround.mask_foreign_faces_in_reference(bgr, output_masked_path=out_path)
        if masked_count == 0:
            out_path.unlink(missing_ok=True)
            return src_path, 0
        return out_path, masked_count

    @staticmethod
    def compile_styling_architecture(
        prepared_items: Sequence[Tuple[AttachmentItem, Optional[Path], int]],
    ) -> Tuple[str, Optional[str], str]:
        """Compile structured Z-order garment layering + anthropometric scale lock + bag carry kinematics."""
        if not prepared_items:
            return (
                "contemporary tailored editorial attire harmonizing naturally with the environment",
                None,
                "medium_standard",
            )

        garments = [it for it, _, _ in prepared_items if it.category == "garment"]
        accessories = [it for it, _, _ in prepared_items if it.category == "accessory"]
        bags = [it for it, _, _ in prepared_items if it.category == "bag"]
        shoes = [it for it, _, _ in prepared_items if it.category == "shoe"]

        has_inner_or_mid = any(g.sub_role in {"inner_top", "mid_layer"} for g in garments)
        has_watch_left = any(a.sub_role == "watch_left" for a in accessories)
        has_watch_right = any(a.sub_role == "watch_right" for a in accessories)

        clauses: List[str] = []

        if garments:
            layer_order = {"inner_top": 1, "mid_layer": 2, "outerwear": 3, "one_piece": 4, "bottom": 5}
            sorted_g = sorted(garments, key=lambda x: layer_order.get(x.sub_role, 3))
            g_parts: List[str] = []
            for g in sorted_g:
                slot_title = GARMENT_SLOTS.get(g.sub_role, f"GARMENT ({g.sub_role or 'piece'})")
                w_style = g.wearing_style or "auto"
                if w_style == "auto" and g.sub_role == "outerwear" and has_inner_or_mid:
                    style_txt = WEARING_STYLES["open_layered"]
                else:
                    style_txt = WEARING_STYLES.get(w_style, "")
                desc = g.description.strip() or "exact attached garment design, fabric weave, color, and seam construction"
                part = f"{slot_title}: {desc}"
                if style_txt:
                    part += f" ({style_txt})"
                g_parts.append(part)
            clauses.append("Garment Layering Z-Order (inside-out): " + " -> ".join(g_parts))

        if accessories:
            a_parts: List[str] = []
            for a in accessories:
                acc_title, acc_scale = ACCESSORY_SLOTS.get(
                    a.sub_role,
                    (f"ACCESSORY ({a.sub_role or 'detail'})", "realistic human-body-relative scale without exaggeration"),
                )
                if a.size_cm:
                    acc_scale = f"exact {a.size_cm:.1f}cm physical dimension ({acc_scale})"
                elif a.scale_mode == "delicate":
                    acc_scale = f"minimalist slim profile ({acc_scale})"
                elif a.scale_mode == "bold":
                    acc_scale = f"statement editorial proportion yet anatomically grounded ({acc_scale})"
                desc = a.description.strip() or "exact attached accessory hardware, material, and finish"
                a_parts.append(f"{acc_title}: {desc} [Scale Lock: {acc_scale}]")
            if has_watch_left:
                a_parts.append("ensure left sleeve cuff sits ~1.5cm above the wrist bone so the left wristwatch is cleanly visible")
            if has_watch_right:
                a_parts.append("ensure right sleeve cuff sits ~1.5cm above the wrist bone so the right wristwatch is cleanly visible")
            clauses.append("Accessories & Anthropometric Scale Lock: " + "; ".join(a_parts))

        active_bag_carry: Optional[str] = None
        active_bag_size = "medium_standard"
        if bags:
            b_parts: List[str] = []
            for b in bags:
                sz_key = b.scale_mode if b.scale_mode in BAG_SIZES else "medium_standard"
                active_bag_size = sz_key
                sz_title, sz_rule, _ = BAG_SIZES[sz_key]
                if b.size_cm:
                    sz_rule = f"strict {b.size_cm:.0f}cm physical width relative to ~38cm adult torso breadth ({sz_rule})"
                cm_key = b.carry_mode if b.carry_mode in BAG_CARRY_MODES else "hand_right"
                active_bag_carry = cm_key
                cm_rule = BAG_CARRY_MODES[cm_key]
                desc = b.description.strip() or "exact attached bag structure, leather/fabric grain, stitching, and hardware"
                b_parts.append(f"{sz_title}: {desc} [Scale Lock: {sz_rule}; Carry & Position: {cm_rule}]")
            clauses.append("Bag Proportion & Carry Kinematics: " + "; ".join(b_parts))

        if shoes:
            s_parts: List[str] = []
            for s in shoes:
                sh_title = SHOE_SLOTS.get(s.sub_role, f"FOOTWEAR ({s.sub_role or 'shoes'})")
                desc = s.description.strip() or "exact attached footwear silhouette, sole thickness, upper material, and color"
                s_parts.append(
                    f"{sh_title}: {desc} [Scale Lock: natural foot-to-height anatomical proportion (~15% of body height) grounded firmly on the floor]"
                )
            clauses.append("Footwear: " + "; ".join(s_parts))

        return " | ".join(clauses), active_bag_carry, active_bag_size

    def compose_batch(
        self,
        canonical_front_path: Path,
        subject: SubjectProfile,
        locations: Sequence[SceneLocationInput],
        poses: Sequence[PosePrevizSpec],
        angle_bank: Optional[Dict[str, Path]] = None,
        master_8dir_sheet_path: Optional[Path] = None,
        items: Sequence[AttachmentItem] = (),
        work_dir: Optional[Path] = None,
        aspect_ratio_label: str = "16:9",
        face_direction_mode: str = "manual",  # Studio UI passes "auto" by default or "manual" when user chooses an angle
        attach_previz_image: bool = False,  # HARD DEFAULT FALSE (`1원본들 → 1생성`: never pollute `--image` with synthetic wireframes)
    ) -> List[ComposedSceneJob]:
        """Compose `len(locations) * len(poses)` pure `1원본들 → 1생성` scene jobs attaching 8-Direction Face + Raw Original Assets."""
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
        raw_bank = angle_bank or {}

        # Normalize & whitelist all approved 8-Direction Face Anchors + Canonical 0° Front
        self.origin_guard.register_approved_origin(canonical_front_path, kind="8dir_face")
        norm_bank: Dict[str, Path] = {"front": Path(canonical_front_path)}
        for k, p in raw_bank.items():
            if p is not None and Path(p).exists():
                canon_k = LEGACY_VIEW_ALIASES.get(k, k)
                norm_bank[canon_k] = Path(p)
                self.origin_guard.register_approved_origin(Path(p), kind="8dir_face")

        # Ensure Master 8-Direction Turnaround Sheet exists when >= 3 angles are in the bank so ALL 8 directions are visually attached!
        resolved_sheet_path: Optional[Path] = None
        if master_8dir_sheet_path is not None and Path(master_8dir_sheet_path).exists():
            resolved_sheet_path = Path(master_8dir_sheet_path)
        elif len(norm_bank) >= 3:
            auto_sheet = stage_dir / "subject_8dir_master_sheet.jpg"
            self.turnaround.assemble_master_turnaround_sheet(norm_bank, auto_sheet, include_labels=True)
            resolved_sheet_path = auto_sheet
        if resolved_sheet_path is not None:
            self.origin_guard.register_approved_origin(resolved_sheet_path, kind="8dir_face")

        # Prepare raw original styling items (byte-for-byte original unless a foreign model's face required grey masking)
        prepared_items: List[Tuple[AttachmentItem, Optional[Path], int]] = []
        for idx, it in enumerate(items):
            if it.image_path is not None and Path(it.image_path).exists():
                m_path, cnt = self._prepare_masked_copy(Path(it.image_path), stage_dir, f"item{idx:02d}_{it.category}")
                self.origin_guard.register_approved_origin(m_path, kind="raw_asset")
                prepared_items.append((it, m_path, cnt))
            else:
                prepared_items.append((it, None, 0))

        styling_directive, active_bag_carry, active_bag_size = self.compile_styling_architecture(prepared_items)
        has_shoes = any(it.category == "shoe" for it, _, _ in prepared_items)

        jobs: List[ComposedSceneJob] = []
        job_counter = 0

        for loc_idx, loc in enumerate(locations):
            loc_img_prepared: Optional[Path] = None
            loc_masked_cnt = 0
            loc_auto_info: Optional[Dict[str, object]] = None

            src_for_analysis = loc.raw_unmasked_path or loc.image_path
            if src_for_analysis is not None and Path(src_for_analysis).exists():
                raw_loc_bgr = cv2.imread(str(src_for_analysis))
                if raw_loc_bgr is not None:
                    loc_auto_info = self.turnaround.analyze_reference_pose_before_masking(raw_loc_bgr)

            if loc.image_path is not None and Path(loc.image_path).exists():
                loc_img_prepared, loc_masked_cnt = self._prepare_masked_copy(
                    Path(loc.image_path), stage_dir, f"loc{loc_idx:02d}"
                )
                self.origin_guard.register_approved_origin(loc_img_prepared, kind="raw_asset")

            for pose_idx, orig_pose in enumerate(poses):
                job_counter += 1
                job_tag = f"scene_{loc_idx + 1:02d}_pose_{pose_idx + 1:02d}"

                # Resolve Head & Torso Direction: Auto (from location reference pose / light) vs Manual Override
                if loc.override_yaw_deg is not None:
                    eff_head_yaw = float(loc.override_yaw_deg)
                    eff_torso_yaw = eff_head_yaw * 0.85
                    eff_x_norm = orig_pose.subject_x_norm
                    dir_source = "manual_location_override"
                elif face_direction_mode == "auto":
                    if loc_auto_info is not None:
                        eff_head_yaw = float(loc_auto_info["auto_yaw_deg"])  # type: ignore[arg-type]
                        eff_torso_yaw = eff_head_yaw * 0.80
                        eff_x_norm = float(loc_auto_info["auto_subject_x_norm"])  # type: ignore[arg-type]
                        dir_source = f"auto_{loc_auto_info['auto_source']}"
                    else:
                        eff_head_yaw = 0.0
                        eff_torso_yaw = 0.0
                        eff_x_norm = orig_pose.subject_x_norm
                        dir_source = "auto_default_front"
                else:
                    eff_head_yaw = orig_pose.head_azimuth_deg
                    eff_torso_yaw = orig_pose.torso_azimuth_deg
                    eff_x_norm = orig_pose.subject_x_norm
                    dir_source = "manual_override"

                pose = PosePrevizSpec(
                    preset_id=orig_pose.preset_id,
                    head_azimuth_deg=eff_head_yaw,
                    head_pitch_deg=orig_pose.head_pitch_deg,
                    torso_azimuth_deg=eff_torso_yaw,
                    camera_elevation_deg=orig_pose.camera_elevation_deg,
                    focal_length_mm=orig_pose.focal_length_mm,
                    subject_x_norm=eff_x_norm,
                    shot_scale=(
                        "full-body editorial head-to-toe shot (framed closely from shoes near bottom edge to crown near top edge)"
                        if (has_shoes and "full-body" not in orig_pose.shot_scale)
                        else orig_pose.shot_scale
                    ),
                    custom_pose_note=orig_pose.custom_pose_note,
                    joints_override=dict(orig_pose.joints_override) if orig_pose.joints_override else None,
                    bag_carry_mode=active_bag_carry or orig_pose.bag_carry_mode,
                    bag_size=active_bag_size or orig_pose.bag_size,
                )

                id_refs, matched_view = self.turnaround.select_angle_matched_references(
                    canonical_front_path=canonical_front_path,
                    angle_bank=raw_bank,
                    target_azimuth_deg=pose.head_azimuth_deg,
                )

                ordered_imgs: List[Dict[str, str]] = []
                role_clauses: List[str] = []

                # ── FACE ANCHORS (8-Direction Angle-Matched Primary + 0° Origin + Full 8-Direction Master Sheet) ──
                if len(id_refs) > 1:
                    # Put the exact angle-matched 8-direction portrait FIRST for 1:1 angle lock, paired with untouched 0° front!
                    ordered_imgs.append(
                        {
                            "path": str(id_refs[1]),
                            "role": f"identity_angle_{matched_view}",
                            "label": f"Primary target-angle 8-direction portrait ({VIEW_LABELS_KO.get(matched_view, matched_view)})",
                        }
                    )
                    role_clauses.append(
                        f"Image 1 (TARGET-ANGLE-8DIR-FACE — {matched_view.upper()}, {pose.head_azimuth_deg:+.1f} deg): "
                        "approved target-angle portrait of the subject from the 8-Direction Turnaround Bank—copy this EXACT facial "
                        "structure, eye/nose/mouth/jawline geometry, hair parting, and eyewear temple silhouette at this head angle "
                        "(ignore its studio background and dark studio suit)"
                    )
                    ordered_imgs.append(
                        {
                            "path": str(id_refs[0]),
                            "role": "identity_primary_front",
                            "label": "Untouched 0-deg original front photograph (true iris color & skin texture ground-truth)",
                        }
                    )
                    role_clauses.append(
                        "Image 2 (IDENTITY-ORIGIN-0DEG): untouched 0-degree original photograph of the exact same person—"
                        "ground-truth authority for authentic iris color, real skin pore texture, and exact facial likeness"
                    )
                else:
                    ordered_imgs.append(
                        {
                            "path": str(id_refs[0]),
                            "role": "identity_primary_front",
                            "label": "Untouched 0-deg original front photograph",
                        }
                    )
                    role_clauses.append(
                        "Image 1 (IDENTITY-ORIGIN-0DEG): untouched 0-degree original photograph of the target subject—"
                        "sole authority for exact 1:1 facial likeness, authentic iris color, real skin texture, and eyewear"
                    )

                if resolved_sheet_path is not None:
                    ordered_imgs.append(
                        {
                            "path": str(resolved_sheet_path),
                            "role": "identity_8dir_master_sheet",
                            "label": "Complete -90°..+90° 8-Direction Facial Turnaround Sheet",
                        }
                    )
                    role_clauses.append(
                        f"Image {len(ordered_imgs)} (8-DIRECTION-MASTER-TURNAROUND-SHEET): full -90 deg to +90 deg 8-direction "
                        "reference sheet of this exact person confirming 3D craniofacial consistency across all angles "
                        "(render ONE single person in the scene, never a multi-panel grid)"
                    )

                # ── RAW ORIGINAL STYLING ASSETS (`1원본들 → 1생성`: pass every raw item original directly!) ──
                for it, m_path, m_cnt in prepared_items:
                    if m_path is not None:
                        cat_title = {
                            "garment": f"ORIGINAL-GARMENT-{it.sub_role.upper() if it.sub_role else 'ITEM'}",
                            "accessory": f"ORIGINAL-ACCESSORY-{it.sub_role.upper() if it.sub_role else 'ITEM'}",
                            "bag": f"ORIGINAL-BAG-{it.scale_mode.upper()}",
                            "shoe": f"ORIGINAL-FOOTWEAR-{it.sub_role.upper() if it.sub_role else 'SHOES'}",
                        }[it.category]
                        ordered_imgs.append(
                            {
                                "path": str(m_path),
                                "role": f"item_{it.category}",
                                "label": f"{cat_title}: {it.description}".strip(": "),
                            }
                        )
                        mask_hint = " (ignore grey box masking original model's face)" if m_cnt > 0 else ""
                        role_clauses.append(
                            f"Image {len(ordered_imgs)} ({cat_title}): raw original product photo—reproduce its exact material, "
                            f"color, weave, stitching, and hardware{mask_hint}"
                        )

                # ── RAW ORIGINAL LOCATION / STYLE ASSET ──
                loc_clause = ""
                if loc_img_prepared is not None:
                    ordered_imgs.append(
                        {
                            "path": str(loc_img_prepared),
                            "role": "location_or_mood_reference",
                            "label": f"Original Location / Style Reference ({loc.name})",
                        }
                    )
                    mask_hint = " (ignore grey box masking any foreign person's face)" if loc_masked_cnt > 0 else ""
                    if loc.scope == "style_mood_only":
                        role_clauses.append(
                            f"Image {len(ordered_imgs)} (STYLE-MOOD-ORIGIN): extract its exact 16-point tone & manner—brightness, "
                            "key-light direction, shadow softness, color temperature, color grading, and editorial mood—into a fresh spatial composition"
                            + mask_hint
                        )
                    else:
                        role_clauses.append(
                            f"Image {len(ordered_imgs)} (LOCATION-AND-TONE-ORIGIN): place the subject naturally inside this environment's "
                            "architecture, spatial depth, directional lighting, shadow falloff, color temperature, and overall tone & manner"
                            + mask_hint
                        )

                if loc.mood_or_place_prompt.strip():
                    loc_clause += f" Environment & Mood Direction: {loc.mood_or_place_prompt.strip()}."
                if loc.lighting_note.strip():
                    loc_clause += f" Lighting & Atmosphere: {loc.lighting_note.strip()}."
                if not loc_clause and loc_img_prepared is None:
                    loc_clause = " Environment & Mood Direction: clean contemporary architectural editorial setting with natural directional key light."

                # Render UI-only 3D Previz thumbnail (saved under `ui_preview_*` and NEVER added to `ordered_imgs`!)
                ui_previz_p = stage_dir / f"ui_preview_{job_tag}.jpg"
                Previz3DRenderer.render_pose_guide(pose, canvas_size=(960, 540), output_path=ui_previz_p)

                # Hard `1원본들 → 1생성` verification on all input image paths!
                self.origin_guard.verify_one_shot_and_inputs([Path(r["path"]) for r in ordered_imgs])

                prompt = (
                    "Editorial portrait. Shot on a medium-format 100MP camera with prime optics, tack-sharp optical micro-contrast, "
                    "deep true blacks, and zero atmospheric haze or milky veil. "
                    "CRITICAL MANDATORY RULE — 1-ORIGINALS-TO-1-GENERATION REAL IDENTITY LOCK + HIERARCHICAL LAYERING + SCALE LOCK: "
                    f"Create ONE single photorealistic {aspect_ratio_label} photograph of the {subject.subject_identity_label} "
                    "whose exact face at this angle is shown in Image 1 (and confirmed by the 0-degree original & 8-Direction Turnaround Sheet). "
                    "Attached Original Reference Roles: " + "; ".join(role_clauses) + ". "
                    f"Identity & Eyewear Continuity: Preserve {subject.facial_identity_descriptor()}. "
                    "Do NOT invent a generic model face—lock 100% onto the exact person in Image 1 & Image 2 with authentic iris color "
                    "(zero green or purple lens-coating tint), and if eyewear is worn, render both temple arms continuously over the ears. "
                    f"{blueprint.to_3d_invariance_clause()} "
                    f"Physical Build: {subject.body_frame_descriptor()}, maintaining balanced adult 7.5-to-8-head head-to-shoulder proportions. "
                    f"Pose & Camera Rig: {pose.to_prompt_clause()}. "
                    f"Layered Wardrobe, Scale-Locked Accessories, Bag Kinematics & Footwear: {styling_directive}. "
                    f"16-Point Tone & Manner Integration:{loc_clause} Harmonize key-light direction, shadow softness, floor/furniture contact shadows, "
                    "ambient wall color bounce, fabric surface reflectance, and depth of field so the subject, layered outfit, accessories, bag, "
                    "and space read as one authentic single-camera photograph. "
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
                        direction_source=dir_source,
                        prompt=prompt,
                        ordered_reference_images=ordered_imgs,
                        previz_ui_preview_path=str(ui_previz_p),
                        location_name=loc.name,
                    )
                )

        return jobs
