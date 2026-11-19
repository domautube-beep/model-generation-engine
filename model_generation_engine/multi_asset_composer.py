"""Universal Multi-Asset, Hierarchical Layering, Anthropometric Scale Lock & Multi-Location Batch Composer (`multi_asset_composer.py` v3.2).

Solves the 3 real-world styling challenges universally (0% hardcoding):
1. Multi-Piece Garment Layering (`inner_top` -> `mid_layer` -> `outerwear` + `bottom` / `one_piece`):
   - Explicit Z-order layering rules so inner shirts/t-shirts are never swallowed by buttoned outerwear unless explicitly closed.
2. Anthropometric Scale & Proportion Lock (`scale_mode`, `size_cm`, and slot-to-body ratio anchors):
   - Anchors every accessory (`watch ~40mm / 70% of wrist width`, `eyewear 1:1 temple span`, `belt 3.2cm buckle`, `fine jewelry 1-2.5cm`)
     and every bag (`micro_mini ~20cm`, `small_compact ~26cm`, `medium_standard ~36cm`, `large_tote_weekender ~48cm`) against human body
     landmarks (`palm ~8.5cm`, `head height ~22cm`, `torso width ~38cm`) so standalone product photos never blow up or shrink.
   - Builds a proportional **Composite Styling Board (`composite_styling_sheet.jpg`)** with anatomically weighted cell sizes so
     small accessories do not visually overpower full garments in multi-image attention.
3. Bag Carry & Placement Kinematics (`carry_mode` coupled with 3D Previz Pose):
   - Supports hand-held (`hand_right`, `hand_left`), crook-of-arm (`crook_right`, `crook_left`), shoulder (`shoulder_right`, `shoulder_left`),
     crossbody (`crossbody_l2r`, `crossbody_r2l`), underarm clutch (`underarm_clutch`), backpack (`backpack`), and floor/table placement (`placed_beside`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
from PIL import Image, ImageDraw, ImageFont

from .config import SubjectProfile
from .identity_turnaround import AnatomicalBlueprint, RealPersonTurnaroundEngine
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
    """Single user-attached styling item with explicit layering slot, anthropometric scale, and carry/wear mode."""

    category: str  # "garment" | "accessory" | "bag" | "shoe"
    image_path: Optional[Path] = None
    description: str = ""
    sub_role: str = ""  # e.g. "inner_top", "outerwear", "bottom", "watch_left", "loafers"
    wearing_style: str = "auto"  # for garments: "auto", "open_layered", "closed_buttoned", "tucked_in", etc.
    scale_mode: str = "medium_standard"  # for bags: BAG_SIZES key; for accessories: "auto" | "delicate" | "bold"
    size_cm: Optional[float] = None  # optional explicit width/diameter in cm
    carry_mode: str = "hand_right"  # for bags: BAG_CARRY_MODES key

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
    scope: str = "scene_and_mood"


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
    styling_board_path: Optional[str] = None
    location_name: str = ""


class MultiSceneBatchComposer:
    """Expands (Subject 8-Direction Bank × Layered/Scale-Locked Items × N Locations × M Previz Poses) into parallel jobs."""

    def __init__(self, turnaround_engine: RealPersonTurnaroundEngine) -> None:
        self.turnaround = turnaround_engine

    def _prepare_masked_copy(self, src_path: Path, work_dir: Path, prefix: str) -> Tuple[Path, int]:
        bgr = cv2.imread(str(src_path))
        if bgr is None:
            raise FileNotFoundError(f"Cannot read attachment image: {src_path}")
        out_path = work_dir / f"{prefix}_{src_path.stem}_masked.jpg"
        _, masked_count = self.turnaround.mask_foreign_faces_in_reference(bgr, output_masked_path=out_path)
        return (out_path if masked_count > 0 else src_path), masked_count

    @staticmethod
    def _build_proportional_styling_board(
        prepared_items: Sequence[Tuple[AttachmentItem, Optional[Path], int]],
        output_board_path: Path,
    ) -> Optional[Path]:
        """Assemble multiple attached item photos into a single anatomically proportioned, labeled Styling Board."""
        visual_items = [(it, p) for it, p, _ in prepared_items if p is not None and p.exists()]
        if len(visual_items) < 2:
            return None

        # Anatomically proportional cell heights so watches/accessories don't look as huge as coats!
        cols = min(4, len(visual_items))
        rows = (len(visual_items) + cols - 1) // cols
        cell_w, cell_h, header_h, pad = 380, 420, 54, 18
        board_w = pad + cols * (cell_w + pad)
        board_h = pad + rows * (cell_h + header_h + pad)
        board = Image.new("RGB", (board_w, board_h), (246, 246, 244))
        draw = ImageDraw.Draw(board)
        font = ImageFont.load_default()
        for fp in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/AppleSDGothicNeo.ttc"):
            if Path(fp).exists():
                try:
                    font = ImageFont.truetype(fp, 17)
                    break
                except Exception:
                    pass

        for idx, (it, p) in enumerate(visual_items):
            r, c = divmod(idx, cols)
            x0 = pad + c * (cell_w + pad)
            y0 = pad + r * (cell_h + header_h + pad)
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h + header_h], fill=(255, 255, 255), outline=(215, 218, 224), width=2)

            # Relative visual occupancy inside the cell (accessories/mini-bags occupy smaller visual area!)
            occ = 0.90
            if it.category == "accessory":
                occ = 0.52
            elif it.category == "shoe":
                occ = 0.68
            elif it.category == "bag":
                occ = {"micro_mini": 0.48, "small_compact": 0.62, "medium_standard": 0.76, "large_tote_weekender": 0.92}.get(
                    it.scale_mode, 0.74
                )

            max_iw = int(round(cell_w * occ))
            max_ih = int(round(cell_h * occ))
            im = Image.open(p).convert("RGB")
            im.thumbnail((max_iw, max_ih), Image.Resampling.LANCZOS)
            ix = x0 + (cell_w - im.width) // 2
            iy = y0 + (cell_h - im.height) // 2
            board.paste(im, (ix, iy))

            tag_role = (it.sub_role or it.category).upper()[:24]
            extra = ""
            if it.category == "bag":
                extra = f" [{it.scale_mode.split('_')[0].upper()}]"
            elif it.category == "garment" and it.wearing_style != "auto":
                extra = f" [{it.wearing_style.split('_')[0].upper()}]"
            lbl = f"#{idx + 1} {tag_role}{extra}"
            draw.text((x0 + 10, y0 + cell_h + 14), lbl, fill=(32, 36, 44), font=font)

        output_board_path.parent.mkdir(parents=True, exist_ok=True)
        board.save(str(output_board_path), format="JPEG", quality=95)
        return output_board_path

    @staticmethod
    def compile_styling_architecture(
        prepared_items: Sequence[Tuple[AttachmentItem, Optional[Path], int]],
    ) -> Tuple[str, Optional[str], str]:
        """Compile structured Z-order garment layering + anthropometric scale lock + bag carry kinematics.

        Returns `(full_styling_directive, active_bag_carry_mode, active_bag_size)`.
        """
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

        # 1. Hierarchical Garment Layering (Inner -> Mid -> Outer + Bottom)
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

        # 2. Anthropometric Scale-Locked Accessories
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

        # 3. Bag Size Ratio & Carry/Placement Kinematics
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

        # 4. Footwear
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

        prepared_items: List[Tuple[AttachmentItem, Optional[Path], int]] = []
        for idx, it in enumerate(items):
            if it.image_path is not None and Path(it.image_path).exists():
                m_path, cnt = self._prepare_masked_copy(Path(it.image_path), stage_dir, f"item{idx:02d}_{it.category}")
                prepared_items.append((it, m_path, cnt))
            else:
                prepared_items.append((it, None, 0))

        styling_directive, active_bag_carry, active_bag_size = self.compile_styling_architecture(prepared_items)
        has_shoes = any(it.category == "shoe" for it, _, _ in prepared_items)

        # Build composite proportional styling board when >=2 item images are attached
        board_path = self._build_proportional_styling_board(prepared_items, stage_dir / "composite_styling_board.jpg")

        jobs: List[ComposedSceneJob] = []
        job_counter = 0

        for loc_idx, loc in enumerate(locations):
            loc_img_prepared: Optional[Path] = None
            loc_masked_cnt = 0
            if loc.image_path is not None and Path(loc.image_path).exists():
                loc_img_prepared, loc_masked_cnt = self._prepare_masked_copy(
                    Path(loc.image_path), stage_dir, f"loc{loc_idx:02d}"
                )

            for pose_idx, orig_pose in enumerate(poses):
                job_counter += 1
                job_tag = f"scene_{loc_idx + 1:02d}_pose_{pose_idx + 1:02d}"

                # Synchronize pose with bag prop & full-body shoe visibility if needed
                pose = PosePrevizSpec(
                    preset_id=orig_pose.preset_id,
                    head_azimuth_deg=orig_pose.head_azimuth_deg,
                    head_pitch_deg=orig_pose.head_pitch_deg,
                    torso_azimuth_deg=orig_pose.torso_azimuth_deg,
                    camera_elevation_deg=orig_pose.camera_elevation_deg,
                    focal_length_mm=orig_pose.focal_length_mm,
                    subject_x_norm=orig_pose.subject_x_norm,
                    shot_scale=(
                        "full-body editorial head-to-toe shot showing complete footwear"
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
                    angle_bank=bank,
                    target_azimuth_deg=pose.head_azimuth_deg,
                )

                ordered_imgs: List[Dict[str, str]] = []
                role_clauses: List[str] = []

                ordered_imgs.append(
                    {
                        "path": str(id_refs[0]),
                        "role": "identity_primary_front",
                        "label": "Primary untouched 0-deg front identity anchor",
                    }
                )
                role_clauses.append(
                    "Image 1 (IDENTITY-PRIMARY-FRONT): untouched original photograph of the target subject—sole authority "
                    "for true facial identity, authentic iris color, skin texture, eyewear frame material, and hair color"
                )

                if len(id_refs) > 1:
                    ordered_imgs.append(
                        {
                            "path": str(id_refs[1]),
                            "role": f"identity_angle_{matched_view}",
                            "label": f"Approved {matched_view} frontal-arc anchor ({pose.head_azimuth_deg:+.1f} deg)",
                        }
                    )
                    role_clauses.append(
                        f"Image {len(ordered_imgs)} (IDENTITY-8DIR-{matched_view.upper()}): approved {matched_view} angle view "
                        "of the exact same subject—use strictly for 3D facial/nasal/cheek/ear/eyewear-temple foreshortening at this yaw "
                        "(do NOT copy its grey studio background or studio suit)"
                    )

                if board_path is not None:
                    ordered_imgs.append(
                        {
                            "path": str(board_path),
                            "role": "composite_styling_board",
                            "label": "Proportional Multi-Item Styling Board (all garments, accessories, bag, footwear)",
                        }
                    )
                    role_clauses.append(
                        f"Image {len(ordered_imgs)} (PROPORTIONAL-STYLING-BOARD): composite multi-slot product sheet showing the exact "
                        "layered garments, accessories, bag, and footwear in anatomically weighted visual scale—reproduce every product's "
                        "design, fabric texture, color, stitching, and hardware while obeying the Anthropometric Scale Lock (never draw the "
                        "white grid card itself in the scene)"
                    )

                for it, m_path, m_cnt in prepared_items:
                    if m_path is not None:
                        cat_title = {
                            "garment": f"GARMENT-{it.sub_role.upper() if it.sub_role else 'ITEM'}",
                            "accessory": f"ACCESSORY-{it.sub_role.upper() if it.sub_role else 'ITEM'}",
                            "bag": f"BAG-{it.scale_mode.upper()}",
                            "shoe": f"FOOTWEAR-{it.sub_role.upper() if it.sub_role else 'SHOES'}",
                        }[it.category]
                        ordered_imgs.append(
                            {
                                "path": str(m_path),
                                "role": f"item_{it.category}",
                                "label": f"{cat_title}: {it.description}".strip(": "),
                            }
                        )
                        mask_hint = (
                            " (ignore grey face-mask box covering original model)"
                            if m_cnt > 0
                            else ""
                        )
                        role_clauses.append(
                            f"Image {len(ordered_imgs)} ({cat_title}): exact product reference for {it.category}{mask_hint}"
                        )

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
                        " (ignore grey face-mask box covering any foreign person)"
                        if loc_masked_cnt > 0
                        else ""
                    )
                    if loc.scope == "style_mood_only":
                        role_clauses.append(
                            f"Image {len(ordered_imgs)} (STYLE-MOOD-REFERENCE): extract its exact 16-point tone & manner—brightness, "
                            "key-light direction, shadow softness, color temperature, color grading, and editorial mood—into a fresh spatial composition"
                            + mask_hint
                        )
                    else:
                        role_clauses.append(
                            f"Image {len(ordered_imgs)} (LOCATION-AND-TONE-REFERENCE): place the subject naturally inside this environment's "
                            "architecture, spatial depth, directional lighting, shadow falloff, color temperature, and overall tone & manner"
                            + mask_hint
                        )

                if loc.mood_or_place_prompt.strip():
                    loc_clause += f" Environment & Mood Direction: {loc.mood_or_place_prompt.strip()}."
                if loc.lighting_note.strip():
                    loc_clause += f" Lighting & Atmosphere: {loc.lighting_note.strip()}."
                if not loc_clause and loc_img_prepared is None:
                    loc_clause = " Environment & Mood Direction: clean contemporary architectural editorial setting with natural directional key light."

                previz_path: Optional[Path] = None
                if attach_previz_image:
                    previz_path = stage_dir / f"previz_{job_tag}.jpg"
                    Previz3DRenderer.render_pose_guide(pose, canvas_size=(1280, 720), output_path=previz_path)
                    ordered_imgs.append(
                        {
                            "path": str(previz_path),
                            "role": "pose_previz_guide",
                            "label": f"3D Previz Pose & Bag Rig ({pose.head_azimuth_deg:+.1f} deg)",
                        }
                    )
                    role_clauses.append(
                        f"Image {len(ordered_imgs)} (3D-PREVIZ-POSE-GUIDE): structural 3D mannequin pose & prop-placement blueprint ONLY—"
                        "match its camera framing, torso/head angle, limb posture, and bag position on the real human subject; "
                        "NEVER draw stick-figure lines, mannequin spheres, wireframe bag boxes, or dark grid backgrounds in the final photograph"
                    )

                prompt = (
                    "Editorial portrait. Shot on a medium-format 100MP camera with prime optics, tack-sharp optical micro-contrast, "
                    "deep true blacks, and zero atmospheric haze or milky veil. "
                    "CRITICAL MANDATORY RULE — REAL IDENTITY LOCK + HIERARCHICAL LAYERING + ANTHROPOMETRIC SCALE LOCK + 16-POINT TONE HARMONY: "
                    f"Create ONE photorealistic {aspect_ratio_label} photograph of the {subject.subject_identity_label} from Image 1. "
                    "Attached Reference Roles: " + "; ".join(role_clauses) + ". "
                    f"Identity & Eyewear Continuity: Preserve {subject.facial_identity_descriptor()}. "
                    "Keep exact 1:1 facial anatomy, authentic iris color from Image 1 (zero green or purple lens-coating tint), "
                    "and if eyewear is worn, render both temple arms continuously from the outer front hinges over the ears. "
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
                        prompt=prompt,
                        ordered_reference_images=ordered_imgs,
                        previz_guide_path=str(previz_path) if previz_path else None,
                        styling_board_path=str(board_path) if board_path else None,
                        location_name=loc.name,
                    )
                )

        return jobs
