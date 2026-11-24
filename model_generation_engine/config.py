from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple


def resolve_gdrive_vault(project_names: Sequence[str] = ("릴프로필", "프로필", "모델생성", "인물사진")) -> Optional[Path]:
    """Dynamically locate the Google Drive Heavy Asset Vault across macOS/Windows/Linux."""
    env_override = (
        os.environ.get("REELPROFILE_VAULT")
        or os.environ.get("MODEL_ENGINE_VAULT")
        or os.environ.get("PORTRAIT_ENGINE_VAULT")
    )
    if env_override and os.path.isdir(env_override):
        return Path(env_override)

    for project_name in project_names:
        patterns = [
            f"~/Library/CloudStorage/GoogleDrive-*/내 드라이브/Antigravity_Assets/{project_name}",
            f"~/Library/CloudStorage/GoogleDrive-*/My Drive/Antigravity_Assets/{project_name}",
            f"~/Google Drive/내 드라이브/Antigravity_Assets/{project_name}",
            f"~/Google Drive/My Drive/Antigravity_Assets/{project_name}",
            f"G:/내 드라이브/Antigravity_Assets/{project_name}",
            f"G:/My Drive/Antigravity_Assets/{project_name}",
        ]
        for pattern in patterns:
            for m in glob.glob(os.path.expanduser(pattern)):
                if os.path.isdir(m):
                    return Path(m)
    return None


@dataclass(frozen=True)
class SubjectProfile:
    """Universal, zero-hardcoding human subject specification.

    Works for ANY model (any gender, age, ethnicity, height/weight, eyewear state, hairstyle, or expression).
    When height_cm and weight_kg are provided, body build applies STRICTLY below the neck.
    When optional attributes (hairstyle, eyewear) are None, they are inherited 1:1 from the reference photo.
    """

    subject_identity_label: str = "human subject in the reference portrait"
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    hairstyle: Optional[str] = None
    eyewear: Optional[str] = None
    expression: str = "natural relaxed micro-expression harmonizing with the scene"
    facial_traits: Tuple[str, ...] = ()

    @property
    def bmi(self) -> Optional[float]:
        if self.height_cm is None or self.weight_kg is None or self.height_cm <= 0:
            return None
        h_m = self.height_cm / 100.0
        return self.weight_kg / (h_m * h_m)

    def body_frame_descriptor(self) -> str:
        """Return a gender-neutral, below-the-neck-only physical build descriptor."""
        bmi_val = self.bmi
        if bmi_val is None:
            return "natural proportional physique BELOW THE NECK ONLY"
        h_str = f"{int(round(self.height_cm))}cm"
        w_str = f"{int(round(self.weight_kg))}kg"
        if bmi_val >= 27.5:
            build = "broad, heavy-set, robust physical frame"
        elif bmi_val >= 24.5:
            build = "solid, broad-shouldered, sturdy physical build"
        elif bmi_val <= 19.5:
            build = "slender, lean tailored physical frame"
        else:
            build = "balanced, well-proportioned athletic physical build"
        return (
            f"{h_str}, {w_str} {build} BELOW THE NECK ONLY "
            "(applying strictly to shoulders, chest, torso, and legs without altering head or face proportions)"
        )

    def facial_identity_descriptor(self) -> str:
        """Return a universal 1:1 facial identity lock descriptor without hardcoded gender or glasses."""
        parts = [
            "exact 1:1 facial identity, bone structure, eye shape, authentic iris color, nose bridge and tip, "
            "mouth contour, and jawline of the reference portrait",
            self.expression,
        ]
        if self.eyewear is None:
            parts.append(
                "preserving the exact eyewear presence or absence shown in the reference portrait "
                "(if eyewear is worn, rendering both temple arms continuously from the outer front frame hinges over both ears)"
            )
        elif self.eyewear.lower() in ("none", "no glasses", "without glasses"):
            parts.append("wearing no glasses (bare eyes)")
        else:
            parts.append(
                f"wearing {self.eyewear} with both temple arms extending continuously and uninterruptedly "
                "from the outer front frame hinges over both ears"
            )

        if self.hairstyle is None:
            parts.append("preserving the exact hairstyle, parting, and hair texture of the reference portrait")
        else:
            parts.append(f"with {self.hairstyle}")

        for trait in self.facial_traits:
            parts.append(trait)
        return ", ".join(parts)


@dataclass(frozen=True)
class WardrobeSpec:
    """Universal outfit specification for any clothing style, season, or dress code."""

    style_category: str = "editorial_minimal"
    outfit_description: str = "contemporary minimalist tailored attire"
    accessories: Optional[str] = None

    def to_prompt_clause(self) -> str:
        if self.accessories:
            return f"{self.outfit_description}, paired with {self.accessories}"
        return self.outfit_description


@dataclass(frozen=True)
class LocationSpec:
    """Universal environment, location, backdrop, lighting, and prop specification."""

    environment_category: str = "studio_editorial"
    location_description: str = "clean minimalist studio backdrop"
    color_palette_hex: Optional[str] = None
    props_and_objects: Optional[str] = None
    lighting_description: str = "soft diffused natural-matte lighting with zero harsh specular glare"

    def to_prompt_clause(self) -> str:
        loc = self.location_description
        if self.color_palette_hex:
            loc = f"{loc} ({self.color_palette_hex})"
        if self.props_and_objects:
            return f"{loc}, featuring {self.props_and_objects}; {self.lighting_description}"
        return f"{loc}; {self.lighting_description}"


@dataclass(frozen=True)
class CameraRigSpec:
    """Universal camera angle, shot scale, focal length, aperture bokeh, and 50:50 negative-space specification."""

    shot_scale: str = "balanced 50:50 environmental medium portrait (subject ~48% of frame, ~52% open architectural negative space)"
    camera_angle: str = "eye-level camera angle"
    focal_length_mm: int = 85
    aperture: str = "f/1.8"
    framing_mode: str = "balanced_50_50"  # "balanced_50_50" (default) or "closeup" (on user request)
    pose_description: str = "natural composed editorial posture"
    negative_space_side: str = "left"  # "left", "right", "center", "top", "none"
    aspect_ratio_label: str = "16:9"

    def to_prompt_clause(self) -> str:
        neg_map = {
            "left": "subject positioned right-of-center (~45%–50% of frame) leaving ~50%–55% open out-of-focus architectural negative space and headroom on the left",
            "right": "subject positioned left-of-center (~45%–50% of frame) leaving ~50%–55% open out-of-focus architectural negative space and headroom on the right",
            "top": "subject positioned in the lower half (~48% of frame) leaving ~52% open softly defocused architectural headroom above",
            "center": "balanced 50:50 subject-to-background environmental framing with generous surrounding negative space (never edge-to-edge tight)",
            "none": "balanced 50:50 subject-to-background composition with natural breathing room",
        }
        if self.framing_mode == "closeup":
            neg_clause = "intimate user-requested close-up portrait framing (~70% subject occupancy) with creamy out-of-focus background bokeh"
        else:
            neg_clause = neg_map.get(self.negative_space_side.lower(), self.negative_space_side)
        return (
            f"{self.shot_scale} captured from a {self.camera_angle} with an {self.focal_length_mm}mm {self.aperture} prime lens "
            f"(shallow optical depth of field—tack-sharp focus on eyes/face, smooth creamy background bokeh without busy background detail, strong 3D Z-depth); "
            f"pose: {self.pose_description}; composition: {neg_clause}; {self.aspect_ratio_label} aspect ratio"
        )


@dataclass(frozen=True)
class CanvasSpec:
    """Universal target canvas specification supporting any resolution, aspect ratio, and auto-padding."""

    target_width: int = 5040
    target_height: int = 2160
    pad_left_ratio: Optional[float] = None  # None = auto-compute dynamically from detected subject position
    crop_aspect_ratio: float = 16.0 / 9.0
    crop_render_size: Tuple[int, int] = (1280, 720)
    seam_blend_width_px: int = 240
    wall_grain_sigma: float = 1.0

    @property
    def aspect_ratio(self) -> float:
        return self.target_width / float(self.target_height)


@dataclass
class EngineConfig:
    """Master configuration for 릴프로필 (프로필 / 모델생성 — 정본 v1.0.0+)."""

    project_name: str = "릴프로필"
    vault_dir: Optional[Path] = field(default_factory=lambda: resolve_gdrive_vault(("릴프로필", "프로필", "모델생성", "인물사진")))
    upscale_mode: str = "optical_lanczos"  # "optical_lanczos" (default, zero GAN corrosion) or "esrgan"
    zero_touch_4k_passthrough: bool = True  # Default True (v1.0.4): 0.0% face/lighting/skin/acuity post-mutation during 4K save!
    default_framing_mode: str = "balanced_50_50"  # Default 50:50 subject-to-space balance; switches to "closeup" on request
    upscale_model_name: str = "4xNomos8kSC"
    upscale_factor: int = 4
    codeformer_fidelity_w: float = 0.82
    unsharp_amount: float = 0.35
    unsharp_sigma: float = 1.0
    max_seam_l1_diff: float = 6.0
    min_face_laplacian_var: float = 12.0
    min_organ_gradient_energy: float = 1.6
    min_inter_ocular_5k_px: float = 40.0
    auto_facelock_below_iod_5k_px: float = 0.0
    max_face_shape_diff_pct: float = 4.5
    max_cheek_area_diff_pct: float = 6.5
    enforce_ocular_gate: bool = True
    min_editorial_face_to_shoulder_ratio: float = 0.285
    max_editorial_face_to_shoulder_ratio: float = 0.375
    max_cranial_dome_diff_pct: float = 18.0
    min_identity_cosine_sim: float = 0.65
    auto_restore_distant_face_iod_px: float = 0.0
    max_specular_glare_ratio: float = 0.04
    min_skin_micro_texture_std: float = 0.85
    subject: SubjectProfile = field(default_factory=SubjectProfile)
    canvas: CanvasSpec = field(default_factory=CanvasSpec)

    def __post_init__(self) -> None:
        if not (0.72 <= self.codeformer_fidelity_w <= 0.98):
            raise ValueError(
                f"codeformer_fidelity_w={self.codeformer_fidelity_w} is outside safe identity range [0.72, 0.98]."
            )
        if self.upscale_factor not in (2, 3, 4):
            raise ValueError(f"upscale_factor must be 2, 3, or 4 (got {self.upscale_factor}).")

    def resolve_model_path(self, filename: str) -> Path:
        """Locate a binary or ONNX/NCNN model file in the Google Drive Vault or local directories."""
        candidates = []
        if self.vault_dir:
            candidates.append(self.vault_dir / "models" / filename)
        for env_key in ("REELPROFILE_MODELS_DIR", "MODEL_ENGINE_MODELS_DIR", "PORTRAIT_ENGINE_MODELS_DIR"):
            env_models = os.environ.get(env_key)
            if env_models:
                candidates.append(Path(env_models) / filename)
        candidates.append(Path(__file__).resolve().parent.parent / "models" / filename)

        for cand in candidates:
            if cand.exists():
                return cand
        raise FileNotFoundError(
            f"Required model/binary '{filename}' not found. Checked: {[str(c) for c in candidates]}"
        )


# Backward-compatible alias
SubjectSpec = SubjectProfile
