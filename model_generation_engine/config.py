from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple


def resolve_gdrive_vault(project_names: Sequence[str] = ("모델생성", "인물사진")) -> Optional[Path]:
    """Dynamically locate the Google Drive Heavy Asset Vault across macOS/Windows/Linux."""
    env_override = os.environ.get("MODEL_ENGINE_VAULT") or os.environ.get("PORTRAIT_ENGINE_VAULT")
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
    expression: str = "natural composed expression matching the reference portrait 1:1"
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
            "exact 1:1 facial identity, bone structure, eye shape, iris gaze, nose bridge and tip, "
            "mouth contour, and jawline of the reference portrait",
            self.expression,
        ]
        if self.eyewear is None:
            parts.append("preserving the exact eyewear presence or absence shown in the reference portrait")
        elif self.eyewear.lower() in ("none", "no glasses", "without glasses"):
            parts.append("wearing no glasses (bare eyes)")
        else:
            parts.append(f"wearing {self.eyewear}")

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
    """Universal camera angle, shot scale, focal length, pose, and negative-space specification."""

    shot_scale: str = "three-quarter portrait shot"
    camera_angle: str = "eye-level camera angle"
    focal_length_mm: int = 85
    pose_description: str = "natural composed editorial posture"
    negative_space_side: str = "left"  # "left", "right", "center", "top", "none"
    aspect_ratio_label: str = "16:9"

    def to_prompt_clause(self) -> str:
        neg_map = {
            "left": "subject positioned right-of-center leaving clean negative space on the left side of the frame",
            "right": "subject positioned left-of-center leaving clean negative space on the right side of the frame",
            "top": "subject positioned in the lower two-thirds leaving clean negative space in the upper sky/wall",
            "center": "balanced centered editorial framing",
            "none": "full-frame balanced composition",
        }
        neg_clause = neg_map.get(self.negative_space_side.lower(), self.negative_space_side)
        return (
            f"{self.shot_scale} captured from a {self.camera_angle} with an {self.focal_length_mm}mm lens; "
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
    """Master configuration for 모델생성 (Universal Model Generation & 5K Super-Resolution Engine)."""

    project_name: str = "모델생성"
    vault_dir: Optional[Path] = field(default_factory=lambda: resolve_gdrive_vault(("모델생성", "인물사진")))
    upscale_model_name: str = "4xNomos8kSC"
    upscale_factor: int = 4
    codeformer_fidelity_w: float = 0.82
    unsharp_amount: float = 0.35
    unsharp_sigma: float = 1.0
    max_seam_l1_diff: float = 6.0
    min_face_laplacian_var: float = 25.0
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
        for env_key in ("MODEL_ENGINE_MODELS_DIR", "PORTRAIT_ENGINE_MODELS_DIR"):
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
