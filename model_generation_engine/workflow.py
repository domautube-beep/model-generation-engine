from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import CameraRigSpec, LocationSpec, SubjectProfile, WardrobeSpec
from .prompt_builder import PromptFirewall


class GenerationalDegradationError(ValueError):
    """Raised when an already-generated AI image is fed back into AI generation."""


class ReferenceIsolationError(ValueError):
    """Raised when multiple face images or a mood reference containing another person's face is passed into generation."""


@dataclass(frozen=True)
class VariationSpec:
    """A single universal variation combining any WardrobeSpec, LocationSpec, and CameraRigSpec."""

    variation_id: str
    title: str
    wardrobe: WardrobeSpec
    location: LocationSpec
    camera: CameraRigSpec


class SinglePassGenerationGuard:
    """Enforces the universal Single-Pass From-Scratch rule:
    - Never feed a generated image back into generation (generation_depth == 0).
    - Pass ONLY the 1 pure identity reference image (never mix another person's face photo).
    """

    GENERATED_FILENAME_PATTERNS: Sequence[str] = (
        r"^pure_scratch_",
        r"^21x9_",
        r"^refined_",
        r"^dyn_crop_",
        r"^crop_",
        r"^studio_seated_var",
        r"^custom_person_",
        r"^closeup_",
        r"^macro_face_",
        r"^up_src_",
        r"^up_ref_",
        r"^generated_",
        r"^out_5k_",
    )

    @classmethod
    def validate_reference_lineage(
        cls,
        reference_image_paths: Sequence[str | Path],
        generation_depth: int = 0,
        auxiliary_reference_paths: Optional[Sequence[str | Path]] = None,
        face_detector: Optional[object] = None,
    ) -> Path:
        if generation_depth > 0:
            raise GenerationalDegradationError(
                f"generation_depth={generation_depth} is forbidden. "
                "Always generate from scratch (depth=0) to prevent generational degradation and plastic skin gloss."
            )

        if len(reference_image_paths) != 1:
            raise ReferenceIsolationError(
                f"Expected exactly 1 pure identity reference image, got {len(reference_image_paths)}. "
                "Never pass a mood reference photo containing another person's face alongside the identity reference."
            )

        if auxiliary_reference_paths:
            if face_detector is not None:
                import cv2

                for aux_p in auxiliary_reference_paths:
                    aux_img = cv2.imread(str(aux_p), cv2.IMREAD_COLOR)
                    if aux_img is not None:
                        try:
                            face_detector.detect(aux_img)
                            raise ReferenceIsolationError(
                                f"Auxiliary reference image '{Path(aux_p).name}' contains a human face. "
                                "Passing any second image with a human face causes facial contamination."
                            )
                        except ReferenceIsolationError:
                            raise
                        except Exception:
                            pass
            raise ReferenceIsolationError(
                "Auxiliary reference images are forbidden in Single-Pass generation. "
                "Encode outfit, location, and camera angle via WardrobeSpec, LocationSpec, and CameraRigSpec."
            )

        ref_path = Path(reference_image_paths[0])
        fname = ref_path.name.lower()
        for pat in cls.GENERATED_FILENAME_PATTERNS:
            if re.search(pat, fname):
                raise GenerationalDegradationError(
                    f"Input reference '{ref_path.name}' matches generated artifact pattern '{pat}'. "
                    "Feeding a generated image back into generation is strictly forbidden."
                )

        if "target_scene" in fname or "studio_seated" in fname or "mood_ref" in fname:
            raise ReferenceIsolationError(
                f"Input reference '{ref_path.name}' is a scene/mood reference. "
                "Extract its non-facial attributes into LocationSpec/WardrobeSpec/CameraRigSpec and pass ONLY the subject's identity photo."
            )

        return ref_path


class UniversalVariationPlanner:
    """Universal Combinatorial Variation Matrix Planner for ANY model, outfit, location, and camera angle.

    Supports:
    1. Arbitrary user-supplied lists of WardrobeSpec, LocationSpec, and CameraRigSpec.
    2. Built-in multi-domain preset catalogs ("studio_editorial", "architectural_interior",
       "outdoor_natural", "urban_cinematic", "multi_angle_portfolio") so any new model can be
       placed into any domain, outfit, location, or camera angle immediately without hardcoding.
    """

    WARDROBE_CATALOG: Dict[str, Sequence[WardrobeSpec]] = {
        "tailored_editorial": (
            WardrobeSpec("tailored_editorial", "tailored dark navy single-breasted blazer over a crisp white crew-neck top and cream tailored trousers"),
            WardrobeSpec("tailored_editorial", "charcoal-gray double-breasted jacket over a fine black merino turtleneck and charcoal pleated trousers"),
            WardrobeSpec("tailored_editorial", "warm camel wool-cashmere coat layered over an ivory ribbed knit sweater and dark charcoal slacks"),
            WardrobeSpec("tailored_editorial", "tailored midnight-navy waistcoat over a crisp light-blue fine-striped shirt with rolled sleeves", "minimalist leather watch"),
            WardrobeSpec("tailored_editorial", "dark espresso-brown suede shirt-jacket over a cream knit polo and charcoal wool trousers"),
            WardrobeSpec("tailored_editorial", "sharp charcoal-black tailored suit with an open-collar crisp white dress shirt"),
        ),
        "smart_casual": (
            WardrobeSpec("smart_casual", "relaxed oatmeal linen overshirt over a white cotton tee and olive chinos"),
            WardrobeSpec("smart_casual", "soft sage-green cashmere crewneck sweater with tailored cream trousers"),
            WardrobeSpec("smart_casual", "minimalist black boxy workwear jacket over a heather-gray tee and dark denim trousers"),
            WardrobeSpec("smart_casual", "ivory cable-knit cardigan over a crisp white tee and charcoal wide-leg slacks"),
        ),
        "architectural_outerwear": (
            WardrobeSpec("architectural_outerwear", "long structured charcoal wool trench coat over an all-black minimalist ensemble"),
            WardrobeSpec("architectural_outerwear", "tailored sand-beige gabardine mac coat over a dark navy knit and tailored trousers"),
            WardrobeSpec("architectural_outerwear", "minimalist deep-espresso structured overcoat over a charcoal turtleneck"),
        ),
    }

    LOCATION_CATALOG: Dict[str, Sequence[LocationSpec]] = {
        "studio_editorial": (
            LocationSpec("studio_editorial", "solid matte warm terracotta-brown seamless studio backdrop", "#8E472A", "a mid-century walnut wood chair with a woven cane backrest"),
            LocationSpec("studio_editorial", "solid matte muted olive-sage seamless studio backdrop", "#3B493F", "a sleek Bauhaus tubular chrome and black saddle-leather armchair and travertine plinth with a brass lamp"),
            LocationSpec("studio_editorial", "solid matte warm taupe-greige seamless studio backdrop", "#8C8278", "a sculptural solid oak stool and slim matte-black metal side table with two hardcover books and a ceramic cup"),
            LocationSpec("studio_editorial", "solid matte deep slate-navy seamless studio backdrop", "#1E2B3A", "a dark walnut studio stool beside a walnut console holding a brushed-steel geometric sculpture and leather notebook"),
            LocationSpec("studio_editorial", "solid matte deep burgundy-chestnut seamless studio backdrop", "#5A2926", "a low-profile cognac saddle-leather lounge chair and smoked-glass round side table"),
            LocationSpec("studio_editorial", "solid matte warm sand-ivory seamless studio backdrop", "#D8D0C5", "a minimalist matte-black ashwood design chair and low concrete plinth with a matte-white ceramic vessel"),
        ),
        "architectural_interior": (
            LocationSpec("architectural_interior", "sunlit Tadao Ando-inspired board-formed concrete gallery corridor with soft natural daylight beams", None, "a minimalist raw oak bench"),
            LocationSpec("architectural_interior", "modern minimalist private library with floor-to-ceiling warm walnut shelving and diffused skylight", None, "an architectural reading table with a single open art monograph"),
            LocationSpec("architectural_interior", "contemporary glass-and-limestone atrium overlooking a serene courtyard garden", None, "a cantilevered travertine ledge"),
            LocationSpec("architectural_interior", "minimalist studio doorway portal with a bold vermilion-red architectural wall and warm pastel interior glow", "#C8281E", "a clean architectural doorway frame"),
        ),
        "outdoor_natural": (
            LocationSpec("outdoor_natural", "vast minimalist white salt flat under a clear deep cobalt sky at golden hour", None, "a freestanding minimalist architectural portal frame", "warm low-angle golden hour sunlight with long soft shadows"),
            LocationSpec("outdoor_natural", "serene coastal basalt cliff terrace overlooking a calm misty horizon at dawn", None, "a low weathered stone parapet", "soft diffused dawn sky light"),
            LocationSpec("outdoor_natural", "minimalist modern courtyard with reflection pond and pale travertine colonnade", None, "calm water surface reflecting architectural geometry", "clean natural daylight"),
        ),
        "urban_cinematic": (
            LocationSpec("urban_cinematic", "minimalist white architectural plaza with bold vermilion-orange crosswalk stripes", None, "long dramatic architectural cast shadows", "crisp directional afternoon sunlight"),
            LocationSpec("urban_cinematic", "modernist financial district colonnade with warm limestone pillars and soft morning light", None, "clean geometric pavement lines", "soft morning directional light"),
        ),
    }

    CAMERA_CATALOG: Dict[str, Sequence[CameraRigSpec]] = {
        "seated_editorial": (
            CameraRigSpec("three-quarter seated portrait shot", "eye-level camera angle", 85, "seated, leaning slightly forward with forearms resting comfortably", "left"),
            CameraRigSpec("three-quarter seated portrait shot", "eye-level camera angle", 85, "seated comfortably upright in a composed editorial posture", "left"),
            CameraRigSpec("medium three-quarter seated shot", "slight 15-degree 3/4 camera angle", 85, "seated in a relaxed intellectual posture", "left"),
        ),
        "multi_angle_portfolio": (
            CameraRigSpec("medium close-up portrait shot", "eye-level frontal camera angle", 85, "composed upright posture looking directly toward the lens", "left"),
            CameraRigSpec("three-quarter portrait shot", "35-degree 3/4 profile camera angle", 85, "relaxed three-quarter turn with natural shoulder line", "left"),
            CameraRigSpec("full-body wide editorial shot", "subtle low-angle heroic camera angle", 50, "confident forward-stepping stride", "left"),
            CameraRigSpec("medium portrait shot", "slight high-angle editorial perspective", 85, "leaning lightly against an architectural surface", "right"),
            CameraRigSpec("three-quarter side-profile shot", "75-degree side-profile camera angle", 105, "calm observational gaze toward the open negative space", "left"),
        ),
    }

    @classmethod
    def compose_custom_variations(
        cls,
        wardrobes: Sequence[WardrobeSpec],
        locations: Sequence[LocationSpec],
        cameras: Sequence[CameraRigSpec],
        max_count: int = 6,
    ) -> List[VariationSpec]:
        """Dynamically pair any arbitrary lists of wardrobes, locations, and camera rigs into a variation list."""
        if not wardrobes or not locations or not cameras:
            raise ValueError("wardrobes, locations, and cameras must each contain at least 1 spec.")

        n = min(max_count, max(len(wardrobes), len(locations), len(cameras)))
        variations: List[VariationSpec] = []
        for i in range(n):
            w = wardrobes[i % len(wardrobes)]
            loc = locations[i % len(locations)]
            cam = cameras[i % len(cameras)]
            vid = f"{i + 1:02d}"
            title = f"[{loc.environment_category}] {loc.location_description[:36]} | {w.outfit_description[:36]} | {cam.camera_angle}"
            variations.append(
                VariationSpec(
                    variation_id=vid,
                    title=title,
                    wardrobe=w,
                    location=loc,
                    camera=cam,
                )
            )
        return variations

    @classmethod
    def build_single_pass_variation_prompt(
        cls,
        subject: SubjectProfile,
        variation: VariationSpec,
        reference_image_paths: Sequence[str | Path],
        generation_depth: int = 0,
    ) -> str:
        """Validate single-pass reference lineage and build a universal, firewall-verified prompt."""
        ref_path = SinglePassGenerationGuard.validate_reference_lineage(
            reference_image_paths, generation_depth=generation_depth
        )
        return PromptFirewall.build_universal_single_pass_prompt(
            subject=subject,
            wardrobe=variation.wardrobe,
            location=variation.location,
            camera=variation.camera,
            reference_filename=ref_path.name,
        )

    @classmethod
    def plan_variation_batch(
        cls,
        reference_face_path: str | Path,
        subject: Optional[SubjectProfile] = None,
        wardrobe_preset: str = "tailored_editorial",
        location_preset: str = "studio_editorial",
        camera_preset: str = "seated_editorial",
        custom_variations: Optional[Sequence[VariationSpec]] = None,
        max_count: int = 6,
    ) -> List[dict]:
        """Generate a batch of validated single-pass from-scratch generation plans for ANY model/outfit/location/angle."""
        subj = subject or SubjectProfile()
        if custom_variations is not None:
            vars_use = list(custom_variations)[:max_count]
        else:
            w_list = cls.WARDROBE_CATALOG.get(wardrobe_preset, cls.WARDROBE_CATALOG["tailored_editorial"])
            l_list = cls.LOCATION_CATALOG.get(location_preset, cls.LOCATION_CATALOG["studio_editorial"])
            c_list = cls.CAMERA_CATALOG.get(camera_preset, cls.CAMERA_CATALOG["seated_editorial"])
            vars_use = cls.compose_custom_variations(w_list, l_list, c_list, max_count=max_count)

        plans = []
        for var in vars_use:
            prompt = cls.build_single_pass_variation_prompt(
                subject=subj,
                variation=var,
                reference_image_paths=[reference_face_path],
                generation_depth=0,
            )
            plans.append(
                {
                    "variation_id": var.variation_id,
                    "title": var.title,
                    "wardrobe": var.wardrobe.to_prompt_clause(),
                    "location": var.location.to_prompt_clause(),
                    "camera": var.camera.to_prompt_clause(),
                    "reference_images": [str(reference_face_path)],
                    "generation_depth": 0,
                    "prompt": prompt,
                }
            )
        return plans


# Backward-compatible alias
VariationMatrixPlanner = UniversalVariationPlanner
