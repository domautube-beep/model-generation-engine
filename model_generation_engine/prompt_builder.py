from __future__ import annotations

import re
from typing import Optional, Sequence
from .config import CameraRigSpec, LocationSpec, SubjectProfile, WardrobeSpec


class PromptContaminationError(ValueError):
    """Raised when a prompt couples body-weight/bulk terms with facial features, requests airbrushing, or requests plastic gloss."""


class PromptFirewall:
    """Universal, gender-neutral, model-agnostic Prompt Builder and Safety Firewall.

    Enforces:
    1. Body-vs-Face Decoupling (height/weight applies BELOW THE NECK ONLY).
    2. Anti-Blur / Anti-Airbrush Firewall (blocks porcelain, waxy, beauty-cam filters).
    3. Anti-Plastic-Gloss / Anti-Forced-Wrinkle Firewall (requires natural matte skin with subtle pores).
    """

    FORBIDDEN_FACE_WEIGHT_PATTERNS: Sequence[str] = (
        r"fuller\s+cheeks",
        r"fuller\s+jaw",
        r"chubby\s+face",
        r"heavy\s+face",
        r"round\s+chubby",
        r"wide\s+boxy\s+jaw",
        r"stocky\s+face",
        r"bloated\s+face",
        r"double\s+chin",
        r"slender\s+oval\s+face",
        r"slim\s+oval\s+face",
        r"v[-\s]?line\s+jaw",
        r"chiseled\s+jaw",
        r"narrow\s+face",
        r"small\s+face",
        r"(?<!no\s)(?<!zero\s)(?<!without\s)(?:stocky|heavy|thick|chubby|bloated|puffy|plump|fat|bulky|broad|wide|square|fleshy|beefy|slender|slim|narrow|chiseled|v[-\s]?line)\s+(?:(?:\w+\s+){0,2})?(?:face|cheeks?|jaw(?:line)?|chin|jowls?|neck)\b",
        r"\b(?:face|cheeks?|jaw(?:line)?|chin|jowls?)\s+(?:is|are|looking|with)?\s*(?:stocky|heavy|chubby|bloated|puffy|plump|fat|bulky|wide|square|fleshy|slender|slim|narrow|chiseled)\b",
    )

    FORBIDDEN_BLUR_PATTERNS: Sequence[str] = (
        r"airbrushed\s+skin",
        r"porcelain\s+doll",
        r"plastic\s+skin",
        r"soft\s+focus\s+face",
        r"melted\s+face",
        r"smudged\s+face",
        r"distorted\s+eyes",
        r"(?<!no\s)(?<!zero\s)(?<!without\s)(?:airbrush(?:ed)?|porcelain|plastic|wax(?:y)?|doll[-\s]?like|beauty[-\s]?cam|filter(?:ed)?|over[-\s]?smooth(?:ed)?|blurr?(?:y|ed)|hazy|smudg(?:ed|y)|melt(?:ed|ing)|mush(?:y|ed)|crushed|soft[-\s]?focus|gaussian)\s+(?:(?:\w+\s+){0,2})?(?:skin|face|eyes?|nose|lips?|mouth|features?|complexion|texture|finish|blur)\b",
        r"\b(?:skin|face|eyes?|nose|lips?|complexion)\s+(?:is|are|with|having)?\s*(?:airbrushed|porcelain|plastic|waxy|blurred|hazy|smudged|melted|mushed|over[-\s]?smoothed)\b",
    )

    FORBIDDEN_GLOSS_AND_WRINKLE_PATTERNS: Sequence[str] = (
        r"(?<!no\s)(?<!zero\s)(?<!without\s)(?:dewy|glass[-\s]?skin|oily|greasy|sweaty|glossy|shiny)\s+(?:(?:\w+\s+){0,2})?(?:skin|face|forehead|cheeks?|complexion|finish)\b",
        r"(?<!no\s)(?<!zero\s)(?<!without\s)(?:deep|heavy|pronounced|aged|forced)\s+(?:(?:\w+\s+){0,2})?(?:wrinkles?|nasolabial\s+folds?|creases?|furrows?)\b",
    )

    @classmethod
    def validate_prompt(cls, prompt: str) -> None:
        """Raise PromptContaminationError if the prompt contains face-bloating, skin-blur, plastic-gloss, or forced-wrinkle terms."""
        lower = prompt.lower()
        for pat in cls.FORBIDDEN_FACE_WEIGHT_PATTERNS:
            if re.search(pat, lower):
                raise PromptContaminationError(
                    f"Prompt contains forbidden face-bloating pattern '{pat}'. "
                    "Physical weight/build must NEVER be applied to cheeks or jawline."
                )
        for pat in cls.FORBIDDEN_BLUR_PATTERNS:
            if re.search(pat, lower):
                raise PromptContaminationError(
                    f"Prompt contains forbidden skin-blur pattern '{pat}', which causes hazy/smudged faces."
                )
        for pat in cls.FORBIDDEN_GLOSS_AND_WRINKLE_PATTERNS:
            if re.search(pat, lower):
                raise PromptContaminationError(
                    f"Prompt contains forbidden plastic-gloss or forced-wrinkle pattern '{pat}'. "
                    "Skin must be natural matte texture with subtle pores and zero plastic gloss or forced wrinkles."
                )

    @classmethod
    def build_universal_single_pass_prompt(
        cls,
        subject: SubjectProfile,
        wardrobe: WardrobeSpec,
        location: LocationSpec,
        camera: CameraRigSpec,
        reference_filename: str = "reference_portrait",
    ) -> str:
        """Compose a universal, zero-hardcoding single-pass prompt with Hard Immutable Face Lock for ANY model, outfit, location, and camera angle."""
        prompt = (
            "CRITICAL MANDATORY RULE — ZERO FACIAL & SKIN DEFORMATION (절대 얼굴·눈코입·얼굴형·피부질감·피부톤·피부색감 변형 금지): "
            f"You MUST copy and preserve the exact 1:1 unmodified face AND the exact original skin texture, skin tone, and skin color of the {subject.subject_identity_label} "
            f"from the reference image ({reference_filename}). "
            "NEVER alter, reshape, slim, widen, beautify, idealize, or deform the subject's face shape, lower cheek contour, "
            "jawline, chin, eyes, eyelid shape, eye corners, inter-ocular distance, nose bridge, nose tip, nostril width, "
            "philtrum, mouth shape, lip thickness, ears, eyewear frame, or hairstyle by even 1%, "
            "and NEVER darken, tint, yellow, muddy, or alter the subject's original skin texture, skin brightness, or skin color with background color cast. "
            "The subject's head, face shape, eyes/nose/mouth, and skin tone/texture/color must remain 100% identical to the reference photo as if the exact same head were placed unchanged into the scene. "
            f"Scene & Framing: create a photorealistic {camera.to_prompt_clause()}. "
            f"Identity Lock: preserve {subject.facial_identity_descriptor()}. "
            "Anatomical Focus Lock: keep critical camera focus on the subject's exact reference eyes, nose, and mouth with zero melting or smudging. "
            "Skin Expression: soft, even, natural MATTE skin texture with realistic subtle pores, "
            "ZERO plastic/oily shine, ZERO dewy gloss, ZERO specular glare on the forehead, nose, or cheeks, and ZERO artificial wrinkles. "
            f"Physical Build: {subject.body_frame_descriptor()}. "
            "Do NOT widen, slim, bloat, distort, or age the subject's face; keep the exact face contour of the reference photo. "
            f"Wardrobe: wearing {wardrobe.to_prompt_clause()}. "
            f"Environment & Props: set in {location.to_prompt_clause()}. "
            "Zero text, zero typography, zero watermarks."
        )
        cls.validate_prompt(prompt)
        return prompt

    @classmethod
    def build_stage1_fullbody_prompt(
        cls,
        subject: SubjectProfile,
        scene_description: str,
        wardrobe: Optional[WardrobeSpec] = None,
        camera: Optional[CameraRigSpec] = None,
    ) -> str:
        """Universal wrapper supporting both structured specs and free-form scene description strings."""
        w_spec = wardrobe or WardrobeSpec(outfit_description="contemporary tailored attire matching the scene mood")
        l_spec = LocationSpec(location_description=scene_description)
        c_spec = camera or CameraRigSpec()
        return cls.build_universal_single_pass_prompt(subject, w_spec, l_spec, c_spec)

    @classmethod
    def build_stage2_closeup_refinement_prompt(
        cls,
        subject: SubjectProfile,
        scene_context: str,
    ) -> str:
        """Universal close-up identity refinement prompt without hardcoded gender, clothing, or doorway assumptions."""
        prompt = (
            f"Edit Image 2 ({scene_context}) so that the subject's face and head are an exact 1:1 photorealistic match "
            f"to the {subject.subject_identity_label} in Image 1. "
            f"Preserve {subject.facial_identity_descriptor()}, "
            "and sharp, natural matte photographic skin detail with realistic subtle pores, zero oily shine, and zero dewy gloss. "
            "Keep the exact same framing, head position, neckline/collar, shoulders, and environment background of Image 2 unchanged. "
            "Zero text or watermarks."
        )
        cls.validate_prompt(prompt)
        return prompt
