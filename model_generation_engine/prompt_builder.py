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

    FORBIDDEN_FACIAL_EXAGGERATION_PATTERNS: Sequence[str] = (
        r"(?<!no\s)(?<!zero\s)(?<!without\s)(?<!never\s)(?:plump(?:ed)?|pouty|luscious|fuller|voluminous|juicy|bee[-\s]?stung|glossy\s+red)\s+(?:(?:\w+\s+){0,1})?lips?\b",
        r"(?<!no\s)(?<!zero\s)(?<!without\s)(?<!never\s)(?:big|large|huge|doe|wide[-\s]?open|doll|anime|cat[-\s]?eye|foxy|siren|dramatic)\s+(?:(?:\w+\s+){0,1})?eyes?\b",
        r"(?<!no\s)(?<!zero\s)(?<!without\s)(?<!never\s)(?:sculpted|chiseled|sharp|high|razor|hollow(?:ed)?|contoured|dramatic)\s+(?:(?:\w+\s+){0,1})?(?:cheekbones?|cheeks?|jaw(?:line)?)\b",
        r"(?<!no\s)(?<!zero\s)(?<!without\s)(?<!never\s)(?:button|barbie|upturned|pinched|tiny|slim|narrow|razor[-\s]?sharp)\s+(?:(?:\w+\s+){0,1})?nose\b",
        r"(?<!no\s)(?<!zero\s)(?<!without\s)(?<!never\s)(?:heavy|dramatic|glam(?:orous)?|bold|smokey|editorial|red[-\s]?carpet|full[-\s]?glam)\s+(?:(?:\w+\s+){0,1})?(?:makeup|eyeliner|eyeshadow|lipstick|contour(?:ing)?|brows?)\b",
        r"(?<!no\s)(?<!zero\s)(?<!without\s)(?<!never\s)(?:beautify|beautified|idealized|glamourized|caricature|supermodel\s+face|flawless\s+beauty|perfect\s+symmetrical\s+face)\b",
    )

    @classmethod
    def validate_prompt(cls, prompt: str) -> None:
        """Raise PromptContaminationError if the prompt contains face-bloating, skin-blur, plastic-gloss, forced-wrinkle, or facial-exaggeration terms."""
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
        for pat in cls.FORBIDDEN_FACIAL_EXAGGERATION_PATTERNS:
            for m in re.finditer(pat, lower):
                prefix = lower[max(0, m.start() - 80) : m.start()]
                clause_prefix = re.split(r"[.;\n]", prefix)[-1]
                if re.search(r"\b(?:never|do\s+not|don't|without|zero|no|avoid)\b", clause_prefix):
                    continue
                raise PromptContaminationError(
                    f"Prompt contains forbidden facial-exaggeration / beautification pattern '{pat}'. "
                    "NEVER exaggerate, plump, enlarge, sculpt, or glam-caricature any real facial feature."
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
            "CRITICAL MANDATORY RULE — ZERO FACIAL DEFORMATION & TRUE 3D SCENE HARMONY "
            "(절대 눈코입·얼굴형·피부질감 변형 금지 및 3D 비율·각도·톤앤매너 자연 조화): "
            f"Preserve the exact 1:1 facial identity and natural matte real skin texture of the {subject.subject_identity_label} "
            f"from the reference image ({reference_filename}). "
            "NEVER alter, reshape, slim, widen, beautify, idealize, or deform the subject's face shape, lower cheek contour, "
            "jawline, chin, eyes, eyelid shape, eye corners, inter-ocular distance, nose bridge, nose tip, nostril width, "
            "philtrum, mouth shape, lip thickness, ears, eyewear frame, or hairstyle. "
            "Integrate the subject naturally in 3D perspective matching the target scene's exact camera angle, head turn/tilt, "
            "balanced adult 7.5-to-8-head head-to-shoulder proportions, directional scene lighting, soft shadow falloff, "
            "color temperature, and overall editorial tone & manner (NEVER look flatly pasted or crammed into a cutout mold). "
            f"Scene & Framing: create a photorealistic {camera.to_prompt_clause()}. "
            f"Identity & Eyewear Temple Continuity Lock: preserve {subject.facial_identity_descriptor()}. "
            "Anatomical & Optical Focus Lock: Shot on a medium-format 100MP camera with an 80mm f/5.6 prime lens—keep tack-sharp "
            "crystal-clear optical focus and crisp micro-contrast on the subject's iris, eyelashes, eyebrow hairs, eyewear frame edges, "
            "nose, lips, individual hair strands, and fabric weave, with deep true blacks and ZERO atmospheric haze, milky veil, melting, or smudging. "
            "Skin Expression: crisp, authentic natural real skin texture harmonized with the scene's directional lighting and subtle real pores, "
            "ZERO plastic/oily shine, ZERO dewy gloss, ZERO specular glare on the forehead, nose, or cheeks, and ZERO artificial wrinkles. "
            f"Physical Build: {subject.body_frame_descriptor()}. "
            "Do NOT widen, slim, bloat, distort, or age the subject's face; keep the exact facial proportions of the reference photo. "
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
    def build_tonematched_3d_refinement_prompt(
        cls,
        subject: SubjectProfile,
        scene_context: str,
    ) -> str:
        """Universal 3-pillar Tone-Matched 3D In-Place Upper-Body Refinement prompt (replaces 2D oval pixel pasting)."""
        prompt = (
            f"Edit Image 2 ({scene_context}) into a crisp, high-resolution 85mm medium close-up photograph while keeping "
            "Image 2's exact framing, shoulder position, 3D posture, wardrobe, props, and background environment 100% unchanged. "
            f"1. True 3D Identity & Balanced Proportions: Render the {subject.subject_identity_label} from Image 1 with "
            f"{subject.facial_identity_descriptor()}, naturally rotated in 3D to match Image 2's exact head turn/tilt and "
            "calibrated to balanced adult 7.5-to-8-head head-to-shoulder proportions (never flat, crooked, oversized, or pasted). "
            "2. Eyewear Temple Continuity: If eyewear is present in Image 1, render both temple arms clearly, continuously, "
            "and uninterruptedly extending from the outer front frame hinges all the way back over the tops of both ears. "
            "3. Scene Tone & Manner Harmony: Naturally relight the subject's face and skin to harmonize with Image 2's "
            "directional key light, soft dimensional shadow falloff, color temperature, and overall editorial mood—"
            "preserving natural matte real skin texture with subtle pores, zero plastic/oily shine, zero dewy gloss, "
            "and zero artificial white-studio cutout look. Zero text, zero watermarks."
        )
        cls.validate_prompt(prompt)
        return prompt

    @classmethod
    def build_stage2_closeup_refinement_prompt(
        cls,
        subject: SubjectProfile,
        scene_context: str,
    ) -> str:
        """Universal close-up identity refinement prompt delegating to Tone-Matched 3D In-Place Refinement."""
        return cls.build_tonematched_3d_refinement_prompt(subject, scene_context)
