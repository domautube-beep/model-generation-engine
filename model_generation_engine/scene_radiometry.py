"""Universal Single-Camera Optical & Radiometric Cohesion Engine (`scene_radiometry.py`).

Solves the multi-reference "floating / pasted catalog cutout" problem in `1원본들 → 1생성` workflows by:
1. Quantitatively measuring the target scene's illumination physics directly from any uploaded mood/location photo
   (or synthesizing from lighting/mood text when no photo is provided):
   - Color temperature (`Kelvin`) & highlight/shadow CIE LAB chromaticity cast
   - 3x3 spatial luminance vector (`key_light_vector`, `lighting_regime`: backlit chiaroscuro, side window key, golden sunbeam, etc.)
   - Key-to-Fill contrast ratio (in photographic stops) & shadow black-point floor (deep vs. warm filmic lifted shadows)
   - Optical highlight bloom/halation & edge-vs-center Laplacian Depth-of-Field (`f-stop`) profile
2. Separating **Intrinsic Subject/Product Albedo & 3D Geometry** from **Extrinsic Studio Reference Lighting**:
   - Strips away flat `5600K` studio flash from Face Anchors (`Image 1–3`) and pure `#FFFFFF` shadowless e-commerce lighting
     from garment, watch, bag, and shoe product photos.
   - Re-renders the entire frame through ONE physical camera sensor, ONE prime lens focal plane, ONE unified directional light
     vector, realistic contact ambient occlusion (`접촉 그림자`), and environmental color spill (`반사·색 번짐`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

LIGHTING_MODES: Dict[str, Dict[str, object]] = {
    "auto": {
        "label_ko": "자동 광학 추출 (장소·레퍼런스 빛 자동 계측)",
    },
    "natural_window_backlight": {
        "label_ko": "자연 창가 역광·측후방광 (입체 명암 + 림라이트)",
        "kelvin_est": 3300,
        "contrast_ratio_stops": 3.4,
        "shadow_floor_luma": 22.0,
        "highlight_bloom": 0.16,
        "lighting_regime": "backlit_chiaroscuro",
        "key_light_vector": "warm directional sunlight streaming from upper-rear/side window blinds, wrapping hair/ear/shoulder rims in golden highlight while placing the camera-facing cheek and torso in rich luminous shadow",
        "color_cast_summary": "warm golden-amber highlight tint (~3300K) paired with deep warm umber-olive shadows (zero neutral studio white)",
        "recommended_aperture": "f/2.0",
    },
    "warm_golden_sunbeam": {
        "label_ko": "따뜻한 오후 직사광·그림자 패턴 (골든아워)",
        "kelvin_est": 3500,
        "contrast_ratio_stops": 3.1,
        "shadow_floor_luma": 26.0,
        "highlight_bloom": 0.12,
        "lighting_regime": "directional_sunbeam",
        "key_light_vector": "angled late-afternoon sunbeams casting crisp yet organic architectural light-and-shadow geometry across the room, subject, and furniture",
        "color_cast_summary": "warm sunlit ivory-gold highlights (~3500K) and warm walnut-tinted midtones/shadows",
        "recommended_aperture": "f/2.2",
    },
    "soft_diffused_daylight": {
        "label_ko": "부드러운 자연 확산광 (창가·야외 그늘)",
        "kelvin_est": 5000,
        "contrast_ratio_stops": 2.1,
        "shadow_floor_luma": 32.0,
        "highlight_bloom": 0.05,
        "lighting_regime": "directional_side_key",
        "key_light_vector": "large single-side natural daylight source creating a smooth 2-stop luminance gradient from the lit cheekbone across the nasal bridge to the shaded cheek",
        "color_cast_summary": "organic daylight white balance (~5000K) with subtle ambient room/foliage color bounce in shadow folds",
        "recommended_aperture": "f/2.5",
    },
    "moody_chiaroscuro_interior": {
        "label_ko": "무디 로우키 실내광 (깊은 음영·시네마틱)",
        "kelvin_est": 3000,
        "contrast_ratio_stops": 3.8,
        "shadow_floor_luma": 16.0,
        "highlight_bloom": 0.10,
        "lighting_regime": "moody_low_key",
        "key_light_vector": "focused warm architectural side-top light carving dimensional chiaroscuro relief with deep enveloping ambient room shadows",
        "color_cast_summary": "intimate tungsten-amber warmth (~3000K) with deep rich espresso-toned shadows",
        "recommended_aperture": "f/2.0",
    },
    "cool_overcast_architectural": {
        "label_ko": "차분한 건축 주광 (모던 콘크리트·갤러리)",
        "kelvin_est": 5800,
        "contrast_ratio_stops": 1.9,
        "shadow_floor_luma": 34.0,
        "highlight_bloom": 0.03,
        "lighting_regime": "diffused_architectural",
        "key_light_vector": "directional skylight/clerestory daylight from upper side with soft grounded shadow falloff toward the floor",
        "color_cast_summary": "clean architectural stone/daylight palette (~5800K) with cool-neutral slate shadows",
        "recommended_aperture": "f/2.8",
    },
}

APERTURE_MODES: Dict[str, str] = {
    "auto": "",
    "f1_8": "f/1.8",
    "f2_0": "f/2.0",
    "f2_8": "f/2.8",
    "f4_0": "f/4.0",
}

COHESION_MASTERING_MODES: Dict[str, Dict[str, object]] = {
    "auto": {
        "label_ko": "자동 광학 일체화 (장면·레퍼런스 톤앤매너 100% 동기화)",
        "strength": 1.0,
        "contrast_boost": 0.11,
        "bloom_strength": 0.055,
        "light_wrap_strength": 0.05,
        "grain_sigma": 1.05,
        "temp_bias_b": 0.0,
        "tint_bias_a": 0.0,
    },
    "warm_cinema_amber": {
        "label_ko": "웜 시네마 앰버 (따뜻한 오후 창빛·필름 무드)",
        "strength": 1.10,
        "contrast_boost": 0.14,
        "bloom_strength": 0.075,
        "light_wrap_strength": 0.065,
        "grain_sigma": 1.20,
        "temp_bias_b": 6.5,
        "tint_bias_a": 1.8,
    },
    "deep_chiaroscuro": {
        "label_ko": "시네마틱 딥 명암 (들뜬 암부 밀착·입체 대비 강화)",
        "strength": 1.15,
        "contrast_boost": 0.18,
        "bloom_strength": 0.05,
        "light_wrap_strength": 0.055,
        "grain_sigma": 1.15,
        "temp_bias_b": 2.0,
        "tint_bias_a": 0.5,
    },
    "soft_editorial_natural": {
        "label_ko": "소프트 내추럴 필름 (은은한 톤 통합·부드러운 계조)",
        "strength": 0.72,
        "contrast_boost": 0.06,
        "bloom_strength": 0.04,
        "light_wrap_strength": 0.04,
        "grain_sigma": 0.85,
        "temp_bias_b": 0.0,
        "tint_bias_a": 0.0,
    },
    "off": {
        "label_ko": "후처리 끄기 (생성 원본 그대로)",
        "strength": 0.0,
        "contrast_boost": 0.0,
        "bloom_strength": 0.0,
        "light_wrap_strength": 0.0,
        "grain_sigma": 0.0,
        "temp_bias_b": 0.0,
        "tint_bias_a": 0.0,
    },
}


@dataclass(frozen=True)
class SingleLensRadiometricSpec:
    """Quantitative & semantic optical specification unifying subject, garments, props, and space into ONE camera capture."""

    kelvin_est: int = 4400
    contrast_ratio_stops: float = 2.5
    shadow_floor_luma: float = 26.0
    median_luma: float = 108.0
    highlight_p95_luma: float = 218.0
    highlight_bloom: float = 0.07
    lab_a_cast: float = 3.0
    lab_b_cast: float = 8.0
    lighting_regime: str = "directional_side_key"
    key_light_vector: str = (
        "single directional natural key light from upper-side creating a clear 2.5-stop highlight-to-shadow gradient across the subject and space"
    )
    color_cast_summary: str = (
        "unified natural color temperature across highlights, midtones, and shadows (zero sterile neutral-white studio cutout tones)"
    )
    recommended_aperture: str = "f/2.4"
    source: str = "measured_scene_radiometry"

    def to_single_lens_header(self, focal_length_mm: int = 85) -> str:
        """Emit a physics-true single-camera, single-lens optical capture header (replacing rigid f/16 studio boilerplate)."""
        if self.highlight_bloom >= 0.09 or self.lighting_regime in {"backlit_chiaroscuro", "directional_sunbeam", "moody_low_key"}:
            optics_feel = (
                f"captured through ONE single full-frame camera and ONE {focal_length_mm}mm {self.recommended_aperture} prime lens "
                f"at {self.kelvin_est}K color balance—featuring organic optical highlight roll-off, subtle natural sunlight bloom along "
                "high-contrast window/rim edges, rich tonal shadow gradation, and fine unified full-frame sensor grain across the entire frame"
            )
        else:
            optics_feel = (
                f"captured through ONE single full-frame camera and ONE {focal_length_mm}mm {self.recommended_aperture} prime lens "
                f"at {self.kelvin_est}K color balance—featuring cohesive natural contrast, smooth optical depth-of-field roll-off, "
                "authentic shadow depth, and unified organic full-frame sensor response across every pixel"
            )
        return f"Editorial portrait. Authentic single-exposure location photograph {optics_feel}."

    def to_relighting_and_cohesion_clause(
        self,
        has_catalog_items: bool = True,
        has_placed_bag: bool = False,
    ) -> str:
        """Emit the 5-Law Single-Lens Relighting & Tone-and-Manner Unity directive (`전체 무드앤톤·조명·렌즈 완전 일체화`)."""
        shadow_side_rule = (
            "In this backlit/high-contrast scene, keep the camera-facing side of the face, neck, and chest in rich warm luminous "
            f"shadow (~{self.contrast_ratio_stops:.1f} stops darker than the sunlit rim) while bright directional sunlight catches "
            "the hair crown, ear edge, shoulder contour, eyewear rim, and forearm/hands—NEVER illuminate both cheeks evenly with front fill light"
            if self.lighting_regime in {"backlit_chiaroscuro", "moody_low_key"}
            else f"Maintain a clear {self.contrast_ratio_stops:.1f}-stop directional key-to-shadow ratio across the two sides of the face, "
            "nasal bridge, jawline, and torso folds—NEVER flatten the face with symmetrical front-facing passport flash"
        )

        item_relight_rule = ""
        if has_catalog_items:
            item_relight_rule = (
                " 2) ANTI-CATALOG-CUTOUT GARMENT & PROP RELIGHTING (고유 재질만 유지 · 스튜디오 RGB 고정 해제): "
                "Attached clothing, watch, bag, and footwear photos were shot on bright #FFFFFF e-commerce tables—preserve ONLY their "
                "intrinsic weave, seam construction, pattern geometry, and dye family, and COMPLETELY DISCARD their bright white studio "
                "exposure and neutral white balance! Do NOT lock studio RGB values: white, ivory, or cream fabrics MUST shift into "
                f"{self.color_cast_summary} and drop naturally in brightness inside shaded folds; metallic watch cases/dials and leather "
                "bags MUST reflect this room's exact warm/cool environment and fall into dark ambient shadow when positioned away from the key light."
            )

        placed_prop_rule = ""
        if has_placed_bag:
            placed_prop_rule = (
                " Because the bag rests in the environment rather than in bright front light, angle it naturally in the room's 3D floor/surface "
                "perspective, darken its shaded planes to match adjacent furniture shadows, cast a deep contact occlusion shadow underneath its base, "
                "and soften it slightly according to its distance from the lens focal plane so it NEVER looks like a sharp pasted catalog sticker."
            )

        return (
            "CRITICAL SINGLE-CAMERA RADIOMETRIC & LIGHTING COHESION OVERRIDE (단일 카메라·단일 광원 완전 재조명 — 스튜디오 누끼/합성 티 절대 금지): "
            "1) STRIP STUDIO FLASH FROM FACE ANCHORS: Images 1–3 (face turnaround anchors) were photographed under flat 5600K white studio lamps—"
            "extract ONLY 3D craniofacial bone structure, eye/nose/lip/ear geometry, hair structure, and eyewear silhouette; 100% DISCARD their "
            "flat white studio lighting, cool white balance, and studio catchlights. Relight the subject from scratch under this scene's exact light: "
            f"{self.key_light_vector}. {shadow_side_rule}. Eye catchlights and subtle eyeglass reflections must match ONLY this scene's light source."
            f"{item_relight_rule}"
            f"{placed_prop_rule} "
            "3) ENVIRONMENTAL COLOR SPILL & CONTACT OCCLUSION (반사·색 번짐 및 접촉 그림자): Cast real physical shadows from the head onto the collar/neck, "
            "from lapels onto the inner shirt, and deep dark ambient-occlusion contact shadows where forearms/hands rest on tables, thighs sink into "
            "cushions, or shoes/bags touch the floor; allow warm/cool color bounce from surrounding wood, walls, or sunlit surfaces onto lower chin, "
            "jawline, sleeves, and fabric folds. "
            f"4) THREE-PLANE SPATIAL DEPTH & SINGLE-LENS DEPTH OF FIELD ({self.recommended_aperture} 3단 공간 깊이감 · 대구경 단렌즈 심도): "
            "Compose with real architectural depth—include subtle out-of-focus foreground framing cues near the camera edge, "
            "lock ONE continuous optical focal plane on the subject's eyes and active foreground hand/task, and progressively "
            "defocus background walls, window slats, and distant objects into creamy dimensional bokeh according to real optical "
            "circle-of-confusion—NEVER paste the subject flat against a 2D wall or render foreground, subject, floor bag, and "
            "distant background all in hyper-sharp f/16 focus. "
            f"5) UNIFIED COLOR GRADING & SENSOR FINISH: Apply {self.color_cast_summary} and one shared fine optical grain curve across skin, hair, "
            "garments, accessories, bag, and architecture with zero bright cutout halo around shoulders or hair."
        )


class SceneRadiometryAnalyzer:
    """Deterministically measures single-lens lighting, Kelvin cast, shadow contrast, bloom, and DOF from any scene image or prompt."""

    @classmethod
    def analyze(
        cls,
        scene_bgr: Optional[np.ndarray] = None,
        mood_prompt: str = "",
        lighting_note: str = "",
        lighting_mode: str = "auto",
        aperture_mode: str = "auto",
    ) -> SingleLensRadiometricSpec:
        """Compute `SingleLensRadiometricSpec` from manual preset override, reference image pixels, or text mood."""
        forced_ap = APERTURE_MODES.get(aperture_mode, "")

        # 1. Explicit non-auto lighting mode selected by user
        if lighting_mode in LIGHTING_MODES and lighting_mode != "auto":
            preset = LIGHTING_MODES[lighting_mode]
            return SingleLensRadiometricSpec(
                kelvin_est=int(preset["kelvin_est"]),  # type: ignore[arg-type]
                contrast_ratio_stops=float(preset["contrast_ratio_stops"]),  # type: ignore[arg-type]
                shadow_floor_luma=float(preset["shadow_floor_luma"]),  # type: ignore[arg-type]
                highlight_bloom=float(preset["highlight_bloom"]),  # type: ignore[arg-type]
                lighting_regime=str(preset["lighting_regime"]),
                key_light_vector=str(preset["key_light_vector"]),
                color_cast_summary=str(preset["color_cast_summary"]),
                recommended_aperture=forced_ap or str(preset["recommended_aperture"]),
                source=f"preset_{lighting_mode}",
            )

        # 2. Quantitative pixel radiometry when a location/mood reference image is provided
        if scene_bgr is not None and scene_bgr.size > 0:
            return cls._analyze_bgr(scene_bgr, forced_aperture=forced_ap)

        # 3. Semantic inference from text mood/lighting prompt when no reference image is uploaded
        return cls._infer_from_text(f"{mood_prompt} {lighting_note}".strip(), forced_aperture=forced_ap)

    @classmethod
    def _analyze_bgr(cls, bgr: np.ndarray, forced_aperture: str = "") -> SingleLensRadiometricSpec:
        h, w = bgr.shape[:2]
        scale = min(1.0, 640.0 / float(max(h, w)))
        work = cv2.resize(bgr, (max(32, int(w * scale)), max(32, int(h * scale))), interpolation=cv2.INTER_AREA)
        wh, ww = work.shape[:2]

        lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_ch = lab[:, :, 0]  # 0..255
        a_shift = float(np.median(lab[:, :, 1]) - 128.0)
        b_shift = float(np.median(lab[:, :, 2]) - 128.0)

        p05, p15, p50, p85, p95 = [float(v) for v in np.percentile(l_ch, [5, 15, 50, 85, 95])]
        bright_frac = float(np.mean(l_ch >= 215.0))
        dark_frac = float(np.mean(l_ch <= 55.0))

        # Estimate dynamic contrast ratio in photographic stops
        contrast_stops = float(np.clip(np.log2(max(1.2, (p95 + 12.0) / max(14.0, p15 + 8.0))), 1.4, 4.5))

        # Estimate Kelvin color temperature from LAB b* (yellow-blue) and a* (magenta-green)
        kelvin = int(round(np.clip(5400.0 - b_shift * 145.0 - a_shift * 38.0, 2850.0, 6800.0)))

        # Spatial 3x3 luminance grid to detect key light direction & window backlight
        gh, gw = max(1, wh // 3), max(1, ww // 3)
        top_left = float(np.mean(l_ch[:gh, :gw]))
        top_mid = float(np.mean(l_ch[:gh, gw : 2 * gw]))
        top_right = float(np.mean(l_ch[:gh, 2 * gw :]))
        mid_left = float(np.mean(l_ch[gh : 2 * gh, :gw]))
        center_luma = float(np.mean(l_ch[gh : 2 * gh, gw : 2 * gw]))
        mid_right = float(np.mean(l_ch[gh : 2 * gh, 2 * gw :]))

        left_energy = 0.55 * top_left + 0.45 * mid_left
        right_energy = 0.55 * top_right + 0.45 * mid_right
        upper_bg_energy = max(top_left, top_mid, top_right)

        if left_energy > right_energy + 12.0:
            side_desc = "upper-left"
        elif right_energy > left_energy + 12.0:
            side_desc = "upper-right"
        else:
            side_desc = "upper-center/rear"

        # Detect backlit chiaroscuro (bright window/background highlights + darker shaded center subject)
        is_backlit = (bright_frac >= 0.06 and p50 < 118.0 and upper_bg_energy > center_luma + 16.0) or (
            p95 >= 220.0 and p50 <= 98.0 and contrast_stops >= 2.6
        )
        if is_backlit:
            regime = "backlit_chiaroscuro"
            key_vec = (
                f"strong natural window backlight and angled {side_desc} sunbeams streaming from behind/beside the subject—"
                "creating golden rim highlights along hair, ear, shoulder, and forearm edges while placing the camera-facing "
                "cheek and chest in warm luminous shadow with subtle upward bounce light from the foreground surface"
            )
        elif p50 < 78.0 and dark_frac > 0.35:
            regime = "moody_low_key"
            key_vec = (
                f"intimate low-key directional {side_desc} light carving deep dimensional shadows across the far side of the face and room"
            )
        elif contrast_stops >= 2.55:
            regime = "directional_sunbeam"
            key_vec = (
                f"directional {side_desc} natural sunlight casting clear architectural light-and-shadow gradients across walls, furniture, face, and outfit"
            )
        else:
            regime = "directional_side_key"
            key_vec = (
                f"soft directional {side_desc} natural light creating a smooth three-dimensional luminance falloff from the lit cheek to the shadow side"
            )

        # Shadow & Highlight chromaticity cast description
        sh_mask = l_ch <= max(45.0, p15)
        hi_mask = l_ch >= min(210.0, p85)
        sh_b = float(np.median(lab[:, :, 2][sh_mask]) - 128.0) if np.any(sh_mask) else b_shift
        hi_b = float(np.median(lab[:, :, 2][hi_mask]) - 128.0) if np.any(hi_mask) else b_shift

        if b_shift >= 8.0:
            color_summary = (
                f"unified warm golden-amber {kelvin}K color cast (CIE b* {b_shift:+.1f}) where highlights glow warm ivory-amber "
                f"(b* {hi_b:+.1f}) and shadows settle into rich warm umber-brown/olive tones (b* {sh_b:+.1f}, black floor L={p05:.0f})—"
                "completely eliminating cool #FFFFFF catalog whites"
            )
        elif b_shift <= -5.0:
            color_summary = (
                f"unified cool architectural {kelvin}K daylight cast (CIE b* {b_shift:+.1f}) with crisp cool-slate shadow undertones "
                f"(black floor L={p05:.0f}) across skin, garments, and space"
            )
        else:
            color_summary = (
                f"cohesive natural {kelvin}K editorial color balance (CIE b* {b_shift:+.1f}, shadow floor L={p05:.0f}) "
                "sharing identical warm-neutral shadow roll-off across skin, textiles, leather, and room surfaces"
            )

        # Estimate Depth-of-Field aperture from periphery-vs-center high-frequency Laplacian ratio
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        center_sharp = float(np.mean(lap[gh : 2 * gh, gw : 2 * gw])) + 1e-4
        border_mask = np.ones_like(lap, dtype=bool)
        border_mask[gh // 2 : wh - gh // 2, gw // 2 : ww - gw // 2] = False
        border_sharp = float(np.mean(lap[border_mask]))
        dof_ratio = border_sharp / center_sharp

        if forced_aperture:
            ap = forced_aperture
        elif dof_ratio < 0.58 or is_backlit:
            ap = "f/2.0"
        elif dof_ratio < 0.82:
            ap = "f/2.4"
        else:
            ap = "f/2.8"

        return SingleLensRadiometricSpec(
            kelvin_est=kelvin,
            contrast_ratio_stops=round(contrast_stops, 2),
            shadow_floor_luma=round(p05, 1),
            median_luma=round(p50, 1),
            highlight_p95_luma=round(p95, 1),
            highlight_bloom=round(bright_frac, 3),
            lab_a_cast=round(a_shift, 1),
            lab_b_cast=round(b_shift, 1),
            lighting_regime=regime,
            key_light_vector=key_vec,
            color_cast_summary=color_summary,
            recommended_aperture=ap,
            source="measured_scene_radiometry",
        )

    @classmethod
    def _infer_from_text(cls, text: str, forced_aperture: str = "") -> SingleLensRadiometricSpec:
        low = text.lower()
        warm_kw = ("warm", "walnut", "golden", "sunset", "sunlit", "amber", "wood", "lounge", "terracotta", "afternoon", "따뜻", "노을", "우드", "원목", "창가", "역광")
        cool_kw = ("cool", "concrete", "overcast", "slate", "nordic", "rain", "twilight", "blue", "차분", "콘크리트", "새벽")
        backlit_kw = ("backlit", "backlight", "louver", "blind", "rim light", "역광", "블라인드", "루버")

        if any(k in low for k in backlit_kw):
            base = LIGHTING_MODES["natural_window_backlight"]
        elif any(k in low for k in warm_kw):
            base = LIGHTING_MODES["warm_golden_sunbeam"]
        elif any(k in low for k in cool_kw):
            base = LIGHTING_MODES["cool_overcast_architectural"]
        else:
            base = LIGHTING_MODES["soft_diffused_daylight"]

        return SingleLensRadiometricSpec(
            kelvin_est=int(base["kelvin_est"]),  # type: ignore[arg-type]
            contrast_ratio_stops=float(base["contrast_ratio_stops"]),  # type: ignore[arg-type]
            shadow_floor_luma=float(base["shadow_floor_luma"]),  # type: ignore[arg-type]
            highlight_bloom=float(base["highlight_bloom"]),  # type: ignore[arg-type]
            lighting_regime=str(base["lighting_regime"]),
            key_light_vector=str(base["key_light_vector"]),
            color_cast_summary=str(base["color_cast_summary"]),
            recommended_aperture=forced_aperture or str(base["recommended_aperture"]),
            source="inferred_from_scene_text",
        )


class PostGenCohesionMasterer:
    """Deterministic Post-Generation Single-Camera Optical & Radiometric Mastering Engine (`생성 후 무드·톤·명암·색감 일체화`).

    Harmonizes mood, tonal contrast, directional irradiance, white-point chrominance, optical light-wrap,
    subtle highlight bloom, and unified sensor grain AFTER generation—with ZERO AI re-generation (`Depth == 1` preserved)
    and ZERO facial geometry warp (`SFace` identity 100% preserved).
    """

    @classmethod
    def master(
        cls,
        canvas_bgr: np.ndarray,
        face_det=None,
        reference_scene_bgr: Optional[np.ndarray] = None,
        target_spec: Optional[SingleLensRadiometricSpec] = None,
        mode: str = "auto",
        seed: int = 42,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """Master `canvas_bgr` so subject, garments, accessories, bag, and room share ONE lens & color science."""
        cfg = COHESION_MASTERING_MODES.get(mode, COHESION_MASTERING_MODES["auto"])
        strength = float(cfg["strength"])  # type: ignore[arg-type]
        if strength <= 0.001 or canvas_bgr is None or canvas_bgr.size == 0:
            return canvas_bgr, {"mode": mode, "applied": False}

        h, w = canvas_bgr.shape[:2]
        small_w, small_h = max(64, w // 8), max(36, h // 8)

        lab = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]
        A = lab[:, :, 1] - 128.0
        B = lab[:, :, 2] - 128.0

        # ── Build soft anatomical masks (Subject Foreground Mask, Facial Skin Damper, Inner Organ Shield) ──
        subj_mask_s = np.zeros((small_h, small_w), dtype=np.float32)
        skin_damper = np.zeros((h, w), dtype=np.float32)
        organ_shield = np.zeros((h, w), dtype=np.float32)

        if face_det is not None:
            fx, fy, fw, fh = face_det.bbox
            cx_f = fx + 0.5 * fw
            cy_f = fy + 0.5 * fh
            eye_d = max(10.0, float(face_det.inter_ocular_distance))

            # Broad upper-and-lower body subject zone on 1/8th grid
            sx = int(round(cx_f * small_w / float(w)))
            sy = int(round((cy_f + 2.2 * fh) * small_h / float(h)))
            rx = max(8, int(round(2.3 * fw * small_w / float(w))))
            ry = max(12, int(round(3.6 * fh * small_h / float(h))))
            cv2.ellipse(subj_mask_s, (sx, sy), (rx, ry), 0.0, 0, 360, 1.0, -1)
            subj_mask_s = cv2.GaussianBlur(subj_mask_s, (0, 0), max(3.0, rx * 0.28))

            # Face skin damper (keeps physiological skin blood-tone natural while still matching scene light)
            cv2.ellipse(
                skin_damper,
                (int(round(cx_f)), int(round(cy_f))),
                (int(round(0.56 * fw)), int(round(0.62 * fh))),
                float(face_det.eye_tilt_degrees),
                0,
                360,
                1.0,
                -1,
            )
            k_sk = max(7, int(round(0.22 * fw)) | 1)
            skin_damper = cv2.GaussianBlur(skin_damper, (k_sk, k_sk), max(2.5, 0.08 * fw))

            # Inner facial organ shield (protects eyes, eyewear rims, nose tip, lips from any bloom softening)
            le, re, nose, lm, rm = face_det.landmarks_5pt
            r_org = max(6, int(round(0.36 * eye_d)))
            for pt in (le, re, nose, 0.5 * (lm + rm)):
                cv2.circle(organ_shield, (int(round(pt[0])), int(round(pt[1]))), r_org, 1.0, -1)
            k_org = max(5, int(round(0.18 * eye_d)) | 1)
            organ_shield = cv2.GaussianBlur(organ_shield, (k_org, k_org), max(1.5, 0.05 * eye_d))
        else:
            cv2.ellipse(
                subj_mask_s,
                (small_w // 2, int(small_h * 0.58)),
                (int(small_w * 0.24), int(small_h * 0.40)),
                0.0,
                0,
                360,
                1.0,
                -1,
            )
            subj_mask_s = cv2.GaussianBlur(subj_mask_s, (0, 0), max(4.0, small_w * 0.06))

        subj_mask = cv2.resize(subj_mask_s, (w, h), interpolation=cv2.INTER_LINEAR)

        # ── Stage 1: Measure Peripheral Scene Environment + Optional Reference Scene Target Envelope ──
        lab_s = cv2.resize(lab, (small_w, small_h), interpolation=cv2.INTER_AREA)
        Ls = lab_s[:, :, 0]
        As = lab_s[:, :, 1] - 128.0
        Bs = lab_s[:, :, 2] - 128.0
        env_bool = subj_mask_s < 0.35
        if int(np.sum(env_bool)) < 64:
            env_bool = np.ones_like(Ls, dtype=bool)

        env_l_p05, env_l_p50, env_l_p90 = [float(v) for v in np.percentile(Ls[env_bool], [5, 50, 90])]
        env_sh_mask = env_bool & (Ls <= max(50.0, env_l_p05 + 24.0))
        env_hi_mask = env_bool & (Ls >= min(215.0, env_l_p90 - 20.0))

        tgt_a_sh = float(np.median(As[env_sh_mask])) if np.any(env_sh_mask) else float(np.median(As[env_bool]))
        tgt_b_sh = float(np.median(Bs[env_sh_mask])) if np.any(env_sh_mask) else float(np.median(Bs[env_bool]))
        tgt_a_mid = float(np.median(As[env_bool]))
        tgt_b_mid = float(np.median(Bs[env_bool]))
        tgt_a_hi = float(np.median(As[env_hi_mask])) if np.any(env_hi_mask) else tgt_a_mid * 0.75
        tgt_b_hi = float(np.median(Bs[env_hi_mask])) if np.any(env_hi_mask) else tgt_b_mid * 0.85
        tgt_l_floor = env_l_p05

        # Blend with uploaded reference mood photo statistics when provided
        if reference_scene_bgr is not None and reference_scene_bgr.size > 0:
            ref_s = cv2.resize(reference_scene_bgr, (160, 160), interpolation=cv2.INTER_AREA)
            ref_lab = cv2.cvtColor(ref_s, cv2.COLOR_BGR2LAB).astype(np.float32)
            rL = ref_lab[:, :, 0]
            rA = ref_lab[:, :, 1] - 128.0
            rB = ref_lab[:, :, 2] - 128.0
            r_p05, r_p50, r_p90 = [float(v) for v in np.percentile(rL, [5, 50, 90])]
            r_sh = rL <= max(48.0, r_p05 + 22.0)
            r_hi = rL >= min(215.0, r_p90 - 18.0)
            w_ref = 0.62
            tgt_l_floor = (1.0 - w_ref) * tgt_l_floor + w_ref * r_p05
            tgt_a_mid = (1.0 - w_ref) * tgt_a_mid + w_ref * float(np.median(rA))
            tgt_b_mid = (1.0 - w_ref) * tgt_b_mid + w_ref * float(np.median(rB))
            if np.any(r_sh):
                tgt_a_sh = (1.0 - w_ref) * tgt_a_sh + w_ref * float(np.median(rA[r_sh]))
                tgt_b_sh = (1.0 - w_ref) * tgt_b_sh + w_ref * float(np.median(rB[r_sh]))
            if np.any(r_hi):
                tgt_a_hi = (1.0 - w_ref) * tgt_a_hi + w_ref * float(np.median(rA[r_hi]))
                tgt_b_hi = (1.0 - w_ref) * tgt_b_hi + w_ref * float(np.median(rB[r_hi]))
        elif target_spec is not None:
            tgt_b_mid = 0.55 * tgt_b_mid + 0.45 * float(target_spec.lab_b_cast)
            tgt_a_mid = 0.65 * tgt_a_mid + 0.35 * float(target_spec.lab_a_cast)
            tgt_b_hi = 0.55 * tgt_b_hi + 0.45 * float(target_spec.lab_b_cast * 0.85)
            tgt_l_floor = 0.55 * tgt_l_floor + 0.45 * float(target_spec.shadow_floor_luma)

        tgt_a_mid += float(cfg["tint_bias_a"])  # type: ignore[arg-type]
        tgt_b_mid += float(cfg["temp_bias_b"])  # type: ignore[arg-type]
        tgt_b_hi += float(cfg["temp_bias_b"]) * 0.85  # type: ignore[arg-type]
        tgt_b_sh += float(cfg["temp_bias_b"]) * 0.65  # type: ignore[arg-type]

        # ── Stage 2: Anti-Studio-Cutout Neutral-White & Lifted-Shadow Harmonization on Foreground ──
        # Detect sterile catalog whites (high L, near-zero chroma) on shirts/watches/props inside warm/tinted scenes
        chroma = np.hypot(A, B)
        neutral_white_weight = (
            np.clip((L - 138.0) / 68.0, 0.0, 1.0)
            * np.clip((11.5 - chroma) / 9.5, 0.0, 1.0)
            * (1.0 - 0.70 * skin_damper)
        )
        # Tint sterile catalog whites toward the scene's true highlight white balance & slightly tame catalog peak glare
        A = A + strength * 0.68 * neutral_white_weight * (tgt_a_hi - A)
        B = B + strength * 0.78 * neutral_white_weight * (tgt_b_hi - B)
        if env_l_p50 < 125.0:
            L = L - strength * 7.5 * neutral_white_weight * subj_mask

        # Anchor lifted foreground studio shadows down to the room's true shadow floor
        fg_bool = subj_mask_s >= 0.45
        if np.any(fg_bool):
            fg_p08 = float(np.percentile(Ls[fg_bool], 8))
            shadow_lift_gap = float(np.clip(fg_p08 - (tgt_l_floor + 4.0), 0.0, 22.0))
            if shadow_lift_gap > 1.5:
                dark_pull_w = np.clip((118.0 - L) / 95.0, 0.0, 1.0) * subj_mask * (1.0 - 0.35 * organ_shield)
                L = L - strength * 0.62 * shadow_lift_gap * dark_pull_w

        # ── Stage 3: 2D Spatial Ambient Irradiance Coupling (grounds props/bags sitting in darker room zones!) ──
        illum_field_s = cv2.GaussianBlur(Ls, (0, 0), max(6.0, small_w * 0.16))
        illum_norm_s = (illum_field_s - float(np.mean(illum_field_s))) / max(18.0, float(np.std(illum_field_s)) * 2.2)
        illum_norm = cv2.resize(np.clip(illum_norm_s, -1.0, 1.0), (w, h), interpolation=cv2.INTER_LINEAR)
        # Subtle irradiance coupling (stronger outside face skin so props/bags/clothes match local room brightness)
        irradiance_gain = 1.0 + strength * 0.055 * illum_norm * (1.0 - 0.55 * skin_damper)
        L = np.clip(L * irradiance_gain, 0.0, 255.0)

        # ── Stage 4: Optical Light-Wrap Across High-Contrast Window/Backlight Edges ──
        lw_strength = float(cfg["light_wrap_strength"]) * strength  # type: ignore[arg-type]
        if lw_strength > 0.001:
            bright_env_s = np.clip((Ls - 196.0) / 48.0, 0.0, 1.0) * (1.0 - 0.65 * subj_mask_s)
            if float(np.max(bright_env_s)) > 0.05:
                wrap_glow_s = cv2.GaussianBlur(bright_env_s, (0, 0), max(2.2, small_w * 0.025))
                wrap_glow = cv2.resize(wrap_glow_s, (w, h), interpolation=cv2.INTER_LINEAR)
                receiving_edge = np.clip((185.0 - L) / 120.0, 0.0, 1.0) * (1.0 - organ_shield)
                wrap_delta = lw_strength * 34.0 * wrap_glow * receiving_edge
                L = np.clip(L + wrap_delta, 0.0, 255.0)
                B = B + wrap_delta * 0.22 * np.sign(tgt_b_hi if abs(tgt_b_hi) > 1.0 else 1.0)

        # ── Stage 5: Unified Filmic S-Curve Contrast & 3-Zone Split-Toning Color Science ──
        ln = np.clip(L / 255.0, 0.0, 1.0)
        c_boost = float(cfg["contrast_boost"]) * strength  # type: ignore[arg-type]
        if c_boost > 0.001:
            # Smooth filmic S-curve centered near 0.46 keeping organ micro-gradients crisp
            s_curve = ln - c_boost * 0.22 * np.sin(2.0 * np.pi * ln) * (1.0 - 0.45 * skin_damper)
            L = np.clip(s_curve * 255.0, 0.0, 255.0)

        # Soft filmic highlight shoulder above L=230 (prevents harsh digital clipping & plastic specular glare!)
        hi_excess = np.maximum(0.0, L - 230.0)
        L = np.where(L > 230.0, 230.0 + 16.0 * np.tanh(hi_excess / 18.0), L)

        # 3-Zone Split-Toning weights (Shadows / Midtones / Highlights)
        ln2 = np.clip(L / 255.0, 0.0, 1.0)
        w_sh = np.clip(1.0 - ln2 * 2.2, 0.0, 1.0) ** 1.4
        w_hi = np.clip((ln2 - 0.52) * 2.1, 0.0, 1.0) ** 1.4
        w_mid = np.clip(1.0 - w_sh - w_hi, 0.0, 1.0)

        cur_a_sh = float(np.median(A[L <= 68.0])) if np.any(L <= 68.0) else float(np.median(A))
        cur_b_sh = float(np.median(B[L <= 68.0])) if np.any(L <= 68.0) else float(np.median(B))
        cur_a_mid = float(np.median(A))
        cur_b_mid = float(np.median(B))
        cur_a_hi = float(np.median(A[L >= 185.0])) if np.any(L >= 185.0) else cur_a_mid
        cur_b_hi = float(np.median(B[L >= 185.0])) if np.any(L >= 185.0) else cur_b_mid

        da_sh = float(np.clip((tgt_a_sh - cur_a_sh) * 0.55, -9.0, 9.0))
        db_sh = float(np.clip((tgt_b_sh - cur_b_sh) * 0.65, -14.0, 14.0))
        da_mid = float(np.clip((tgt_a_mid - cur_a_mid) * 0.45, -7.0, 7.0))
        db_mid = float(np.clip((tgt_b_mid - cur_b_mid) * 0.55, -12.0, 12.0))
        da_hi = float(np.clip((tgt_a_hi - cur_a_hi) * 0.55, -8.0, 8.0))
        db_hi = float(np.clip((tgt_b_hi - cur_b_hi) * 0.65, -14.0, 14.0))

        # Protect natural human skin blood-chrominance from extreme shifts (0.48x damper on face skin)
        chroma_skin_guard = 1.0 - 0.52 * skin_damper
        A = A + strength * chroma_skin_guard * (w_sh * da_sh + w_mid * da_mid + w_hi * da_hi)
        B = B + strength * chroma_skin_guard * (w_sh * db_sh + w_mid * db_mid + w_hi * db_hi)

        # ── Stage 6: Optical Peripheral Lens Vignette (`cos⁴` falloff), Highlight Bloom & Unified Sensor Grain ──
        # Natural large-aperture prime lens peripheral falloff centered halfway between frame center and subject
        vx_c = (0.5 * w if face_det is None else 0.55 * float(face_det.center[0]) + 0.45 * (0.5 * w)) / float(w)
        vy_c = (0.5 * h if face_det is None else 0.55 * float(face_det.center[1]) + 0.45 * (0.5 * h)) / float(h)
        yy_s, xx_s = np.meshgrid(
            np.linspace(0.0, 1.0, small_h, dtype=np.float32),
            np.linspace(0.0, 1.0, small_w, dtype=np.float32),
            indexing="ij",
        )
        r2_s = ((xx_s - vx_c) * 1.15) ** 2 + ((yy_s - vy_c) * 1.05) ** 2
        vignette_s = np.clip(1.0 - strength * 0.13 * (r2_s ** 1.35), 0.82, 1.0)
        vignette_full = cv2.resize(vignette_s, (w, h), interpolation=cv2.INTER_LINEAR)
        L = np.clip(L * (1.0 - (1.0 - vignette_full) * (1.0 - 0.80 * skin_damper)), 0.0, 255.0)

        bloom_k = float(cfg["bloom_strength"]) * strength  # type: ignore[arg-type]
        if bloom_k > 0.001:
            hi_src_s = np.clip((Ls - 210.0) / 42.0, 0.0, 1.0)
            if float(np.max(hi_src_s)) > 0.04:
                bloom_s = 0.6 * cv2.GaussianBlur(hi_src_s, (0, 0), max(1.8, small_w * 0.018)) + 0.4 * cv2.GaussianBlur(
                    hi_src_s, (0, 0), max(4.0, small_w * 0.045)
                )
                bloom_full = cv2.resize(bloom_s, (w, h), interpolation=cv2.INTER_LINEAR)
                # Mask bloom completely away from facial skin & organs so Gate 6 Glare & Gate 7 Organ Sharpness stay pristine!
                bloom_safe = bloom_full * (1.0 - np.clip(skin_damper + organ_shield, 0.0, 1.0))
                L = np.clip(L + bloom_k * 36.0 * bloom_safe * (1.0 - L / 255.0), 0.0, 255.0)

        grain_sigma = float(cfg["grain_sigma"]) * min(1.0, strength)  # type: ignore[arg-type]
        if grain_sigma > 0.05:
            rng = np.random.default_rng(seed)
            # Generate fine organic film grain at 1/2 scale + full-res mix so grain has optical sensor character
            g_full = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
            mid_weight = 4.0 * (L / 255.0) * (1.0 - L / 255.0)  # Peaks at 1.0 in midtones, 0 at pure black/white
            organ_grain_guard = 1.0 - 0.55 * organ_shield
            L = np.clip(L + grain_sigma * g_full * (0.35 + 0.65 * mid_weight) * organ_grain_guard, 0.0, 255.0)

        lab_out = np.stack(
            [
                np.clip(L, 0.0, 255.0),
                np.clip(A + 128.0, 0.0, 255.0),
                np.clip(B + 128.0, 0.0, 255.0),
            ],
            axis=-1,
        ).astype(np.uint8)
        out_bgr = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

        meta = {
            "mode": mode,
            "applied": True,
            "strength": round(strength, 2),
            "target_shadow_floor_L": round(tgt_l_floor, 1),
            "target_lab_a_mid": round(tgt_a_mid, 2),
            "target_lab_b_mid": round(tgt_b_mid, 2),
            "applied_shifts_ab": {
                "shadow": [round(da_sh, 2), round(db_sh, 2)],
                "midtone": [round(da_mid, 2), round(db_mid, 2)],
                "highlight": [round(da_hi, 2), round(db_hi, 2)],
            },
        }
        return out_bgr, meta
