"""Universal Camera × Film Stock × Lens DOF & Anti-AI-Yellow/Crunch Optical Engine (`scene_radiometry.py` v3.5).

Eliminates the three classic AI generation artifacts while delivering authentic, dimensional real-photo editorials:
1. **Kills the Infamous AI Yellow/Amber Cast (`특유의 이상한 노란끼 원천 제거`)**:
   - Enforces grey-card calibrated neutral daylight white balance (`5400K–5600K`) in generation prompts (never triggering
     GPT-Image's sepia/amber bias) AND applies quantitative CIE LAB `b*`/`a*` highlight-and-midtone white-balance neutralization
     in `PostGenCohesionMasterer`.
2. **Kills AI Hyper-Texture & Repetitive Procedural Patterns (`AI 과잉 묘사·반복 패턴 재질 차단`)**:
   - Enforces Distance-Appropriate Optical MTF Resolution (anchoring sensor texture to the real 0° camera photograph so
     30cm macro product pebble/suede/pore patterns are NEVER stamped onto 2.5–3.5m editorial shots) + subtle post-gen
     Optical MTF De-Crunch filtering outside facial organs.
3. **Explicit Camera Body × Film Stock × Lens & Aperture DOF Architecture (`카메라 종류 · 필름 종류 · 렌즈 심도 분리`)**:
   - `CAMERA_PROFILES`: Hasselblad X2D 100C, Leica SL3, Sony α7R V, Contax 645, Canon EOS R5
   - `FILM_PROFILES`: Clean Digital Neutral RAW, Kodak Portra 400, Fujifilm Pro 400H, CineStill 50D, Kodak Gold 200, Ilford HP5 B&W
   - `APERTURE_MODES`: f/1.4 (ultra-shallow planar bokeh), f/2.0 (editorial 3-plane depth), f/2.8 (outfit-balanced DOF), f/5.6 (architectural depth)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

CAMERA_PROFILES: Dict[str, Dict[str, str]] = {
    "hasselblad_x2d": {
        "label_ko": "Hasselblad X2D 100C (중형 디지털 · HNCS 중립 리얼 컬러)",
        "sensor_desc": (
            "Hasselblad X2D 100C medium-format digital camera with Hasselblad Natural Color Solution (HNCS)—"
            "effortless 16-bit highlight-to-shadow roll-off, true neutral daylight white balance, calm unforced surface "
            "rendering with zero digital oversharpening or HDR crunch"
        ),
    },
    "leica_sl3": {
        "label_ko": "Leica SL3 (라이카 풀프레임 · 맑고 깊은 입체 계조)",
        "sensor_desc": (
            "Leica SL3 full-frame camera paired with Leica Summilux/APO glass—clean neutral-cool shadow transparency, "
            "subtle three-dimensional subject-to-background separation, natural unretouched skin, and understated European documentary realism"
        ),
    },
    "sony_a7rv": {
        "label_ko": "Sony α7R V (모던 커머셜 · 정밀 화이트밸런스)",
        "sensor_desc": (
            "Sony Alpha 7R V full-frame digital camera calibrated to a grey card at 5500K—accurate neutral garment color fidelity, "
            "clean crisp whites, smooth modern optical bokeh, and zero color cast"
        ),
    },
    "contax_645": {
        "label_ko": "Contax 645 (중형 필름 바디 · 부드러운 공간 입체감)",
        "sensor_desc": (
            "Contax 645 medium-format camera with Carl Zeiss Planar 80mm optics—gentle organic spatial depth, "
            "soft highlight shoulder, and smooth natural transition from the focal plane into calm out-of-focus depth"
        ),
    },
    "canon_r5": {
        "label_ko": "Canon EOS R5 (맑고 자연스러운 인물 혈색 · 소프트 에디토리얼)",
        "sensor_desc": (
            "Canon EOS R5 full-frame portrait camera with RF L prime optics—clean, healthy translucent skin undertones "
            "(never yellow or orange), gentle tonal contrast, and natural daylight realism"
        ),
    },
}

FILM_PROFILES: Dict[str, Dict[str, object]] = {
    "digital_neutral_raw": {
        "label_ko": "무보정 클린 디지털 RAW (노란끼 제로 · 순수 중립 화이트밸런스)",
        "prompt_color_desc": (
            "unretouched Capture One neutral RAW color profile calibrated to 5500K daylight—clean pure neutral whites and creams, "
            "translucent true-to-life skin tone, neutral-clean shadows, zero yellow/amber/sepia filter, and zero artificial film grain"
        ),
        "strength": 1.0,
        "de_yellow": 0.88,
        "de_crunch": 0.22,
        "target_hi_b": 1.0,
        "target_mid_b": 3.5,
        "target_sh_b": 0.0,
        "target_mid_a": 1.2,
        "target_sh_a": 0.0,
        "contrast_boost": 0.03,
        "bloom_strength": 0.02,
        "light_wrap_strength": 0.035,
        "grain_sigma": 0.0,
        "monochrome": False,
    },
    "kodak_portra_400": {
        "label_ko": "Kodak Portra 400 (맑은 인물 스킨톤 & 크리미 뉴트럴 하이라이트)",
        "prompt_color_desc": (
            "authentic daylight-balanced Kodak Portra 400 color negative stock scanned cleanly without yellow cast—soft creamy "
            "neutral highlights, clean natural peach-rose human skin tones, calm muted wardrobe colors, gentle shadow latitude, "
            "and ultra-fine organic 35mm emulsion structure"
        ),
        "strength": 1.0,
        "de_yellow": 0.80,
        "de_crunch": 0.24,
        "target_hi_b": 3.0,
        "target_mid_b": 5.2,
        "target_sh_b": -1.0,
        "target_mid_a": 1.8,
        "target_sh_a": 0.2,
        "contrast_boost": 0.02,
        "bloom_strength": 0.03,
        "light_wrap_strength": 0.04,
        "grain_sigma": 0.35,
        "monochrome": False,
    },
    "fujifilm_pro_400h": {
        "label_ko": "Fujifilm Pro 400H (청량하고 맑은 데이라이트 · 쿨-에어리 암부)",
        "prompt_color_desc": (
            "authentic Fujifilm Pro 400H daylight editorial stock—airy luminous neutral-cool whites, crystal-clear skin tones "
            "free of any yellow/orange cast, subtle cool slate-mint airiness in shadows, and calm refined pastel-neutral tonality"
        ),
        "strength": 1.0,
        "de_yellow": 0.95,
        "de_crunch": 0.24,
        "target_hi_b": -1.0,
        "target_mid_b": 1.5,
        "target_sh_b": -3.2,
        "target_mid_a": 0.6,
        "target_sh_a": -1.4,
        "contrast_boost": 0.02,
        "bloom_strength": 0.025,
        "light_wrap_strength": 0.035,
        "grain_sigma": 0.30,
        "monochrome": False,
    },
    "cinestill_50d": {
        "label_ko": "CineStill 50D (초미립자 시네마 데이라이트 · 투명한 뉴트럴)",
        "prompt_color_desc": (
            "CineStill 50D fine-grain daylight motion-picture stock—crisp neutral 5500K daylight balance, cool-clean architectural "
            "shadows, natural subtle highlight halation around bright windows only, and smooth grainless tonal transitions"
        ),
        "strength": 1.0,
        "de_yellow": 0.90,
        "de_crunch": 0.20,
        "target_hi_b": 0.5,
        "target_mid_b": 2.5,
        "target_sh_b": -2.2,
        "target_mid_a": 0.8,
        "target_sh_a": -0.6,
        "contrast_boost": 0.05,
        "bloom_strength": 0.045,
        "light_wrap_strength": 0.045,
        "grain_sigma": 0.22,
        "monochrome": False,
    },
    "kodak_gold_200": {
        "label_ko": "Kodak Gold 200 (담백한 오후 햇살 · 과장 없는 내추럴 웜)",
        "prompt_color_desc": (
            "subtle daylight Kodak Gold 200 look—gentle natural afternoon sunlight warmth while keeping white fabrics and skin "
            "clean and believable without muddy sepia or heavy orange saturation"
        ),
        "strength": 0.95,
        "de_yellow": 0.60,
        "de_crunch": 0.22,
        "target_hi_b": 5.5,
        "target_mid_b": 7.8,
        "target_sh_b": 1.5,
        "target_mid_a": 2.0,
        "target_sh_a": 0.5,
        "contrast_boost": 0.04,
        "bloom_strength": 0.035,
        "light_wrap_strength": 0.04,
        "grain_sigma": 0.40,
        "monochrome": False,
    },
    "ilford_hp5_bw": {
        "label_ko": "Ilford HP5 Plus 400 (클래식 흑백 리얼 에디토리얼)",
        "prompt_color_desc": (
            "authentic Ilford HP5 Plus 400 silver-gelatin black-and-white photograph—rich midtone tonal separation, "
            "deep natural blacks, luminous unclipped highlights, and classic monochrome editorial presence"
        ),
        "strength": 1.0,
        "de_yellow": 1.0,
        "de_crunch": 0.18,
        "target_hi_b": 0.0,
        "target_mid_b": 0.0,
        "target_sh_b": 0.0,
        "target_mid_a": 0.0,
        "target_sh_a": 0.0,
        "contrast_boost": 0.07,
        "bloom_strength": 0.02,
        "light_wrap_strength": 0.03,
        "grain_sigma": 0.48,
        "monochrome": True,
    },
    "off": {
        "label_ko": "후처리 끄기 (생성 원본 그대로)",
        "prompt_color_desc": "neutral unretouched daylight color balance at 5500K with zero yellow cast",
        "strength": 0.0,
        "de_yellow": 0.0,
        "de_crunch": 0.0,
        "target_hi_b": 0.0,
        "target_mid_b": 0.0,
        "target_sh_b": 0.0,
        "target_mid_a": 0.0,
        "target_sh_a": 0.0,
        "contrast_boost": 0.0,
        "bloom_strength": 0.0,
        "light_wrap_strength": 0.0,
        "grain_sigma": 0.0,
        "monochrome": False,
    },
}

# Backward-compatible aliases for `COHESION_MASTERING_MODES` callers & saved state
COHESION_MASTERING_MODES: Dict[str, Dict[str, object]] = {
    "auto": FILM_PROFILES["digital_neutral_raw"],
    **FILM_PROFILES,
    "warm_cinema_amber": FILM_PROFILES["kodak_gold_200"],
    "deep_chiaroscuro": dict(FILM_PROFILES["cinestill_50d"], contrast_boost=0.09, label_ko="시네마틱 딥 명암 (중립 화이트·암부 밀착)"),
    "soft_editorial_natural": FILM_PROFILES["kodak_portra_400"],
}

LIGHTING_MODES: Dict[str, Dict[str, object]] = {
    "auto": {
        "label_ko": "자동 광학 추출 (장소·레퍼런스 자연광 계측)",
    },
    "natural_window_backlight": {
        "label_ko": "자연 창가 측후방광 (은은한 역광 + 입체 음영 · 노란끼 배제)",
        "kelvin_est": 5400,
        "contrast_ratio_stops": 2.6,
        "shadow_floor_luma": 24.0,
        "highlight_bloom": 0.08,
        "lighting_regime": "backlit_chiaroscuro",
        "key_light_vector": (
            "clean daylight streaming naturally through side-rear window blinds, creating a gentle unforced highlight along hair "
            "and shoulder edges while placing the camera-facing cheek and torso in calm, dimensional natural shade"
        ),
        "color_cast_summary": "clean 5400K neutral daylight white balance where white fabrics stay crisp off-white and shadows stay clean neutral-taupe (ZERO yellow/sepia cast)",
        "recommended_aperture": "f/2.0",
    },
    "warm_golden_sunbeam": {
        "label_ko": "오후 자연 채광·그림자 (맑은 데이라이트 화이트밸런스)",
        "kelvin_est": 5200,
        "contrast_ratio_stops": 2.4,
        "shadow_floor_luma": 26.0,
        "highlight_bloom": 0.06,
        "lighting_regime": "directional_sunbeam",
        "key_light_vector": (
            "angled afternoon daylight entering from the side window, casting soft believable architectural shadow geometry "
            "across the room and creating natural 2-to-2.5-stop light-to-shadow depth across the subject"
        ),
        "color_cast_summary": "calibrated 5200K daylight white balance—natural sunlight presence with clean neutral whites and zero muddy yellow/orange tint",
        "recommended_aperture": "f/2.0",
    },
    "soft_diffused_daylight": {
        "label_ko": "부드러운 자연 확산 창가광 (가장 담백한 리얼 에디토리얼)",
        "kelvin_est": 5500,
        "contrast_ratio_stops": 2.0,
        "shadow_floor_luma": 30.0,
        "highlight_bloom": 0.03,
        "lighting_regime": "directional_side_key",
        "key_light_vector": (
            "large single-side diffused window daylight creating a smooth, unforced 2-stop luminance transition from the lit cheek "
            "across the nose bridge to the softly shaded cheek"
        ),
        "color_cast_summary": "neutral 5500K grey-card calibrated daylight with clean whites, true garment colors, and translucent natural skin",
        "recommended_aperture": "f/2.0",
    },
    "moody_chiaroscuro_interior": {
        "label_ko": "로우키 실내 자연광 (차분한 깊은 암부 · 중립 색온도)",
        "kelvin_est": 5000,
        "contrast_ratio_stops": 3.0,
        "shadow_floor_luma": 18.0,
        "highlight_bloom": 0.04,
        "lighting_regime": "moody_low_key",
        "key_light_vector": (
            "quiet directional side-window light inside a dimly lit room, illuminating one side of the subject cleanly while "
            "the opposite side and deep room interior settle into calm dark neutral shadow"
        ),
        "color_cast_summary": "clean 5000K neutral-toned low-key palette with deep neutral charcoal shadows and zero orange/sepia muddiness",
        "recommended_aperture": "f/1.8",
    },
    "cool_overcast_architectural": {
        "label_ko": "청량한 흐린 날 야외·건축 주광 (쿨-뉴트럴 데이라이트)",
        "kelvin_est": 5800,
        "contrast_ratio_stops": 1.8,
        "shadow_floor_luma": 32.0,
        "highlight_bloom": 0.02,
        "lighting_regime": "diffused_architectural",
        "key_light_vector": "open sky / clerestory daylight from upper-side with soft grounded contact shadows toward the floor",
        "color_cast_summary": "crisp 5800K architectural daylight with airy cool-neutral shadows and pure whites",
        "recommended_aperture": "f/2.8",
    },
}

APERTURE_MODES: Dict[str, str] = {
    "auto": "",
    "f1_4": "f/1.4",
    "f1_8": "f/1.8",
    "f2_0": "f/2.0",
    "f2_8": "f/2.8",
    "f4_0": "f/4.0",
    "f5_6": "f/5.6",
}

APERTURE_DOF_DESCRIPTIONS: Dict[str, str] = {
    "f/1.4": (
        "wide-open f/1.4 ultra-shallow planar depth of field—focus plane razor-thin strictly on the nearer eye and glasses frame; "
        "even the far shoulder softens gently, foreground edges dissolve into soft translucent defocus, and the background melts "
        "into smooth, calm optical blur (strictly NO fake CGI neon bokeh circles)"
    ),
    "f/1.8": (
        "fast f/1.8 shallow editorial depth of field—crisp focus on the eyes, face, and active hand while foreground elements and "
        "receding background architecture fall smoothly and progressively out of focus according to real optical distance"
    ),
    "f/2.0": (
        "natural f/2.0 prime-lens editorial depth of field—clean optical focus across the subject's face and upper torso with "
        "clear 3-plane spatial separation (softly defocused foreground edge, crisp subject plane, calm out-of-focus background)"
    ),
    "f/2.4": (
        "balanced f/2.4 portrait depth of field—subject and upper outfit cleanly resolved while background space softens naturally with distance"
    ),
    "f/2.8": (
        "f/2.8 lookbook-editorial depth of field—the entire outfit silhouette, watch, and bag on the subject plane remain clearly "
        "readable while the background room or street recedes into gentle, unbusy optical defocus"
    ),
    "f/4.0": (
        "moderate f/4.0 environmental portrait depth of field—subject crisp in the foreground-middle plane with recognizable yet "
        "gently softened architectural context behind"
    ),
    "f/5.6": (
        "deep f/5.6 architectural editorial depth of field—both the subject and surrounding architectural geometry remain clearly "
        "readable with natural distance falloff and zero artificial oversharpening"
    ),
}


@dataclass(frozen=True)
class SingleLensRadiometricSpec:
    """Autonomous DoP + Camera × Film Stock × Lens DOF + Illuminance (Lux) + Light Flow specification."""

    kelvin_est: int = 5400
    illuminance_lux: int = 650
    contrast_ratio_stops: float = 2.4
    shadow_floor_luma: float = 24.0
    median_luma: float = 108.0
    highlight_p95_luma: float = 220.0
    highlight_bloom: float = 0.05
    lab_a_cast: float = 1.5
    lab_b_cast: float = 3.5
    lighting_regime: str = "directional_side_key"
    key_light_vector: str = (
        "single directional natural daylight source creating a believable 2.4-stop highlight-to-shadow transition across the subject and space"
    )
    light_flow_desc: str = (
        "light enters from a single off-camera architectural opening, skims gently across the lit cheekbone, collar edge, "
        "and natural fabric folds to reveal quiet real-world texture, bounces softly from nearby surfaces into the lower chin, "
        "and falls off smoothly into calm, deep room shadow"
    )
    illuminance_desc: str = "natural indoor-window editorial illuminance (~650 lux key, ~120 lux ambient shadow)"
    color_cast_summary: str = (
        "grey-card calibrated 5500K neutral daylight white balance—clean whites, true garment colors, and natural skin free of any yellow cast"
    )
    recommended_aperture: str = "f/1.8"
    recommended_focal_mm: int = 85
    camera_id: str = "hasselblad_x2d"
    film_id: str = "kodak_portra_400"
    dop_rationale: str = "auto_curated_editorial_dop"
    source: str = "measured_scene_radiometry"

    def to_single_lens_header(self, focal_length_mm: int = 85) -> str:
        """Emit an authentic Master DoP Camera Body + Lens Focal/Aperture + Film Stock + Illuminance capture header."""
        cam = CAMERA_PROFILES.get(self.camera_id, CAMERA_PROFILES["hasselblad_x2d"])
        film = FILM_PROFILES.get(self.film_id, FILM_PROFILES["digital_neutral_raw"])
        ap = self.recommended_aperture or "f/1.8"
        return (
            "Editorial portrait. Masterpiece of natural light, quiet real-world texture, and optical depth—unstaged real-world "
            f"photograph shot on a {cam['sensor_desc']}, fitted with a {focal_length_mm}mm {ap} prime lens under "
            f"{self.illuminance_desc}, developed with {film['prompt_color_desc']}."
        )

    def to_relighting_and_cohesion_clause(
        self,
        has_catalog_items: bool = True,
        has_placed_bag: bool = False,
    ) -> str:
        """Emit the Master DoP Law: Light Flow, Illuminance, True White Balance, Grazing Texture Restraint & Lens DOF."""
        ap = self.recommended_aperture or "f/1.8"
        dof_desc = APERTURE_DOF_DESCRIPTIONS.get(ap, APERTURE_DOF_DESCRIPTIONS["f/1.8"])
        film = FILM_PROFILES.get(self.film_id, FILM_PROFILES["digital_neutral_raw"])

        shadow_side_rule = (
            f"Sculpt authentic {min(3.2, max(1.9, self.contrast_ratio_stops)):.1f}-stop three-dimensional chiaroscuro under "
            f"{self.illuminance_desc}: {self.light_flow_desc}. NEVER illuminate both cheeks evenly with flat studio flash, "
            "and NEVER crush the eyes into muddy black or add fake HDR halos"
        )

        catalog_exposure_note = (
            " Discard any attached product photos' bright #FFFFFF studio table exposure and relight all garments/accessories "
            "strictly under the room's ambient luminance."
            if has_catalog_items
            else ""
        )
        item_relight_rule = (
            " 2) GRAZING-LIGHT TEXTURE & DISTANCE-TRUE MTF (빛이 스치는 곳에만 살아나는 진짜 질감 · 반복 패턴 금지): "
            "Photography is the art of light skimming real surfaces—let fabric drape, wool/cotton/suede softness, and leather character "
            "reveal themselves ONLY where directional light grazes a fold or turn of form, while broad lit and shaded areas stay "
            f"calm, smooth-yet-organic, and proportionate to 2–3.5m camera distance!{catalog_exposure_note} "
            "NEVER stamp uniform repetitive pebble-leather bumps or synthetic sandpaper grit."
        )

        placed_prop_rule = ""
        if has_placed_bag:
            placed_prop_rule = (
                " Ground any placed bag naturally in the room's 3D perspective and local floor shadow with real contact occlusion "
                "and optical distance defocus (never a bright pasted catalog sticker)."
            )

        return (
            "CRITICAL DIRECTOR-OF-PHOTOGRAPHY LIGHT, ILLUMINANCE, TEXTURE & DEPTH LAWS: "
            "1) LIVING LIGHT FLOW & ILLUMINANCE (빛의 흐름 · 밝기 · 조도 · 명암): Images 1–3 (face anchors) were shot under flat studio flash—"
            "take ONLY 3D craniofacial bone structure, hair, eyewear silhouette, and real camera skin realism; 100% discard their studio flash "
            f"and relight under {self.key_light_vector}. {shadow_side_rule}. "
            f"{item_relight_rule}{placed_prop_rule} "
            "3) TRUE WHITE BALANCE & COLOR HARMONY (자연스러운 화이트밸런스와 색감 · 이상한 노란끼 금지): "
            f"Enforce {film['prompt_color_desc']} and {self.color_cast_summary}. Keep white/ivory garments, watch dials, and skin "
            "clean, breathable, and true-to-life—strictly ZERO jaundiced yellow/sepia/orange AI filter! "
            f"4) THREE-PLANE SPATIAL DEPTH & LENS DOF ({ap} 심도의 예술): Apply {dof_desc}. Frame with genuine architectural depth "
            "(subtle softly defocused foreground cue, crisp focal plane on the subject's eyes, and calm spatial recession behind) "
            "in an uncluttered, authentic real-world space free of cliché staged props. "
            "5) UNRETOUCHED REAL-CAMERA SKIN & HAIR: Match the organic real-camera pixel smoothness of Image 2—natural skin tone and "
            "subtle real texture where light touches, without repeated artificial pores, wire-sharp hair strands, or oversharpening."
        )


class AutonomousDoPDirector:
    """Translates ANY rough/slang/contradictory user intent or reference photo into optimal Camera + Film + Lens + Lux + Light Flow."""

    @classmethod
    def curate_rig(
        cls,
        raw_text: str = "",
        is_backlit: bool = False,
        median_luma: float = 110.0,
        b_shift: float = 0.0,
        contrast_stops: float = 2.2,
        has_reference_photo: bool = False,
    ) -> Dict[str, object]:
        low = raw_text.lower()

        bw_kw = ("흑백", "모노크롬", "black and white", "monochrome", "b&w", "느와르")
        rain_night_kw = ("비", "비오는", "밤", "저녁", "새벽", "어둑", "어두", "가로등", "위스키", "바", "쓸쓸", "고독", "rain", "night", "dusk", "twilight", "evening", "moody", "dark", "low-key")
        backlit_afternoon_kw = ("오후", "창가", "역광", "루버", "블라인드", "노을", "햇살", "따스", "커피", "카페", "서재", "작업", "책상", "afternoon", "window", "louver", "sunlit", "cafe", "desk")
        airy_fresh_kw = ("아침", "청량", "맑은", "미니멀", "갤러리", "중정", "산책", "봄", "여름", "담백", "깨끗", "흐린", "morning", "airy", "minimal", "gallery", "courtyard", "clean", "fresh", "overcast")
        contradict_dark_bright = any(k in low for k in ("어두", "어둑", "컴컴", "밤")) and any(
            k in low for k in ("잘 보이", "환하", "선명", "살아", "죽지 않")
        )

        if any(k in low for k in bw_kw):
            return {
                "camera_id": "leica_sl3",
                "film_id": "ilford_hp5_bw",
                "aperture": "f/1.8",
                "focal_length_mm": 50,
                "kelvin_est": 5400,
                "contrast_ratio_stops": 2.8,
                "illuminance_lux": 320,
                "illuminance_desc": "sculpted monochrome directional window illuminance (~320 lux key, ~45 lux deep shadow)",
                "light_flow_desc": (
                    "crisp side-window light carves silver-toned grazing highlights along cheekbone, glasses rim, and lapel edge "
                    "before dropping steeply into velvety monochrome shadow"
                ),
                "color_cast_summary": "true silver-gelatin monochrome tonality with luminous unclipped highlights and rich midtone gradation",
                "rationale": "monochrome_classic_leica_hp5",
            }

        if contradict_dark_bright or any(k in low for k in rain_night_kw) or (
            has_reference_photo and median_luma < 82.0 and not is_backlit and b_shift < 7.0
        ):
            return {
                "camera_id": "leica_sl3",
                "film_id": "cinestill_50d",
                "aperture": "f/1.4",
                "focal_length_mm": 50,
                "kelvin_est": 5000,
                "contrast_ratio_stops": 2.9,
                "illuminance_lux": 150,
                "illuminance_desc": "intimate low-key twilight/evening window illuminance (~150 lux focused key on face, ~25 lux enveloping room shadow)",
                "light_flow_desc": (
                    "a narrow beam of cool-clean twilight window light enters from the side, catching the nearer eye, nasal bridge, "
                    "and upper fabric folds with crystal clarity so the subject's expression and garment texture stay vividly readable "
                    "while the rest of the room recedes into deep, quiet, atmospheric shadow"
                ),
                "color_cast_summary": "clean 5000K neutral-cool twilight white balance—crisp highlights on skin and fabric with deep charcoal-slate shadows and ZERO muddy orange/sepia cast",
                "rationale": "low_key_rembrandt_leica_f14_cinestill50d",
            }

        if is_backlit or any(k in low for k in backlit_afternoon_kw) or (has_reference_photo and b_shift >= 7.0):
            return {
                "camera_id": "contax_645",
                "film_id": "kodak_portra_400",
                "aperture": "f/1.8",
                "focal_length_mm": 50 if any(w in low for w in ("책상", "작업", "카페", "테이블", "desk", "cafe", "table")) else 85,
                "kelvin_est": 5300,
                "contrast_ratio_stops": 2.6,
                "illuminance_lux": 580,
                "illuminance_desc": "directional slant window daylight (~2800 lux sunlit rim/louvers vs. ~580 lux soft interior shadow fill)",
                "light_flow_desc": (
                    "slanting natural window light flows from behind-side, kissing hair contours, shoulder seams, and tabletop edges "
                    "with gentle luminous wrap while bouncing soft secondary fill light upward onto the shaded cheek and collar"
                ),
                "color_cast_summary": "natural 5300K daylight white balance—clean off-white fabrics and translucent skin with gentle architectural sunlight warmth ONLY where sunbeams strike",
                "rationale": "slant_window_contax645_portra400_f18",
            }

        if any(k in low for k in airy_fresh_kw) or (has_reference_photo and median_luma >= 128.0 and b_shift <= 3.0):
            return {
                "camera_id": "hasselblad_x2d",
                "film_id": "fujifilm_pro_400h",
                "aperture": "f/2.0",
                "focal_length_mm": 85,
                "kelvin_est": 5650,
                "contrast_ratio_stops": 2.1,
                "illuminance_lux": 1400,
                "illuminance_desc": "open airy architectural daylight (~1400 lux soft sky/window key, ~420 lux clean open shade)",
                "light_flow_desc": (
                    "broad, breathable open daylight wraps smoothly around facial planes and draped fabric from the side courtyard/window, "
                    "creating effortless three-dimensional volume with airy cool-neutral shadow transparency"
                ),
                "color_cast_summary": "grey-card calibrated 5650K airy daylight white balance—pure neutral whites, cool-clean architectural shadows, and zero yellow tint",
                "rationale": "airy_architectural_hasselblad_fuji400h_f20",
            }

        # Universal balanced default when prompt is super brief, slangy, or abstract ("알아서 멋지게")
        return {
            "camera_id": "hasselblad_x2d",
            "film_id": "kodak_portra_400",
            "aperture": "f/1.8",
            "focal_length_mm": 85,
            "kelvin_est": 5450,
            "contrast_ratio_stops": 2.4,
            "illuminance_lux": 750,
            "illuminance_desc": "natural editorial side-window daylight (~750 lux key light, ~160 lux dimensional interior shade)",
            "light_flow_desc": (
                "natural single-window daylight flows obliquely across the frame, skimming the cheekbone, eyewear frame, "
                "and outer garment folds before tapering smoothly into quiet architectural shadow behind the subject"
            ),
            "color_cast_summary": "calibrated 5450K neutral daylight white balance—clean whites, authentic garment hues, and breathable human skin tone",
            "rationale": "universal_editorial_hasselblad_portra400_f18",
        }


class SceneRadiometryAnalyzer:
    """Deterministically measures scene lighting, contrast, DOF, and white-balance targets without AI yellow bias."""

    @classmethod
    def analyze(
        cls,
        scene_bgr: Optional[np.ndarray] = None,
        mood_prompt: str = "",
        lighting_note: str = "",
        lighting_mode: str = "auto",
        aperture_mode: str = "auto",
        camera_id: str = "auto",
        film_id: str = "auto",
    ) -> SingleLensRadiometricSpec:
        forced_ap = APERTURE_MODES.get(aperture_mode, "")
        combined_text = f"{mood_prompt} {lighting_note}".strip()

        if lighting_mode in LIGHTING_MODES and lighting_mode != "auto":
            preset = LIGHTING_MODES[lighting_mode]
            dop = AutonomousDoPDirector.curate_rig(
                raw_text=f"{combined_text} {lighting_mode}".strip(),
                is_backlit=(lighting_mode == "natural_window_backlight"),
                median_luma=72.0 if lighting_mode == "moody_chiaroscuro_interior" else 112.0,
                contrast_stops=float(preset["contrast_ratio_stops"]),  # type: ignore[arg-type]
                has_reference_photo=(scene_bgr is not None),
            )
            eff_cam = camera_id if camera_id in CAMERA_PROFILES else str(dop["camera_id"])
            eff_film = film_id if (film_id in FILM_PROFILES and film_id != "off") else str(dop["film_id"])
            if film_id == "off":
                eff_film = "off"
            return SingleLensRadiometricSpec(
                kelvin_est=int(preset["kelvin_est"]),  # type: ignore[arg-type]
                illuminance_lux=int(dop["illuminance_lux"]),  # type: ignore[arg-type]
                contrast_ratio_stops=float(preset["contrast_ratio_stops"]),  # type: ignore[arg-type]
                shadow_floor_luma=float(preset["shadow_floor_luma"]),  # type: ignore[arg-type]
                highlight_bloom=float(preset["highlight_bloom"]),  # type: ignore[arg-type]
                lighting_regime=str(preset["lighting_regime"]),
                key_light_vector=str(preset["key_light_vector"]),
                light_flow_desc=str(dop["light_flow_desc"]),
                illuminance_desc=str(dop["illuminance_desc"]),
                color_cast_summary=str(preset["color_cast_summary"]),
                recommended_aperture=forced_ap or str(preset["recommended_aperture"]),
                recommended_focal_mm=int(dop["focal_length_mm"]),  # type: ignore[arg-type]
                camera_id=eff_cam,
                film_id=eff_film,
                dop_rationale=str(dop["rationale"]),
                source=f"preset_{lighting_mode}",
            )

        if scene_bgr is not None and scene_bgr.size > 0:
            return cls._analyze_bgr(
                scene_bgr,
                raw_text=combined_text,
                forced_aperture=forced_ap,
                camera_id=camera_id,
                film_id=film_id,
            )

        return cls._infer_from_text(
            combined_text,
            forced_aperture=forced_ap,
            camera_id=camera_id,
            film_id=film_id,
        )

    @classmethod
    def _analyze_bgr(
        cls,
        bgr: np.ndarray,
        raw_text: str = "",
        forced_aperture: str = "",
        camera_id: str = "auto",
        film_id: str = "auto",
    ) -> SingleLensRadiometricSpec:
        h, w = bgr.shape[:2]
        scale = min(1.0, 640.0 / float(max(h, w)))
        work = cv2.resize(bgr, (max(32, int(w * scale)), max(32, int(h * scale))), interpolation=cv2.INTER_AREA)
        wh, ww = work.shape[:2]

        lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_ch = lab[:, :, 0]
        a_shift = float(np.median(lab[:, :, 1]) - 128.0)
        b_shift = float(np.median(lab[:, :, 2]) - 128.0)

        p05, p15, p50, p85, p95 = [float(v) for v in np.percentile(l_ch, [5, 15, 50, 85, 95])]
        bright_frac = float(np.mean(l_ch >= 215.0))
        dark_frac = float(np.mean(l_ch <= 55.0))
        contrast_stops = float(np.clip(np.log2(max(1.2, (p95 + 12.0) / max(14.0, p15 + 8.0))), 1.4, 3.6))

        # Estimate physical source Kelvin (for radiometric diagnostics), while keeping prompt WB clean!
        kelvin_measured = int(round(np.clip(5500.0 - b_shift * 110.0 - a_shift * 30.0, 3200.0, 6500.0)))

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

        is_backlit = (bright_frac >= 0.06 and p50 < 118.0 and upper_bg_energy > center_luma + 16.0) or (
            p95 >= 220.0 and p50 <= 98.0 and contrast_stops >= 2.5
        )
        if is_backlit:
            regime = "backlit_chiaroscuro"
            key_vec = (
                f"natural window backlight and {side_desc} daylight streaming through the space—creating subtle unforced rim "
                "illumination on hair and shoulder edges while keeping the camera-facing cheek and chest in calm, natural shade"
            )
        elif p50 < 78.0 and dark_frac > 0.35:
            regime = "moody_low_key"
            key_vec = f"quiet low-key directional {side_desc} window light with calm dimensional shadows across the opposite side"
        elif contrast_stops >= 2.45:
            regime = "directional_sunbeam"
            key_vec = f"angled {side_desc} natural daylight casting soft architectural shadow gradients across room and subject"
        else:
            regime = "directional_side_key"
            key_vec = f"soft directional {side_desc} daylight creating a smooth natural luminance falloff across face and outfit"

        dop = AutonomousDoPDirector.curate_rig(
            raw_text=raw_text,
            is_backlit=is_backlit,
            median_luma=p50,
            b_shift=b_shift,
            contrast_stops=contrast_stops,
            has_reference_photo=True,
        )
        eff_cam = camera_id if camera_id in CAMERA_PROFILES else str(dop["camera_id"])
        eff_film = film_id if (film_id in FILM_PROFILES and film_id != "off") else str(dop["film_id"])
        if film_id == "off":
            eff_film = "off"

        if b_shift >= 10.0:
            color_summary = (
                "natural daylight white balance (5300K–5500K) preserving subtle warm wood/sunlight atmosphere ONLY on environmental "
                "surfaces while keeping white shirts, highlights, and human skin clean and free of muddy yellow/orange cast"
            )
        elif b_shift <= -4.0:
            color_summary = "clean airy 5700K daylight balance with cool-neutral shadow transparency and crisp whites"
        else:
            color_summary = str(dop["color_cast_summary"])

        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        center_sharp = float(np.mean(lap[gh : 2 * gh, gw : 2 * gw])) + 1e-4
        border_mask = np.ones_like(lap, dtype=bool)
        border_mask[gh // 2 : wh - gh // 2, gw // 2 : ww - gw // 2] = False
        dof_ratio = float(np.mean(lap[border_mask])) / center_sharp

        if forced_aperture:
            ap = forced_aperture
        elif dof_ratio < 0.58 or is_backlit:
            ap = str(dop["aperture"])
        elif dof_ratio < 0.82:
            ap = "f/2.0"
        else:
            ap = "f/2.8"

        return SingleLensRadiometricSpec(
            kelvin_est=kelvin_measured,
            illuminance_lux=int(dop["illuminance_lux"]),  # type: ignore[arg-type]
            contrast_ratio_stops=round(contrast_stops, 2),
            shadow_floor_luma=round(p05, 1),
            median_luma=round(p50, 1),
            highlight_p95_luma=round(p95, 1),
            highlight_bloom=round(bright_frac, 3),
            lab_a_cast=round(float(np.clip(a_shift * 0.45, -4.0, 4.0)), 1),
            lab_b_cast=round(float(np.clip(b_shift * 0.45, -5.0, 6.5)), 1),
            lighting_regime=regime,
            key_light_vector=key_vec,
            light_flow_desc=str(dop["light_flow_desc"]),
            illuminance_desc=str(dop["illuminance_desc"]),
            color_cast_summary=color_summary,
            recommended_aperture=ap,
            recommended_focal_mm=int(dop["focal_length_mm"]),  # type: ignore[arg-type]
            camera_id=eff_cam,
            film_id=eff_film,
            dop_rationale=str(dop["rationale"]),
            source="measured_scene_radiometry",
        )

    @classmethod
    def _infer_from_text(
        cls,
        text: str,
        forced_aperture: str = "",
        camera_id: str = "auto",
        film_id: str = "auto",
    ) -> SingleLensRadiometricSpec:
        low = text.lower()
        dop = AutonomousDoPDirector.curate_rig(raw_text=text, has_reference_photo=False)
        eff_cam = camera_id if camera_id in CAMERA_PROFILES else str(dop["camera_id"])
        eff_film = film_id if (film_id in FILM_PROFILES and film_id != "off") else str(dop["film_id"])
        if film_id == "off":
            eff_film = "off"

        backlit_kw = ("backlit", "backlight", "louver", "blind", "rim", "역광", "블라인드", "루버")
        lowkey_kw = ("비", "비오는", "밤", "저녁", "새벽", "어둑", "어두", "컴컴", "가로등", "쓸쓸", "고독", "rain", "night", "dusk", "twilight", "moody", "dark", "low-key")
        sun_kw = ("sunlit", "afternoon", "window", "lounge", "wood", "오후", "창가", "자연광", "햇살", "사선")
        cool_kw = ("cool", "concrete", "overcast", "slate", "gallery", "courtyard", "차분", "콘크리트", "갤러리", "중정", "흐린", "청량")

        if any(k in low for k in lowkey_kw):
            base = LIGHTING_MODES["moody_chiaroscuro_interior"]
        elif any(k in low for k in backlit_kw):
            base = LIGHTING_MODES["natural_window_backlight"]
        elif any(k in low for k in cool_kw):
            base = LIGHTING_MODES["cool_overcast_architectural"]
        elif any(k in low for k in sun_kw):
            base = LIGHTING_MODES["warm_golden_sunbeam"]
        else:
            base = LIGHTING_MODES["soft_diffused_daylight"]

        return SingleLensRadiometricSpec(
            kelvin_est=int(dop.get("kelvin_est", base["kelvin_est"])),  # type: ignore[arg-type]
            illuminance_lux=int(dop["illuminance_lux"]),  # type: ignore[arg-type]
            contrast_ratio_stops=float(dop.get("contrast_ratio_stops", base["contrast_ratio_stops"])),  # type: ignore[arg-type]
            shadow_floor_luma=float(base["shadow_floor_luma"]),  # type: ignore[arg-type]
            highlight_bloom=float(base["highlight_bloom"]),  # type: ignore[arg-type]
            lighting_regime=str(base["lighting_regime"]),
            key_light_vector=str(base["key_light_vector"]),
            light_flow_desc=str(dop["light_flow_desc"]),
            illuminance_desc=str(dop["illuminance_desc"]),
            color_cast_summary=str(dop.get("color_cast_summary", base["color_cast_summary"])),
            recommended_aperture=forced_aperture or str(dop["aperture"]),
            recommended_focal_mm=int(dop["focal_length_mm"]),  # type: ignore[arg-type]
            camera_id=eff_cam,
            film_id=eff_film,
            dop_rationale=str(dop["rationale"]),
            source="autonomous_dop_director",
        )


class PostGenCohesionMasterer:
    """Post-Generation Anti-AI-Yellow White Balance, Anti-Crunch MTF & Film Stock Color Science Engine."""

    @classmethod
    def master(
        cls,
        canvas_bgr: np.ndarray,
        face_det=None,
        reference_scene_bgr: Optional[np.ndarray] = None,
        target_spec: Optional[SingleLensRadiometricSpec] = None,
        mode: str = "digital_neutral_raw",
        seed: int = 42,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        eff_key = mode if mode in COHESION_MASTERING_MODES else "digital_neutral_raw"
        cfg = COHESION_MASTERING_MODES[eff_key]
        strength = float(cfg["strength"])  # type: ignore[arg-type]
        if strength <= 0.001 or canvas_bgr is None or canvas_bgr.size == 0:
            return canvas_bgr, {"mode": eff_key, "applied": False}

        h, w = canvas_bgr.shape[:2]
        small_w, small_h = max(64, w // 8), max(36, h // 8)

        lab = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]
        A = lab[:, :, 1] - 128.0
        B = lab[:, :, 2] - 128.0

        subj_mask_s = np.zeros((small_h, small_w), dtype=np.float32)
        skin_damper = np.zeros((h, w), dtype=np.float32)
        organ_shield = np.zeros((h, w), dtype=np.float32)

        if face_det is not None:
            fx, fy, fw, fh = face_det.bbox
            cx_f, cy_f = fx + 0.5 * fw, fy + 0.5 * fh
            eye_d = max(10.0, float(face_det.inter_ocular_distance))

            sx = int(round(cx_f * small_w / float(w)))
            sy = int(round((cy_f + 2.2 * fh) * small_h / float(h)))
            rx = max(8, int(round(2.3 * fw * small_w / float(w))))
            ry = max(12, int(round(3.6 * fh * small_h / float(h))))
            cv2.ellipse(subj_mask_s, (sx, sy), (rx, ry), 0.0, 0, 360, 1.0, -1)
            subj_mask_s = cv2.GaussianBlur(subj_mask_s, (0, 0), max(3.0, rx * 0.28))

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

            le, re, nose, lm, rm = face_det.landmarks_5pt
            r_org = max(6, int(round(0.38 * eye_d)))
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

        # ── Stage 1: Optical MTF Anti-Crunch Tamer (calms repetitive 1-2px AI suede/pebble grit outside the face!) ──
        de_crunch = float(cfg.get("de_crunch", 0.22)) * strength  # type: ignore[arg-type]
        if de_crunch > 0.01:
            l_blur_fine = cv2.GaussianBlur(L, (0, 0), 0.95)
            hf_diff = L - l_blur_fine
            face_focus_guard = np.clip(skin_damper + organ_shield, 0.0, 1.0)
            chatter_gate = np.exp(-((hf_diff / 7.5) ** 2)) * (1.0 - face_focus_guard)
            L = L - de_crunch * hf_diff * chatter_gate

        # ── Stage 2: Measure Scene & Optional Reference Envelope ──
        lab_s = cv2.resize(lab, (small_w, small_h), interpolation=cv2.INTER_AREA)
        Ls = lab_s[:, :, 0]
        As = lab_s[:, :, 1] - 128.0
        Bs = lab_s[:, :, 2] - 128.0
        env_bool = subj_mask_s < 0.35
        if int(np.sum(env_bool)) < 64:
            env_bool = np.ones_like(Ls, dtype=bool)
        env_l_p05, env_l_p50, env_l_p90 = [float(v) for v in np.percentile(Ls[env_bool], [5, 50, 90])]

        # Film Stock target chromaticity baseline
        tgt_hi_b = float(cfg.get("target_hi_b", 1.0))  # type: ignore[arg-type]
        tgt_mid_b = float(cfg.get("target_mid_b", 3.5))  # type: ignore[arg-type]
        tgt_sh_b = float(cfg.get("target_sh_b", 0.0))  # type: ignore[arg-type]
        tgt_mid_a = float(cfg.get("target_mid_a", 1.2))  # type: ignore[arg-type]
        tgt_sh_a = float(cfg.get("target_sh_a", 0.0))  # type: ignore[arg-type]
        de_yellow = float(cfg.get("de_yellow", 0.85))  # type: ignore[arg-type]
        tgt_l_floor = env_l_p05

        # If a real reference scene is provided AND de_yellow allows partial scene warmth inheritance, blend gently (capped!)
        if reference_scene_bgr is not None and reference_scene_bgr.size > 0 and not bool(cfg.get("monochrome", False)):
            ref_s = cv2.resize(reference_scene_bgr, (160, 160), interpolation=cv2.INTER_AREA)
            ref_lab = cv2.cvtColor(ref_s, cv2.COLOR_BGR2LAB).astype(np.float32)
            rL = ref_lab[:, :, 0]
            rB = ref_lab[:, :, 2] - 128.0
            r_p05 = float(np.percentile(rL, 5))
            tgt_l_floor = 0.5 * tgt_l_floor + 0.5 * r_p05
            # Cap reference yellow inheritance strictly to <= +8.5 so even warm references stay clean & non-jaundiced!
            ref_b_capped = float(np.clip(np.median(rB), -6.0, 8.5))
            blend_w = 1.0 - 0.65 * de_yellow
            tgt_mid_b = (1.0 - blend_w) * tgt_mid_b + blend_w * ref_b_capped
            tgt_hi_b = (1.0 - blend_w) * tgt_hi_b + blend_w * (ref_b_capped * 0.65)

        # ── Stage 3: True Anti-AI-Yellow White-Balance Neutralizer (`특유의 이상한 노란끼 정밀 제거`) ──
        if bool(cfg.get("monochrome", False)):
            A = A * (1.0 - strength)
            B = B * (1.0 - strength)
        else:
            hi_zone = L >= 170.0
            mid_zone = (L >= 65.0) & (L < 170.0)
            sh_zone = L < 65.0

            cur_hi_b = float(np.median(B[hi_zone])) if np.any(hi_zone) else float(np.median(B))
            cur_mid_b = float(np.median(B[mid_zone])) if np.any(mid_zone) else float(np.median(B))
            cur_sh_b = float(np.median(B[sh_zone])) if np.any(sh_zone) else float(np.median(B))
            cur_hi_a = float(np.median(A[hi_zone])) if np.any(hi_zone) else float(np.median(A))
            cur_mid_a = float(np.median(A[mid_zone])) if np.any(mid_zone) else float(np.median(A))
            cur_sh_a = float(np.median(A[sh_zone])) if np.any(sh_zone) else float(np.median(A))

            # Remove abnormal muddy yellow/orange bias (`cur_b > tgt_b`), while preserving intentional cool twilight airiness!
            if cur_hi_b > tgt_hi_b + 0.5:
                db_hi = -de_yellow * (cur_hi_b - tgt_hi_b)
            else:
                db_hi = 0.22 * (tgt_hi_b - cur_hi_b)

            if cur_mid_b > tgt_mid_b + 0.5:
                db_mid = -de_yellow * 0.85 * (cur_mid_b - tgt_mid_b)
            else:
                db_mid = 0.20 * (tgt_mid_b - cur_mid_b)

            if cur_sh_b > tgt_sh_b + 0.5:
                db_sh = -de_yellow * 0.80 * (cur_sh_b - tgt_sh_b)
            else:
                db_sh = 0.20 * (tgt_sh_b - cur_sh_b)

            # Also pull excess muddy orange-red `a*` (> +5) in highlights/walls back toward clean neutral!
            da_hi = -de_yellow * 0.70 * max(0.0, cur_hi_a - 1.5)
            da_mid = -de_yellow * 0.55 * max(0.0, cur_mid_a - (tgt_mid_a + 3.0)) + 0.35 * (tgt_mid_a - cur_mid_a)
            da_sh = 0.45 * (tgt_sh_a - cur_sh_a) - de_yellow * 0.50 * max(0.0, cur_sh_a - 3.0)

            db_hi = float(np.clip(db_hi, -22.0, 12.0))
            db_mid = float(np.clip(db_mid, -18.0, 12.0))
            db_sh = float(np.clip(db_sh, -16.0, 10.0))
            da_hi = float(np.clip(da_hi, -10.0, 6.0))
            da_mid = float(np.clip(da_mid, -9.0, 6.0))
            da_sh = float(np.clip(da_sh, -9.0, 6.0))

            ln = np.clip(L / 255.0, 0.0, 1.0)
            w_sh = np.clip(1.0 - ln * 2.3, 0.0, 1.0) ** 1.35
            w_hi = np.clip((ln - 0.50) * 2.0, 0.0, 1.0) ** 1.35
            w_mid = np.clip(1.0 - w_sh - w_hi, 0.0, 1.0)

            # On skin, remove abnormal GPT yellow-orange spike while keeping healthy natural warmth (`b*` ~ +10..+13)
            skin_wb_factor = 1.0 - 0.25 * skin_damper
            A = A + strength * skin_wb_factor * (w_sh * da_sh + w_mid * da_mid + w_hi * da_hi)
            B = B + strength * skin_wb_factor * (w_sh * db_sh + w_mid * db_mid + w_hi * db_hi)

            # Harmonize pure neutral catalog patches toward target highlight tint when a warm film stock is explicitly chosen
            chroma = np.hypot(A, B)
            neutral_white_w = (
                np.clip((L - 140.0) / 65.0, 0.0, 1.0)
                * np.clip((10.0 - chroma) / 8.5, 0.0, 1.0)
                * (1.0 - 0.70 * skin_damper)
            )
            B = B + strength * 0.65 * neutral_white_w * (tgt_hi_b - B)

        # ── Stage 4: Foreground Shadow Grounding & Gentle Ambient Irradiance Coupling ──
        fg_bool = subj_mask_s >= 0.45
        if np.any(fg_bool):
            fg_p08 = float(np.percentile(Ls[fg_bool], 8))
            shadow_lift_gap = float(np.clip(fg_p08 - (tgt_l_floor + 4.0), 0.0, 18.0))
            if shadow_lift_gap > 1.5:
                dark_pull_w = np.clip((112.0 - L) / 92.0, 0.0, 1.0) * subj_mask * (1.0 - 0.40 * organ_shield)
                L = L - strength * 0.50 * shadow_lift_gap * dark_pull_w

        illum_field_s = cv2.GaussianBlur(Ls, (0, 0), max(6.0, small_w * 0.16))
        illum_norm_s = (illum_field_s - float(np.mean(illum_field_s))) / max(20.0, float(np.std(illum_field_s)) * 2.4)
        illum_norm = cv2.resize(np.clip(illum_norm_s, -1.0, 1.0), (w, h), interpolation=cv2.INTER_LINEAR)
        L = np.clip(L * (1.0 + strength * 0.04 * illum_norm * (1.0 - 0.65 * skin_damper)), 0.0, 255.0)

        # ── Stage 5: Subtle Optical Light-Wrap + Gentle Highlight Roll-Off (Zero Crunchy S-Curve!) ──
        lw_strength = float(cfg.get("light_wrap_strength", 0.035)) * strength  # type: ignore[arg-type]
        if lw_strength > 0.001:
            bright_env_s = np.clip((Ls - 200.0) / 45.0, 0.0, 1.0) * (1.0 - 0.65 * subj_mask_s)
            if float(np.max(bright_env_s)) > 0.05:
                wrap_glow_s = cv2.GaussianBlur(bright_env_s, (0, 0), max(2.2, small_w * 0.025))
                wrap_glow = cv2.resize(wrap_glow_s, (w, h), interpolation=cv2.INTER_LINEAR)
                receiving_edge = np.clip((180.0 - L) / 120.0, 0.0, 1.0) * (1.0 - organ_shield)
                L = np.clip(L + lw_strength * 28.0 * wrap_glow * receiving_edge, 0.0, 255.0)

        c_boost = float(cfg.get("contrast_boost", 0.03)) * strength  # type: ignore[arg-type]
        if c_boost > 0.001:
            ln_c = np.clip(L / 255.0, 0.0, 1.0)
            L = np.clip((ln_c - c_boost * 0.18 * np.sin(2.0 * np.pi * ln_c) * (1.0 - 0.55 * skin_damper)) * 255.0, 0.0, 255.0)

        hi_excess = np.maximum(0.0, L - 232.0)
        L = np.where(L > 232.0, 232.0 + 15.0 * np.tanh(hi_excess / 18.0), L)

        # ── Stage 6: Subtle Optical Bloom (Window Highlights Only) & Optional Film Emulsion Grain ──
        bloom_k = float(cfg.get("bloom_strength", 0.02)) * strength  # type: ignore[arg-type]
        if bloom_k > 0.001:
            hi_src_s = np.clip((Ls - 215.0) / 38.0, 0.0, 1.0)
            if float(np.max(hi_src_s)) > 0.04:
                bloom_s = cv2.GaussianBlur(hi_src_s, (0, 0), max(2.2, small_w * 0.025))
                bloom_full = cv2.resize(bloom_s, (w, h), interpolation=cv2.INTER_LINEAR)
                bloom_safe = bloom_full * (1.0 - np.clip(skin_damper + organ_shield, 0.0, 1.0))
                L = np.clip(L + bloom_k * 30.0 * bloom_safe * (1.0 - L / 255.0), 0.0, 255.0)

        grain_sigma = float(cfg.get("grain_sigma", 0.0)) * min(1.0, strength)  # type: ignore[arg-type]
        if grain_sigma > 0.05:
            rng = np.random.default_rng(seed)
            # Soft optical emulsion grain (blurred slightly at 0.65px so it NEVER looks like sharp digital noise!)
            g_raw = rng.normal(0.0, 1.0, (h // 2, w // 2)).astype(np.float32)
            g_soft = cv2.resize(g_raw, (w, h), interpolation=cv2.INTER_LINEAR)
            mid_weight = 4.0 * (L / 255.0) * (1.0 - L / 255.0)
            L = np.clip(L + grain_sigma * g_soft * mid_weight * (1.0 - 0.65 * organ_shield), 0.0, 255.0)

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
            "mode": eff_key,
            "applied": True,
            "strength": round(strength, 2),
            "de_yellow_strength": round(de_yellow, 2),
            "de_crunch_strength": round(de_crunch, 2),
            "final_lab_b_median": round(float(np.median(B)), 2),
            "final_lab_hi_b_median": round(float(np.median(B[L >= 170.0])) if np.any(L >= 170.0) else float(np.median(B)), 2),
        }
        return out_bgr, meta
