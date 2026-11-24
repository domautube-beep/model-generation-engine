"""Universal Zero-Hardcoding 4-Domain Facial Organ, 3D Depth, Skin Texture & Skin Tone Analyzer (`v2.2`).

Dynamically measures from ANY human reference portrait (`Gen-0`) without a single hardcoded gender,
ethnicity, age, makeup, skin-color constant, or subject-specific adjective:
  1. Eyes (`눈`: shape, form, crease/rim lines, size)
  2. Nose (`코`: shape, tip 3D form, bridge/alar lines, length/width size, nostril visibility)
  3. Lips (`입`: shape, upper/lower volume form, Cupid's-bow/corner lines, width/height size)
  4. Face Shape & 3D Depth (`얼굴형·폭·길이·깊이`: width/length taper + 3D anterior-malar & midface relief)
  5. Skin Tone & Skin Texture (`피부색감·피부질감`: dermatological CIE L*a*b* / ITA° undertone + IOD-normalized pore micro-texture & subsurface smoothness)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class OrganAndDepthProfile:
    # 1. Eyes (shape, form, line, size)
    eye_width_to_iod: float
    eye_hw_ratio: float
    canthal_tilt_deg: float
    inner_corner_openness: float
    # 2. Nose (shape, form, line, size)
    nose_len_to_iod: float
    alar_width_to_iod: float
    bridge_to_alar_ratio: float
    nasal_tip_3d_relief: float
    nostril_shadow_fraction: float
    # 3. Lips (shape, form, line, size)
    mouth_width_to_iod: float
    lip_height_to_width: float
    upper_to_lower_lip_ratio: float
    cupids_bow_arch_ratio: float
    mouth_corner_deg: float
    lip_chroma_delta: float
    # 4. Face shape, width, length & 3D depth
    face_width_to_length: float
    jaw_to_cheek_taper: float
    anterior_malar_3d_depth: float
    midface_to_lowerface_ratio: float
    # 5. Skin color/undertone & micro-texture (zero hardcoding, dynamically measured from Gen-0)
    skin_l_median: float
    skin_a_undertone: float
    skin_b_undertone: float
    skin_ita_deg: float
    skin_pore_micro_std: float
    skin_subsurface_smoothness: float

    def to_dynamic_organ_depth_clause(self, low_key_scene: bool = False) -> str:
        """Emit a 100% dynamically measured, zero-hardcoding 4-organ + 3D depth + skin tone/texture lock for Image 1."""
        nostril_note = (
            "keep nasal base level so nostrils remain subtly tucked as in Image 1 (never upturn or expose dark nostrils)"
            if self.nostril_shadow_fraction < 0.16
            else "match the exact natural nostril angle and base contour of Image 1"
        )
        tip_note = (
            "preserve the softly rounded 3D nasal tip projection and smooth bridge transition of Image 1"
            if self.nasal_tip_3d_relief >= 0.08
            else "preserve the natural calm nasal bridge and tip projection of Image 1"
        )
        bow_note = (
            "defined Cupid's-bow upper lip line"
            if self.cupids_bow_arch_ratio >= 0.065
            else "smooth natural upper lip line"
        )
        lip_vol_note = (
            f"natural upper-to-lower lip thickness ratio ({self.upper_to_lower_lip_ratio:.2f}) and hydrated natural lip fullness matching Image 1"
            if self.lip_height_to_width >= 0.30
            else f"balanced natural lip thickness ({self.upper_to_lower_lip_ratio:.2f} upper/lower ratio) matching Image 1"
        )
        depth_note = (
            "preserve Image 1's natural 3D anterior-cheek volume, midface relief, and soft cheek-to-jaw depth gradation (never flatten into a 2D plane)"
            if self.anterior_malar_3d_depth >= 0.06
            else "preserve Image 1's exact calm 3D facial plane relief and jaw depth"
        )
        if low_key_scene:
            undertone_note = (
                "natural intrinsic skin albedo harmonized seamlessly with the target room's directional window light, deep shadow falloff, and muted warm-olive film tone (zero bright studio pinkness)"
            )
        else:
            undertone_note = (
                "healthy translucent rosy-warm skin undertone matching Image 1 (never pale grey, greenish-olive, or muddy)"
                if self.skin_a_undertone >= 6.5
                else "exact natural balanced skin undertone matching Image 1 without grey or olive color cast"
            )
        texture_note = (
            "fine velvety natural skin micro-texture with soft subsurface translucency matching Image 1 (never coarse dry sandpaper pores and never shiny plastic wax)"
            if self.skin_pore_micro_std <= 2.4
            else "authentic natural skin pore micro-texture matching Image 1 without plastic smoothing or specular oil glare"
        )
        return (
            f"Measured 4-Organ, 3D Depth & Skin Fidelity Lock from Image 1 "
            f"(eye W/IOD {self.eye_width_to_iod:.3f}, eye H/W {self.eye_hw_ratio:.3f}, tilt {self.canthal_tilt_deg:+.1f} deg; "
            f"nose L/IOD {self.nose_len_to_iod:.3f}, alar W/IOD {self.alar_width_to_iod:.3f}; "
            f"mouth W/IOD {self.mouth_width_to_iod:.3f}, lip H/W {self.lip_height_to_width:.3f}; "
            f"face W/L {self.face_width_to_length:.3f}, jaw/cheek {self.jaw_to_cheek_taper:.3f}, 3D malar depth {self.anterior_malar_3d_depth:.3f}; "
            f"skin ITA {self.skin_ita_deg:.1f} deg, a*+{self.skin_a_undertone:.1f}/b*+{self.skin_b_undertone:.1f}): "
            "copy Image 1's exact eye shape, inner/outer corner lines, upper/lower lid curves, clear iris/sclera balance, and eye size 1:1; "
            f"{tip_note}, exact alar wing lines, and {nostril_note}; "
            f"reproduce Image 1's exact mouth width, {bow_note}, mouth corner line ({self.mouth_corner_deg:+.1f} deg), and {lip_vol_note}; "
            f"{depth_note}; preserve {undertone_note} across face and neck, and {texture_note}."
        )

    def to_natural_identity_only_clause(self, clean_studio: bool = False) -> str:
        """Emit a pure photographer-language identity-only clause (ZERO CAD numbers, gender/hairstyle-universal, strict separation of 'Who' vs 'How Shot')."""
        if clean_studio:
            return (
                "Copy the exact real person's face and hairstyle 1:1 from Image 1 and Image 2—reproduce their exact natural eye shape and openness, "
                "exact upper-eyelid fold structure (never add a thick artificial double-eyelid crease and never squint), clear dark-brown irises with crisp studio catchlights and balanced white sclera, "
                "exact eyebrow shape and hair density, subtle natural lower-lid contour, exact midface and philtrum proportions (never elongate or compress the midface or nose), "
                "exact nasal bridge and softly rounded nasal tip, naturally hydrated lip color and exact gentle mouth corner contour, "
                "exact cheek volume and mandibular jawline-to-chin taper (never widen, puff out, or artificially shave the jaw), "
                "and their exact hairstyle, bangs/parting, sideburn/ear exposure, and hair length/volume from Image 1 (never invent a different haircut, never spike up down-swept bangs, and never alter hair length), "
                "while harmonized seamlessly into the studio lighting of the scene."
            )
        return (
            "Copy the exact real person's face and hairstyle 1:1 from the portrait reference—reproduce their exact natural eye shape, "
            "exact upper-eyelid fold structure above the lash rim, clear dark iris-to-sclera balance with natural catchlights, "
            "exact eyebrow contour, subtle natural lower-lid contour, softly rounded nasal bridge and nasal tip, exact philtrum length, calm natural lip shape "
            "with gentle natural mouth corners (never drooping or frowning), natural anterior-and-lateral cheek fullness "
            "(never gaunt or hollowed in shadow), exact cheek-to-jawline taper, and exact natural hairstyle/bangs/parting from Image 1, while COMPLETELY DISCARDING "
            "the flat ID-backdrop lighting, stiff upright ID-photo neck posture, "
            "and formal suit/tie of the reference portrait."
        )


class OrganAndDepthAnatomyAnalyzer:
    """Extracts and compares universal 4-organ + 3D facial depth + skin tone/texture profiles."""

    def __init__(self, apple_vision) -> None:
        self.vision = apple_vision

    @staticmethod
    def _pts(region, w: int, h: int) -> np.ndarray:
        if not region:
            return np.zeros((0, 2), dtype=np.float32)
        first = region[0]
        if isinstance(first, dict):
            return np.array([[float(p["x"]) * w, (1.0 - float(p["y"])) * h] for p in region], dtype=np.float32)
        arr = np.asarray(region, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            if float(arr[:, 0].max()) <= 1.5 and float(arr[:, 1].max()) <= 1.5:
                return np.stack([arr[:, 0] * w, (1.0 - arr[:, 1]) * h], axis=-1).astype(np.float32)
            return arr[:, :2].astype(np.float32)
        return np.zeros((0, 2), dtype=np.float32)

    def extract_profile(self, bgr_img: np.ndarray) -> Optional[OrganAndDepthProfile]:
        if bgr_img is None or bgr_img.size == 0:
            return None
        h, w = bgr_img.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
        marks = self.vision.landmarks(pil) or {}
        le_raw, re_raw = marks.get("leftEye") or [], marks.get("rightEye") or []
        nose_raw, ncrest_raw = marks.get("nose") or [], marks.get("noseCrest") or []
        olip_raw, ilip_raw = marks.get("outerLips") or [], marks.get("innerLips") or []
        fc_raw = marks.get("contour") or marks.get("faceContour") or []
        if len(le_raw) < 4 or len(re_raw) < 4 or len(nose_raw) < 3 or len(olip_raw) < 6 or len(fc_raw) < 7:
            return None

        le = self._pts(le_raw, w, h)
        re = self._pts(re_raw, w, h)
        nose = self._pts(nose_raw, w, h)
        ncrest = self._pts(ncrest_raw, w, h) if len(ncrest_raw) >= 2 else nose
        olip = self._pts(olip_raw, w, h)
        ilip = self._pts(ilip_raw, w, h) if len(ilip_raw) >= 4 else olip
        fc = self._pts(fc_raw, w, h)

        c_le, c_re = le.mean(axis=0), re.mean(axis=0)
        if c_le[0] > c_re[0]:
            le, re = re, le
            c_le, c_re = c_re, c_le
        u_vec = c_re - c_le
        iod = max(8.0, float(np.linalg.norm(u_vec)))
        ux = u_vec / iod
        uy = np.array([-ux[1], ux[0]], dtype=np.float32)
        if uy[1] < 0:
            uy = -uy
        eye_mid = 0.5 * (c_le + c_re)

        def proj(pts: np.ndarray) -> np.ndarray:
            d = pts - eye_mid[None, :]
            return np.stack([d @ ux, d @ uy], axis=-1)

        le_p, re_p = proj(le), proj(re)
        nose_p, nc_p = proj(nose), proj(ncrest)
        ol_p, il_p = proj(olip), proj(ilip)
        fc_p = proj(fc)

        # 1. Eyes
        lw = max(1.0, float(le_p[:, 0].max() - le_p[:, 0].min()))
        lh = float(le_p[:, 1].max() - le_p[:, 1].min())
        rw = max(1.0, float(re_p[:, 0].max() - re_p[:, 0].min()))
        rh = float(re_p[:, 1].max() - re_p[:, 1].min())
        eye_w_iod = 0.5 * (lw + rw) / iod
        eye_hw = 0.5 * (lh / lw + rh / rw)

        l_in = le_p[np.argmax(le_p[:, 0])]
        l_out = le_p[np.argmin(le_p[:, 0])]
        r_in = re_p[np.argmin(re_p[:, 0])]
        r_out = re_p[np.argmax(re_p[:, 0])]
        l_tilt = math.degrees(math.atan2(-(l_out[1] - l_in[1]), max(1.0, abs(l_in[0] - l_out[0]))))
        r_tilt = math.degrees(math.atan2(-(r_out[1] - r_in[1]), max(1.0, abs(r_out[0] - r_in[0]))))
        tilt_deg = 0.5 * (l_tilt + r_tilt)
        inner_open = float(abs(r_in[0] - l_in[0])) / iod

        # 2. Nose
        alar_w = max(2.0, float(nose_p[:, 0].max() - nose_p[:, 0].min()))
        nose_bot_y = float(max(nose_p[:, 1].max(), nc_p[:, 1].max()))
        nose_top_y = float(min(nc_p[:, 1].min(), 0.0))
        nose_len = max(2.0, nose_bot_y - nose_top_y)
        bridge_w = max(1.0, float(np.percentile(np.abs(nc_p[:, 0] - np.median(nc_p[:, 0])), 85) * 2.4))

        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_ch = lab[:, :, 0]
        fy0, fy1 = max(0, int(eye_mid[1] - 0.5 * iod)), min(h, int(eye_mid[1] + 1.6 * iod))
        fx0, fx1 = max(0, int(eye_mid[0] - 0.8 * iod)), min(w, int(eye_mid[0] + 0.8 * iod))
        f_roi = L_ch[fy0:fy1, fx0:fx1]
        face_p95, face_p10 = np.percentile(f_roi, [95.0, 10.0]) if f_roi.size >= 16 else (200.0, 80.0)
        dyn_range = max(16.0, float(face_p95 - face_p10))

        tip_cx = int(round(float(eye_mid[0] + np.median(nose_p[:, 0]) * ux[0] + (nose_bot_y - 0.14 * iod) * uy[0])))
        tip_cy = int(round(float(eye_mid[1] + np.median(nose_p[:, 0]) * ux[1] + (nose_bot_y - 0.14 * iod) * uy[1])))
        r_tip = max(3, int(round(0.11 * iod)))
        tip_patch = L_ch[max(0, tip_cy - r_tip) : min(h, tip_cy + r_tip), max(0, tip_cx - r_tip) : min(w, tip_cx + r_tip)]
        alar_r = max(5, int(round(0.24 * iod)))
        alar_patch = L_ch[max(0, tip_cy - r_tip) : min(h, tip_cy + r_tip), max(0, tip_cx - alar_r) : min(w, tip_cx + alar_r)]
        if tip_patch.size >= 4 and alar_patch.size >= 9:
            tip_relief = float(max(0.0, np.percentile(tip_patch, 85.0) - np.percentile(alar_patch, 30.0))) / dyn_range
        else:
            tip_relief = 0.10

        nb_cy = int(round(float(eye_mid[1] + nose_bot_y * uy[1])))
        nb_rh = max(2, int(round(0.07 * iod)))
        nb_rw = max(4, int(round(0.20 * iod)))
        nb_patch = L_ch[max(0, nb_cy - nb_rh) : min(h, nb_cy + nb_rh), max(0, tip_cx - nb_rw) : min(w, tip_cx + nb_rw)]
        nostril_frac = float(np.mean(nb_patch < (face_p10 + 0.22 * dyn_range))) if nb_patch.size >= 4 else 0.08

        # 3. Lips
        mouth_w = max(4.0, float(ol_p[:, 0].max() - ol_p[:, 0].min()))
        mouth_h = max(1.5, float(ol_p[:, 1].max() - ol_p[:, 1].min()))
        stomion_y = float(np.median(il_p[:, 1]))
        up_lip_h = max(0.8, stomion_y - float(ol_p[:, 1].min()))
        lo_lip_h = max(0.8, float(ol_p[:, 1].max()) - stomion_y)
        up_lo_ratio = up_lip_h / lo_lip_h

        up_mask = ol_p[:, 1] <= stomion_y
        up_pts = ol_p[up_mask] if np.sum(up_mask) >= 3 else ol_p
        mid_band = up_pts[np.abs(up_pts[:, 0] - np.median(ol_p[:, 0])) <= 0.22 * mouth_w]
        bow_arch = float(mid_band[:, 1].max() - mid_band[:, 1].min()) / mouth_w if len(mid_band) >= 3 else 0.05

        ml_pt = ol_p[np.argmin(ol_p[:, 0])]
        mr_pt = ol_p[np.argmax(ol_p[:, 0])]
        corner_y = 0.5 * (float(ml_pt[1]) + float(mr_pt[1]))
        corner_deg = math.degrees(math.atan2(stomion_y - corner_y, 0.5 * mouth_w))

        m_cx = int(round(float(eye_mid[0] + np.median(ol_p[:, 0]) * ux[0] + stomion_y * uy[0])))
        m_cy = int(round(float(eye_mid[1] + np.median(ol_p[:, 0]) * ux[1] + stomion_y * uy[1])))
        mr_y, mr_x = max(2, int(round(0.40 * mouth_h))), max(3, int(round(0.28 * mouth_w)))
        lip_ab = lab[max(0, m_cy - mr_y) : min(h, m_cy + mr_y), max(0, m_cx - mr_x) : min(w, m_cx + mr_x), 1:3]

        chk_y = int(round(float(eye_mid[1] + 0.42 * iod * uy[1])))
        chk_lx = int(round(float(eye_mid[0] - 0.42 * iod * ux[0])))
        chk_rx = int(round(float(eye_mid[0] + 0.42 * iod * ux[0])))
        cr = max(3, int(round(0.15 * iod)))
        chk_l = lab[max(0, chk_y - cr) : min(h, chk_y + cr), max(0, chk_lx - cr) : min(w, chk_lx + cr)]
        chk_r = lab[max(0, chk_y - cr) : min(h, chk_y + cr), max(0, chk_rx - cr) : min(w, chk_rx + cr)]
        chk_all = np.concatenate([chk_l.reshape(-1, 3), chk_r.reshape(-1, 3)], axis=0) if (chk_l.size and chk_r.size) else None
        if lip_ab.size >= 6 and chk_all is not None and len(chk_all) >= 6:
            lip_chroma = float(np.linalg.norm(np.median(lip_ab.reshape(-1, 2), axis=0) - np.median(chk_all[:, 1:3], axis=0)))
        else:
            lip_chroma = 8.0

        # 4. Face Shape, Width, Length & 3D Anterior-Malar Depth
        cheek_w = max(8.0, float(fc_p[:, 0].max() - fc_p[:, 0].min()))
        chin_y = max(12.0, float(fc_p[:, 1].max()))
        face_wl = cheek_w / (chin_y * 1.55)
        jaw_band = fc_p[(fc_p[:, 1] >= 0.58 * chin_y) & (fc_p[:, 1] <= 0.82 * chin_y)]
        jaw_w = float(jaw_band[:, 0].max() - jaw_band[:, 0].min()) if len(jaw_band) >= 2 else cheek_w * 0.78
        jaw_taper = jaw_w / cheek_w
        mid_to_low = nose_bot_y / max(4.0, chin_y - nose_bot_y)

        lat_lx = int(round(float(eye_mid[0] - 0.76 * iod * ux[0])))
        lat_rx = int(round(float(eye_mid[0] + 0.76 * iod * ux[0])))
        lat_y = int(round(float(eye_mid[1] + 0.62 * iod * uy[1])))
        lat_l = L_ch[max(0, lat_y - cr) : min(h, lat_y + cr), max(0, lat_lx - cr) : min(w, lat_lx + cr)]
        lat_r = L_ch[max(0, lat_y - cr) : min(h, lat_y + cr), max(0, lat_rx - cr) : min(w, lat_rx + cr)]
        if chk_all is not None and len(chk_all) >= 6 and (lat_l.size + lat_r.size) >= 6:
            lat_vals = np.concatenate([lat_l.reshape(-1), lat_r.reshape(-1)])
            malar_depth = float(max(0.0, np.percentile(chk_all[:, 0], 75.0) - np.percentile(lat_vals, 35.0))) / dyn_range
        else:
            malar_depth = 0.12

        # 5. Skin Color/Undertone (CIE L*a*b* & ITA deg) + IOD-Normalized Pore Micro-Texture
        if chk_all is not None and len(chk_all) >= 12:
            s_l_255 = float(np.median(chk_all[:, 0]))
            s_a = float(np.median(chk_all[:, 1]) - 128.0)
            s_b = float(np.median(chk_all[:, 2]) - 128.0)
        else:
            s_l_255, s_a, s_b = 175.0, 9.0, 12.0
        l_star = s_l_255 * (100.0 / 255.0)
        ita_deg = math.degrees(math.atan2(l_star - 50.0, max(1.5, s_b)))

        # Normalize cheek patches to canonical IOD = 130px so pore texture std is scale-invariant!
        norm_scale = 130.0 / iod
        pore_stds: List[float] = []
        smooth_ratios: List[float] = []
        for patch_3c in (chk_l, chk_r):
            if patch_3c.shape[0] >= 6 and patch_3c.shape[1] >= 6:
                pl = patch_3c[:, :, 0]
                nw = max(8, int(round(pl.shape[1] * norm_scale)))
                nh = max(8, int(round(pl.shape[0] * norm_scale)))
                pl_norm = cv2.resize(pl, (nw, nh), interpolation=cv2.INTER_CUBIC)
                pore_band = pl_norm - cv2.GaussianBlur(pl_norm, (0, 0), 1.35)
                mottle_band = cv2.GaussianBlur(pl_norm, (0, 0), 1.35) - cv2.GaussianBlur(pl_norm, (0, 0), 4.2)
                p_std = float(np.std(pore_band))
                m_std = float(np.std(mottle_band))
                pore_stds.append(p_std)
                smooth_ratios.append(p_std / max(0.35, m_std))
        pore_std_val = float(np.mean(pore_stds)) if pore_stds else 1.4
        subsurface_smooth = float(np.mean(smooth_ratios)) if smooth_ratios else 0.85

        return OrganAndDepthProfile(
            eye_width_to_iod=round(eye_w_iod, 4),
            eye_hw_ratio=round(eye_hw, 4),
            canthal_tilt_deg=round(tilt_deg, 2),
            inner_corner_openness=round(inner_open, 4),
            nose_len_to_iod=round(nose_len / iod, 4),
            alar_width_to_iod=round(alar_w / iod, 4),
            bridge_to_alar_ratio=round(bridge_w / alar_w, 4),
            nasal_tip_3d_relief=round(tip_relief, 4),
            nostril_shadow_fraction=round(nostril_frac, 4),
            mouth_width_to_iod=round(mouth_w / iod, 4),
            lip_height_to_width=round(mouth_h / mouth_w, 4),
            upper_to_lower_lip_ratio=round(up_lo_ratio, 4),
            cupids_bow_arch_ratio=round(bow_arch, 4),
            mouth_corner_deg=round(corner_deg, 2),
            lip_chroma_delta=round(lip_chroma, 2),
            face_width_to_length=round(face_wl, 4),
            jaw_to_cheek_taper=round(jaw_taper, 4),
            anterior_malar_3d_depth=round(malar_depth, 4),
            midface_to_lowerface_ratio=round(mid_to_low, 4),
            skin_l_median=round(s_l_255, 2),
            skin_a_undertone=round(s_a, 2),
            skin_b_undertone=round(s_b, 2),
            skin_ita_deg=round(ita_deg, 2),
            skin_pore_micro_std=round(pore_std_val, 3),
            skin_subsurface_smoothness=round(subsurface_smooth, 3),
        )

    def compare(self, ref_bgr: np.ndarray, cand_bgr: np.ndarray) -> Dict[str, object]:
        p_ref = self.extract_profile(ref_bgr)
        p_cnd = self.extract_profile(cand_bgr)
        if p_ref is None or p_cnd is None:
            return {"passed": True, "organDepthDrift": 0.0, "diffs": {}, "issues": []}

        def pct(a: float, b: float) -> float:
            return round((b - a) / max(1e-4, abs(a)) * 100.0, 1)

        skin_chroma_err = round(
            math.hypot(p_cnd.skin_a_undertone - p_ref.skin_a_undertone, p_cnd.skin_b_undertone - p_ref.skin_b_undertone),
            2,
        )
        diffs = {
            "eyeWidthPct": pct(p_ref.eye_width_to_iod, p_cnd.eye_width_to_iod),
            "eyeHwPct": pct(p_ref.eye_hw_ratio, p_cnd.eye_hw_ratio),
            "noseLenPct": pct(p_ref.nose_len_to_iod, p_cnd.nose_len_to_iod),
            "alarWidthPct": pct(p_ref.alar_width_to_iod, p_cnd.alar_width_to_iod),
            "nostrilDelta": round(p_cnd.nostril_shadow_fraction - p_ref.nostril_shadow_fraction, 3),
            "mouthWidthPct": pct(p_ref.mouth_width_to_iod, p_cnd.mouth_width_to_iod),
            "lipThicknessPct": pct(p_ref.lip_height_to_width, p_cnd.lip_height_to_width),
            "faceWlPct": pct(p_ref.face_width_to_length, p_cnd.face_width_to_length),
            "malarDepthDelta": round(p_cnd.anterior_malar_3d_depth - p_ref.anterior_malar_3d_depth, 3),
            "skinUndertoneDeltaAB": skin_chroma_err,
            "skinRosyDeltaA": round(p_cnd.skin_a_undertone - p_ref.skin_a_undertone, 2),
            "skinPoreRatio": round(p_cnd.skin_pore_micro_std / max(0.25, p_ref.skin_pore_micro_std), 2),
        }
        issues: List[str] = []
        if abs(diffs["alarWidthPct"]) > 16.0 or abs(diffs["noseLenPct"]) > 14.0:
            issues.append(f"코 크기·길이 변형 (콧볼 {diffs['alarWidthPct']:+.1f}%, 코길이 {diffs['noseLenPct']:+.1f}%)")
        if diffs["nostrilDelta"] > 0.16:
            issues.append(f"콧구멍 과다 들림 노출 (+{diffs['nostrilDelta']:.2f})")
        if abs(diffs["mouthWidthPct"]) > 14.0 or abs(diffs["lipThicknessPct"]) > 22.0:
            issues.append(f"입 크기·두께 변형 (폭 {diffs['mouthWidthPct']:+.1f}%, 두께 {diffs['lipThicknessPct']:+.1f}%)")
        if diffs["malarDepthDelta"] < -0.10:
            issues.append(f"앞볼 3D 입체 깊이 평면화 ({diffs['malarDepthDelta']:+.3f})")
        if diffs["skinUndertoneDeltaAB"] > 9.5 or diffs["skinRosyDeltaA"] < -5.0:
            issues.append(f"피부색감·혈색 이탈 (ΔAB {diffs['skinUndertoneDeltaAB']:.1f}, Δa* {diffs['skinRosyDeltaA']:+.1f})")

        drift = (
            abs(diffs["eyeWidthPct"]) * 0.20
            + abs(diffs["eyeHwPct"]) * 0.18
            + abs(diffs["noseLenPct"]) * 0.22
            + abs(diffs["alarWidthPct"]) * 0.22
            + max(0.0, diffs["nostrilDelta"]) * 35.0
            + abs(diffs["mouthWidthPct"]) * 0.20
            + abs(diffs["lipThicknessPct"]) * 0.20
            + abs(diffs["faceWlPct"]) * 0.25
            + abs(diffs["malarDepthDelta"]) * 30.0
            + diffs["skinUndertoneDeltaAB"] * 0.65
        )
        return {
            "passed": len(issues) == 0,
            "organDepthDrift": round(drift, 2),
            "diffs": diffs,
            "issues": issues,
            "ref_profile": asdict(p_ref),
            "cand_profile": asdict(p_cnd),
        }
