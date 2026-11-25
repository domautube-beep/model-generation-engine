"""Universal Zero-Hardcoding 7-Metric Ocular Anatomy Analyzer (`눈매 · 쌍꺼풀 높이 · 홍채-흰자 비율 · 눈꼬리 각도`).

Measures real eye anatomy directly from any `Gen-0` portrait using Apple Vision landmarks +
pixel-level lash-line / crease-ridge / iris-sclera segmentation — with ZERO hardcoded traits.

Why landmarks alone are not enough:
Coarse 6-point eye polygons can show <2% aspect-ratio error while the generated eye still looks
like a different person because of:
1. Eyelid crease height/thickness (`속쌍/인라인` vs `두꺼운 겉쌍 아치`)
2. Iris-to-eye-width diameter ratio (`자연 홍채 ~40%` vs `써클렌즈 과대 홍채 ~60%`)
3. Exposed sclera fraction (`좌우 흰자 노출량`)
4. Outer-corner canthal tilt (`수평/순한 눈꼬리` vs `날카롭게 올라간 눈꼬리`)
5. Aligned horizontal eyelid/lash-line pixel correlation (`eyelid_edge_ncc`)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .apple_vision import AppleVisionAnalyzer


@dataclass(frozen=True)
class OcularAnatomyProfile:
    """Measured 7-metric ocular geometry & pixel-level lid/iris/sclera profile from a real photograph."""

    avg_ear: float  # Palpebral fissure Height / Width
    avg_canthal_tilt_deg: float  # Positive = outer corner higher than inner corner
    avg_upper_arch_ratio: float
    avg_lower_drop_ratio: float
    avg_brow_gap_ratio: float
    inter_canthal_to_iod: float
    avg_iris_to_eye_width: float  # Dark iris horizontal span / eye width (0.34..0.65)
    avg_sclera_visible_ratio: float  # Bright sclera area fraction inside eye aperture
    avg_crease_gap_ratio: float  # Vertical distance from upper lash line to strongest crease ridge / eye height
    crease_prominence: float  # Relative Sobel-Y ridge strength of supratarsal crease (0.0..1.0+)
    left_iris_to_eye_width: float = 0.42
    right_iris_to_eye_width: float = 0.42

    def crease_category(self) -> str:
        """Classify eyelid crease strictly from measured supratarsal ridge height (`avg_crease_gap_ratio`) & prominence."""
        if self.avg_crease_gap_ratio <= 0.36 or self.crease_prominence < 0.24:
            return "tight_inline_or_monolid"
        if self.avg_crease_gap_ratio <= 0.60:
            return "subtle_inline_double"
        return "prominent_outline_double"

    def to_ocular_lock_clause(self) -> str:
        """Emit a 100% universal, zero-hardcoding ocular anatomy lock directive derived purely from measured Gen-0 eyes."""
        cat = self.crease_category()
        if cat == "tight_inline_or_monolid":
            lid_desc = (
                "ultra-thin, tight in-line upper eyelid fold sitting almost flush against the natural lash line "
                "(NEVER draw a thick, dark, prominent out-line double-eyelid crease arch floating above the eye)"
            )
        elif cat == "subtle_inline_double":
            lid_desc = (
                "delicate, narrow natural in-line eyelid crease hugging closely to the upper lash line "
                "(avoid thick dramatic high-set western or cartoon crease arcs)"
            )
        else:
            lid_desc = "clear, well-defined natural double-eyelid crease matching Image 1's exact fold height"

        if self.avg_iris_to_eye_width <= 0.46:
            iris_pct = int(round(self.avg_iris_to_eye_width * 100.0))
            iris_desc = (
                f"natural medium-sized irises occupying only ~{iris_pct}% of horizontal eye width with crisp, bright "
                "white sclera clearly visible on BOTH sides of each iris and subtle pink inner corner (NEVER enlarge "
                "irises into oversized dark circle-lens doll eyes)"
            )
        else:
            iris_pct = int(round(self.avg_iris_to_eye_width * 100.0))
            iris_desc = (
                f"natural irises occupying ~{iris_pct}% of horizontal eye width with balanced visible white sclera "
                "matching Image 1 exactly"
            )

        if self.avg_canthal_tilt_deg <= 2.5:
            tilt_desc = f"gentle horizontal-to-softly-relaxed outer eye corners (canthal tilt ~{self.avg_canthal_tilt_deg:+.1f} deg, NEVER sharp upturned cat-eye corners)"
        elif self.avg_canthal_tilt_deg <= 6.8:
            tilt_desc = f"soft, subtly lifted almond outer eye corners (canthal tilt ~{self.avg_canthal_tilt_deg:+.1f} deg) with gently curved lower lash rim and subtle natural lower-lid fullness"
        else:
            tilt_desc = f"distinctly lifted almond outer eye corners (canthal tilt ~{self.avg_canthal_tilt_deg:+.1f} deg) matching Image 1"

        return (
            f"Measured Ocular Anatomy Lock (palpebral H/W {self.avg_ear:.3f}, canthal tilt {self.avg_canthal_tilt_deg:+.1f} deg, "
            f"iris-to-eye-width {self.avg_iris_to_eye_width:.2f}, sclera ratio {self.avg_sclera_visible_ratio:.2f}, "
            f"crease gap {self.avg_crease_gap_ratio:.2f}, brow-eye gap {self.avg_brow_gap_ratio:.2f}): "
            f"reproduce Image 1's exact eye shape 1:1—preserve {lid_desc}; preserve {iris_desc}; preserve {tilt_desc}; "
            "preserve natural fine individual eyebrow hairs and exact brow-to-eye spacing without thickening or darkening the brows."
        )


class OcularAnatomyAnalyzer:
    """Extracts & compares 7 pixel-level + landmark ocular features between Gen-0 and generated candidates."""

    def __init__(self, vision: AppleVisionAnalyzer) -> None:
        self.vision = vision

    @staticmethod
    def _measure_single_eye_pixels(
        gray_img: np.ndarray,
        eye_pts: np.ndarray,
        is_left: bool,
    ) -> Tuple[float, float, float, float]:
        """Measure (iris_to_eye_width, sclera_visible_ratio, crease_gap_ratio, crease_prominence) in pixel space."""
        h_img, w_img = gray_img.shape[:2]
        x_min = max(0, int(math.floor(float(eye_pts[:, 0].min()))))
        x_max = min(w_img - 1, int(math.ceil(float(eye_pts[:, 0].max()))))
        y_min = max(0, int(math.floor(float(eye_pts[:, 1].min()))))
        y_max = min(h_img - 1, int(math.ceil(float(eye_pts[:, 1].max()))))
        ew = max(4, x_max - x_min)
        eh = max(3, y_max - y_min)

        # Normalize crop to fixed width 120px for scale-invariant pixel measurement
        pad_x = max(4, int(round(ew * 0.18)))
        pad_top = max(6, int(round(eh * 1.15)))
        pad_bot = max(4, int(round(eh * 0.45)))
        cx1, cx2 = max(0, x_min - pad_x), min(w_img, x_max + pad_x)
        cy1, cy2 = max(0, y_min - pad_top), min(h_img, y_max + pad_bot)
        patch = gray_img[cy1:cy2, cx1:cx2]
        if patch.size < 36:
            return 0.42, 0.48, 0.25, 0.20

        scale = 120.0 / float(ew)
        rw = max(16, int(round(patch.shape[1] * scale)))
        rh = max(16, int(round(patch.shape[0] * scale)))
        norm_p = cv2.resize(patch, (rw, rh), interpolation=cv2.INTER_CUBIC).astype(np.float32)

        pts_local = (eye_pts - np.array([cx1, cy1], dtype=np.float32)) * scale
        mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(pts_local).astype(np.int32)], 255)
        eye_px = norm_p[mask > 0]
        if eye_px.size < 20:
            return 0.42, 0.48, 0.25, 0.20

        # Iris vs Sclera inside the eye polygon:
        # Dark iris columns have low vertical median luminance; sclera columns have high luminance
        p15, p85 = float(np.percentile(eye_px, 15)), float(np.percentile(eye_px, 85))
        dyn = max(12.0, p85 - p15)
        iris_thr = p15 + 0.42 * dyn
        sclera_thr = p15 + 0.56 * dyn

        ex1 = max(0, int(round(float(pts_local[:, 0].min()))))
        ex2 = min(rw, int(round(float(pts_local[:, 0].max()))))
        ey1 = max(0, int(round(float(pts_local[:, 1].min()))))
        ey2 = min(rh, int(round(float(pts_local[:, 1].max()))))
        n_ew = max(1, ex2 - ex1)
        n_eh = max(1, ey2 - ey1)

        # Horizontal iris width: contiguous central dark run across columns inside the eye aperture
        dark_cols = []
        for col in range(ex1, ex2):
            col_m = mask[ey1:ey2, col] > 0
            if np.any(col_m):
                col_vals = norm_p[ey1:ey2, col][col_m]
                # Use 35th percentile so bright catchlights inside the pupil don't break iris detection
                if float(np.percentile(col_vals, 35)) < iris_thr:
                    dark_cols.append(col)
        if dark_cols:
            # Measure longest contiguous dark band (tolerating studio catchlight reflection inside pupil up to ~11% eye width)
            max_catchlight_gap = max(4, int(round(0.11 * n_ew)))
            best_run = 1
            cur_run = 1
            for k in range(1, len(dark_cols)):
                if dark_cols[k] - dark_cols[k - 1] <= max_catchlight_gap:
                    cur_run += dark_cols[k] - dark_cols[k - 1]
                else:
                    cur_run = 1
                best_run = max(best_run, cur_run)
            iris_w_ratio = float(np.clip(best_run / float(n_ew), 0.25, 0.78))
        else:
            iris_w_ratio = 0.42

        sclera_ratio = float(np.clip(np.mean(eye_px >= sclera_thr), 0.12, 0.80))

        # Supratarsal eyelid crease detection ABOVE the upper lash line:
        # Equalize sharpness via GaussianBlur(5x5, 1.1) before Sobel-Y so Upscayl 4x crispness never triggers false crease prominence!
        norm_eq = cv2.GaussianBlur(norm_p, (5, 5), 1.1)
        sobel_y = np.abs(cv2.Sobel(norm_eq, cv2.CV_32F, 0, 1, ksize=3))
        cx_l = ex1 + int(round(0.20 * n_ew))
        cx_r = ex1 + int(round(0.80 * n_ew))
        if cx_r <= cx_l + 4:
            return iris_w_ratio, sclera_ratio, 0.25, 0.20

        # Lash-line reference edge strength around `ey1`
        lash_band_top = max(0, ey1 - max(2, int(round(0.14 * n_eh))))
        lash_band_bot = min(rh, ey1 + max(3, int(round(0.22 * n_eh))))
        lash_energy = float(np.mean(sobel_y[lash_band_top:lash_band_bot, cx_l:cx_r])) + 1e-4

        # Search supratarsal zone strictly ABOVE lash line: [ey1 - 0.95*n_eh .. ey1 - 0.16*n_eh]
        s_top = max(0, ey1 - max(4, int(round(0.95 * n_eh))))
        s_bot = max(s_top + 2, ey1 - max(2, int(round(0.16 * n_eh))))
        row_profile = np.mean(sobel_y[s_top:s_bot, cx_l:cx_r], axis=1)
        if row_profile.size >= 3:
            peak_idx = int(np.argmax(row_profile))
            peak_val = float(row_profile[peak_idx])
            peak_y = s_top + peak_idx
            crease_gap = float(np.clip((ey1 - peak_y) / float(n_eh), 0.10, 1.0))
            # Compare peak ridge against median skin energy in supratarsal zone AND lash energy
            bg_med = float(np.median(row_profile)) + 1e-4
            prominence = float(np.clip((peak_val - bg_med) / lash_energy, 0.0, 1.5))
        else:
            crease_gap, prominence = 0.25, 0.15

        return iris_w_ratio, sclera_ratio, crease_gap, prominence

    def analyze(self, bgr_img: np.ndarray, marks: Optional[Dict[str, object]] = None) -> Optional[OcularAnatomyProfile]:
        """Extract the full 7-metric OcularAnatomyProfile from any face/portrait BGR image."""
        if marks is None:
            pil = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
            marks = self.vision.landmarks(pil)
        if not marks:
            return None
        for req in ("leftEye", "rightEye", "leftEyebrow", "rightEyebrow"):
            if not marks.get(req) or len(marks[req]) < 3:
                return None

        le = np.array(marks["leftEye"], dtype=np.float32)
        re = np.array(marks["rightEye"], dtype=np.float32)
        leb = np.array(marks["leftEyebrow"], dtype=np.float32)
        reb = np.array(marks["rightEyebrow"], dtype=np.float32)
        lp = np.mean(np.array(marks.get("leftPupil") or le, dtype=np.float32), axis=0)
        rp = np.mean(np.array(marks.get("rightPupil") or re, dtype=np.float32), axis=0)

        eye_vec = rp - lp
        iod = max(1.0, float(np.linalg.norm(eye_vec)))
        roll = math.atan2(float(eye_vec[1]), float(eye_vec[0]))
        c, s = math.cos(-roll), math.sin(-roll)
        rot_m = np.array([[c, -s], [s, c]], dtype=np.float32)
        mid = 0.5 * (lp + rp)
        rot = lambda pts: (pts - mid) @ rot_m.T

        le_r, re_r = rot(le), rot(re)
        leb_r, reb_r = rot(leb), rot(reb)

        def _geom(pts: np.ndarray, brow: np.ndarray, is_left: bool) -> Dict[str, float]:
            idx_min, idx_max = int(np.argmin(pts[:, 0])), int(np.argmax(pts[:, 0]))
            outer = pts[idx_min] if is_left else pts[idx_max]
            inner = pts[idx_max] if is_left else pts[idx_min]
            w = max(1.0, float(np.linalg.norm(outer - inner)))
            h = float(pts[:, 1].max() - pts[:, 1].min())
            dx_abs = max(1e-4, abs(float(outer[0] - inner[0])))
            tilt_deg = math.degrees(math.atan2(float(inner[1] - outer[1]), dx_abs))
            chord_y = 0.5 * (float(inner[1]) + float(outer[1]))
            upper_arch = max(0.0, chord_y - float(pts[:, 1].min())) / w
            lower_drop = max(0.0, float(pts[:, 1].max()) - chord_y) / w
            brow_gap = (float(np.mean(pts[:, 1])) - float(np.mean(brow[:, 1]))) / w
            return {
                "ear": h / w,
                "tilt": tilt_deg,
                "upper_arch": upper_arch,
                "lower_drop": lower_drop,
                "brow_gap": brow_gap,
                "inner_x": float(inner[0]),
            }

        gl = _geom(le_r, leb_r, True)
        gr = _geom(re_r, reb_r, False)

        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        l_iris, l_scl, l_cg, l_cp = self._measure_single_eye_pixels(gray, le, True)
        r_iris, r_scl, r_cg, r_cp = self._measure_single_eye_pixels(gray, re, False)

        return OcularAnatomyProfile(
            avg_ear=round(0.5 * (gl["ear"] + gr["ear"]), 4),
            avg_canthal_tilt_deg=round(0.5 * (gl["tilt"] + gr["tilt"]), 2),
            avg_upper_arch_ratio=round(0.5 * (gl["upper_arch"] + gr["upper_arch"]), 4),
            avg_lower_drop_ratio=round(0.5 * (gl["lower_drop"] + gr["lower_drop"]), 4),
            avg_brow_gap_ratio=round(0.5 * (gl["brow_gap"] + gr["brow_gap"]), 4),
            inter_canthal_to_iod=round(abs(gr["inner_x"] - gl["inner_x"]) / iod, 4),
            avg_iris_to_eye_width=round(0.5 * (l_iris + r_iris), 4),
            avg_sclera_visible_ratio=round(0.5 * (l_scl + r_scl), 4),
            avg_crease_gap_ratio=round(0.5 * (l_cg + r_cg), 4),
            crease_prominence=round(0.5 * (l_cp + r_cp), 4),
            left_iris_to_eye_width=round(l_iris, 4),
            right_iris_to_eye_width=round(r_iris, 4),
        )

    def extract_aligned_eye_strip(
        self,
        bgr_img: np.ndarray,
        marks: Optional[Dict[str, object]] = None,
        out_width: int = 960,
    ) -> Optional[np.ndarray]:
        """Extract a roll-corrected, pupil-aligned 400% ocular verification strip (eyes + eyelids + eyebrows)."""
        if marks is None:
            pil = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
            marks = self.vision.landmarks(pil)
        if not marks or not marks.get("leftEye") or not marks.get("rightEye"):
            return None

        le = np.array(marks["leftEye"], dtype=np.float32)
        re = np.array(marks["rightEye"], dtype=np.float32)
        lp = np.mean(np.array(marks.get("leftPupil") or le, dtype=np.float32), axis=0)
        rp = np.mean(np.array(marks.get("rightPupil") or re, dtype=np.float32), axis=0)

        iod = max(1.0, float(np.linalg.norm(rp - lp)))
        angle_deg = math.degrees(math.atan2(float(rp[1] - lp[1]), float(rp[0] - lp[0])))
        mid = (0.5 * float(lp[0] + rp[0]), 0.5 * float(lp[1] + rp[1]))

        # Canonical eye strip: IOD spans 42% of out_width, eye center sits at 62% of out_height
        out_h = int(round(out_width * 0.44))
        scale = (0.42 * out_width) / iod
        M = cv2.getRotationMatrix2D(mid, angle_deg, scale)
        M[0, 2] += (0.50 * out_width) - mid[0]
        M[1, 2] += (0.62 * out_h) - mid[1]
        return cv2.warpAffine(
            bgr_img,
            M,
            (out_width, out_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def compare(
        self,
        ref_bgr: np.ndarray,
        cand_bgr: np.ndarray,
        ref_marks: Optional[Dict[str, object]] = None,
        cand_marks: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Compare ocular anatomy & pupil-aligned eyelid/iris edge NCC between Gen-0 and candidate."""
        r_prof = self.analyze(ref_bgr, marks=ref_marks)
        c_prof = self.analyze(cand_bgr, marks=cand_marks)
        if r_prof is None or c_prof is None:
            return {"valid": False, "passed": False, "ocularDriftScore": 99.0, "eyelidEdgeNcc": 0.0, "issues": ["눈매 랜드마크 검출 실패"]}

        ear_diff_pct = 100.0 * (c_prof.avg_ear - r_prof.avg_ear) / max(1e-4, r_prof.avg_ear)
        tilt_diff_deg = c_prof.avg_canthal_tilt_deg - r_prof.avg_canthal_tilt_deg
        iris_diff_pct = 100.0 * (c_prof.avg_iris_to_eye_width - r_prof.avg_iris_to_eye_width) / max(1e-4, r_prof.avg_iris_to_eye_width)
        sclera_diff_pct = 100.0 * (c_prof.avg_sclera_visible_ratio - r_prof.avg_sclera_visible_ratio) / max(1e-4, r_prof.avg_sclera_visible_ratio)
        crease_prom_diff = c_prof.crease_prominence - r_prof.crease_prominence
        brow_gap_diff_pct = 100.0 * (c_prof.avg_brow_gap_ratio - r_prof.avg_brow_gap_ratio) / max(1e-4, r_prof.avg_brow_gap_ratio)

        # Pupil-aligned eyelid + iris + sclera edge correlation (NCC)
        eyelid_ncc = 0.0
        s_ref = self.extract_aligned_eye_strip(ref_bgr, marks=ref_marks, out_width=480)
        s_cand = self.extract_aligned_eye_strip(cand_bgr, marks=cand_marks, out_width=480)
        if s_ref is not None and s_cand is not None:
            # Focus strictly on eye+lid band (rows 32%..88%) excluding upper forehead
            h_s = s_ref.shape[0]
            y1, y2 = int(round(h_s * 0.32)), int(round(h_s * 0.88))
            gr = cv2.GaussianBlur(cv2.cvtColor(s_ref[y1:y2], cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0)
            gc = cv2.GaussianBlur(cv2.cvtColor(s_cand[y1:y2], cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0)
            er = cv2.Sobel(gr, cv2.CV_32F, 0, 1, ksize=3)
            ec = cv2.Sobel(gc, cv2.CV_32F, 0, 1, ksize=3)
            er -= er.mean()
            ec -= ec.mean()
            denom = float(np.linalg.norm(er) * np.linalg.norm(ec)) + 1e-6
            eyelid_ncc = float(np.sum(er * ec) / denom)

        issues = []
        if abs(ear_diff_pct) > 16.0:
            issues.append(f"눈 세로 개방도 편차 {ear_diff_pct:+.1f}% (허용 ±16%)")
        if abs(tilt_diff_deg) > 4.5:
            issues.append(f"눈꼬리 각도 편차 {tilt_diff_deg:+.1f}° (허용 ±4.5°)")
        if iris_diff_pct > 20.0:
            issues.append(f"홍채 과대 확대(써클렌즈화) {iris_diff_pct:+.1f}% (허용 +20%)")
        if sclera_diff_pct < -14.5:
            issues.append(f"흰자 노출 감소(눈매 답답함) {sclera_diff_pct:+.1f}% (허용 -14.5%)")
        crease_gap_diff = c_prof.avg_crease_gap_ratio - r_prof.avg_crease_gap_ratio
        if (
            r_prof.crease_category() in ("tight_inline_or_monolid", "subtle_inline_double")
            and ((crease_gap_diff > 0.15 and crease_prom_diff > 0.18) or (c_prof.avg_crease_gap_ratio > 0.42 and crease_prom_diff > 0.32))
        ):
            issues.append(f"쌍꺼풀 라인 과장(두꺼운 아치 선 생성, gap {crease_gap_diff:+.2f}, prom {crease_prom_diff:+.2f})")
        if r_prof.crease_category() == "tight_inline_or_monolid" and c_prof.crease_category() == "prominent_outline_double":
            issues.append("원본 속쌍/얇은 인라인 눈매가 두꺼운 겉쌍 아치로 변형됨")

        # Weighted ocular drift score (lower is better; integrates sclera, crease prominence & NCC)
        drift_score = (
            0.20 * abs(ear_diff_pct)
            + 0.20 * abs(iris_diff_pct)
            + 0.20 * abs(sclera_diff_pct)
            + 0.15 * max(0.0, crease_prom_diff * 50.0)
            + 0.10 * (abs(tilt_diff_deg) * 2.5)
            + 0.15 * max(0.0, (0.65 - eyelid_ncc) * 45.0)
        )
        return {
            "valid": True,
            "passed": len(issues) == 0,
            "ocularDriftScore": round(drift_score, 2),
            "eyelidEdgeNcc": round(eyelid_ncc, 4),
            "refCategory": r_prof.crease_category(),
            "candCategory": c_prof.crease_category(),
            "diffs": {
                "earDiffPct": round(ear_diff_pct, 1),
                "canthalTiltDiffDeg": round(tilt_diff_deg, 2),
                "irisWidthDiffPct": round(iris_diff_pct, 1),
                "scleraDiffPct": round(sclera_diff_pct, 1),
                "creasePromDiff": round(crease_prom_diff, 3),
                "browGapDiffPct": round(brow_gap_diff_pct, 1),
            },
            "refProfile": r_prof,
            "candProfile": c_prof,
            "issues": issues,
        }

    def save_400pct_eye_comparison_strip(
        self,
        ref_bgr: np.ndarray,
        cand_bgr: np.ndarray,
        output_path: str,
    ) -> Optional[str]:
        """Save a pupil-aligned 400% top/bottom ocular comparison strip (Gen-0 vs Generated) for mandatory visual inspection."""
        s_ref = self.extract_aligned_eye_strip(ref_bgr, out_width=1200)
        s_cand = self.extract_aligned_eye_strip(cand_bgr, out_width=1200)
        if s_ref is None or s_cand is None:
            return None
        cv2.putText(s_ref, "TOP: Gen-0 ORIGINAL EYES (GROUND TRUTH)", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2)
        cv2.putText(s_cand, "BOTTOM: GENERATED 4K EYES", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2)
        sep = np.full((14, 1200, 3), 245, dtype=np.uint8)
        strip = np.vstack([s_ref, sep, s_cand])
        cv2.imwrite(str(output_path), strip, [int(cv2.IMWRITE_JPEG_QUALITY), 97])
        return str(output_path)
