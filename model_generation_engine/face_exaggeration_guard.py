"""Universal Zero-Hardcoding 7-Axis Anti-Exaggeration Face Guard (`얼굴 과장 전면 차단 게이트`).

Quantitatively detects and blocks EVERY facial exaggeration / beautification / caricature drift
against the untouched `Gen-0` original photograph:
  1. Eyes (`눈 과장`): enlarged palpebral opening, enlarged circle-lens iris, sclera shrink, exaggerated crease arch, sharp cat-eye tilt
  2. Eyebrows (`눈썹 과장`): thickened painted brows or darkened brow intensity relative to forehead skin
  3. Nose (`코 과장`): pinched narrow rhinoplasty nose, bloated nose width, or elongated sharp bridge
  4. Lips (`입술 과장`): filler-plumped lip height, widened mouth, or exaggerated heavy saturated editorial lipstick
  5. Jawline & Cheeks (`턱·볼 과장`): pinched V-line jaw, hollowed gaunt cheekbones, sharpened gonion angle, or bloated puffiness
  6. Cheek Contouring / Makeup (`윤곽·메이크업 과장`): harsh cheek shadow-contouring inflation vs Gen-0
  7. Overall Exaggeration Index (`0.0` = documentary twin, `> 10.0` = caricatured/exaggerated)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from .apple_vision import AppleVisionAnalyzer
from .ocular_anatomy import OcularAnatomyAnalyzer


@dataclass(frozen=True)
class FacialOrganExaggerationMetrics:
    """Scale-invariant per-organ proportions & natural luminance/chroma contrast from a real face image."""

    eye_width_to_iod: float
    lip_height_to_eye_chin: float
    lip_width_to_iod: float
    lip_chroma_above_skin: float  # Lab Chroma(lips) - Lab Chroma(cheek skin): catches heavy lipstick inflation!
    nose_width_to_iod: float
    nose_len_to_eye_chin: float
    brow_thickness_to_iod: float
    brow_darkness_below_skin: float  # Lab L(forehead skin) - Lab L(eyebrow): catches darkened/painted editorial brows!
    cheek_contour_std: float  # Normalized Lab L std inside mid-cheek: catches harsh hollowed cheekbone sculpting!


class FaceExaggerationGuard:
    """7-Axis Anti-Exaggeration / Zero-Caricature Auditor comparing any generated face against untouched Gen-0."""

    ZERO_EXAGGERATION_LAW_CLAUSE: str = (
        "ABSOLUTE ZERO-FACIAL-EXAGGERATION & ZERO-CARICATURE LAW (얼굴 과장·미화·캐리커처 절대 금지): "
        "Reproduce the subject's real face from Image 1 & Image 2 with 100% calm, unadorned documentary-camera fidelity—"
        "NEVER beautify, idealize, glam-up, sculpt, sharpen, or caricature ANY facial organ; "
        "NEVER enlarge the eyes or irises or draw a thicker eyelid crease; "
        "NEVER darken, thicken, or sharply arch the eyebrows beyond Image 1; "
        "NEVER pinch, narrow, or lengthen the nose; "
        "NEVER plump, enlarge, over-gloss, or darken the lips with editorial lipstick; "
        "NEVER hollow the cheekbones, add contour shading, or taper the mandibular contour into a pinched triangle; "
        "keep the exact calm natural real-human face of Image 1 untouched by fashion-beauty exaggeration."
    )

    def __init__(self, vision: AppleVisionAnalyzer) -> None:
        self.vision = vision
        self.ocular = OcularAnatomyAnalyzer(vision)

    @staticmethod
    def _poly_lab_stats(lab_img: np.ndarray, pts: np.ndarray) -> Optional[tuple[float, float, float]]:
        h, w = lab_img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
        px = lab_img[mask > 0]
        if px.size < 12:
            return None
        L = px[:, 0].astype(np.float32)
        a = px[:, 1].astype(np.float32) - 128.0
        b = px[:, 2].astype(np.float32) - 128.0
        chroma = np.sqrt(a * a + b * b)
        return float(np.median(L)), float(np.median(chroma)), float(np.std(L))

    def extract_metrics(
        self,
        bgr_img: np.ndarray,
        marks: Optional[Dict[str, object]] = None,
    ) -> Optional[FacialOrganExaggerationMetrics]:
        if marks is None:
            pil = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
            marks = self.vision.landmarks(pil)
        if not marks:
            return None
        for req in ("contour", "leftEye", "rightEye", "leftEyebrow", "rightEyebrow", "nose", "outerLips"):
            if not marks.get(req) or len(marks[req]) < 3:
                return None

        c = np.array(marks["contour"], dtype=np.float32)
        le = np.array(marks["leftEye"], dtype=np.float32)
        re = np.array(marks["rightEye"], dtype=np.float32)
        leb = np.array(marks["leftEyebrow"], dtype=np.float32)
        reb = np.array(marks["rightEyebrow"], dtype=np.float32)
        nose = np.array(marks["nose"], dtype=np.float32)
        lips = np.array(marks["outerLips"], dtype=np.float32)

        lp = np.mean(np.array(marks.get("leftPupil") or le, dtype=np.float32), axis=0)
        rp = np.mean(np.array(marks.get("rightPupil") or re, dtype=np.float32), axis=0)
        iod = max(1.0, float(np.linalg.norm(rp - lp)))
        eye_mid_y = 0.5 * float(lp[1] + rp[1])
        chin_y = float(c[:, 1].max())
        eye_chin = max(1.0, chin_y - eye_mid_y)

        # Geometric organ proportions
        le_w = float(le[:, 0].max() - le[:, 0].min())
        re_w = float(re[:, 0].max() - re[:, 0].min())
        eye_w_to_iod = 0.5 * (le_w + re_w) / iod

        lip_h = float(lips[:, 1].max() - lips[:, 1].min())
        lip_w = float(lips[:, 0].max() - lips[:, 0].min())
        lip_h_to_ec = lip_h / eye_chin
        lip_w_to_iod = lip_w / iod

        nose_w = float(nose[:, 0].max() - nose[:, 0].min())
        nose_len = max(1.0, float(nose[:, 1].max()) - eye_mid_y)
        nose_w_to_iod = nose_w / iod
        nose_len_to_ec = nose_len / eye_chin

        # Eyebrow vertical thickness normalized by IOD
        lb_h = float(leb[:, 1].max() - leb[:, 1].min())
        rb_h = float(reb[:, 1].max() - reb[:, 1].min())
        brow_thick_to_iod = 0.5 * (lb_h + rb_h) / iod

        # Lab skin/lip/brow/cheek contrast (lighting-relative: always computed vs the SAME face's own skin!)
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        h, w = bgr_img.shape[:2]

        # Sample clean cheek-skin box just below left/right eyes & outside nose
        cx_l = int(np.clip(lp[0] - 0.12 * iod, 4, w - 5))
        cx_r = int(np.clip(rp[0] + 0.12 * iod, 4, w - 5))
        cy_ck = int(np.clip(eye_mid_y + 0.38 * eye_chin, 4, h - 5))
        r_ck = max(4, int(round(0.18 * iod)))
        cheek_patches = []
        for cx in (cx_l, cx_r):
            p_ck = lab[max(0, cy_ck - r_ck) : min(h, cy_ck + r_ck), max(0, cx - r_ck) : min(w, cx + r_ck)]
            if p_ck.size >= 12:
                cheek_patches.append(p_ck.reshape(-1, 3))
        if cheek_patches:
            ck_all = np.vstack(cheek_patches)
            skin_L = float(np.median(ck_all[:, 0]))
            skin_a = ck_all[:, 1].astype(np.float32) - 128.0
            skin_b = ck_all[:, 2].astype(np.float32) - 128.0
            skin_chroma = float(np.median(np.sqrt(skin_a * skin_a + skin_b * skin_b)))
            cheek_std = float(np.std(ck_all[:, 0].astype(np.float32))) / max(10.0, skin_L)
        else:
            skin_L, skin_chroma, cheek_std = 160.0, 14.0, 0.05

        lip_stats = self._poly_lab_stats(lab, lips)
        lip_chroma_above = (lip_stats[1] - skin_chroma) if lip_stats else 8.0

        # Brow darkness relative to surrounding forehead skin
        brow_Ls = []
        for bpts in (leb, reb):
            bx1 = max(0, int(np.floor(float(bpts[:, 0].min()))))
            bx2 = min(w, int(np.ceil(float(bpts[:, 0].max()))))
            by1 = max(0, int(np.floor(float(bpts[:, 1].min())) - 0.04 * iod))
            by2 = min(h, int(np.ceil(float(bpts[:, 1].max())) + 0.04 * iod))
            bp = lab[by1:by2, bx1:bx2, 0]
            if bp.size >= 12:
                brow_Ls.append(float(np.percentile(bp, 25)))
        brow_L = float(np.mean(brow_Ls)) if brow_Ls else (skin_L - 30.0)
        brow_darkness_below = max(0.0, skin_L - brow_L)

        return FacialOrganExaggerationMetrics(
            eye_width_to_iod=round(eye_w_to_iod, 4),
            lip_height_to_eye_chin=round(lip_h_to_ec, 4),
            lip_width_to_iod=round(lip_w_to_iod, 4),
            lip_chroma_above_skin=round(lip_chroma_above, 2),
            nose_width_to_iod=round(nose_w_to_iod, 4),
            nose_len_to_eye_chin=round(nose_len_to_ec, 4),
            brow_thickness_to_iod=round(brow_thick_to_iod, 4),
            brow_darkness_below_skin=round(brow_darkness_below, 2),
            cheek_contour_std=round(cheek_std, 4),
        )

    def audit(
        self,
        ref_gen0_bgr: np.ndarray,
        cand_bgr: np.ndarray,
        ref_marks: Optional[Dict[str, object]] = None,
        cand_marks: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Audit all non-jawline facial exaggeration axes (eyes, brows, nose, lips, makeup/contouring) vs Gen-0."""
        rm = self.extract_metrics(ref_gen0_bgr, marks=ref_marks)
        cm = self.extract_metrics(cand_bgr, marks=cand_marks)
        if rm is None or cm is None:
            return {"valid": False, "passed": True, "exaggerationIndex": 0.0, "diffs": {}, "issues": []}

        pct = lambda a, b: 100.0 * (b - a) / max(1e-4, abs(a))
        eye_w_pct = pct(rm.eye_width_to_iod, cm.eye_width_to_iod)
        lip_h_pct = pct(rm.lip_height_to_eye_chin, cm.lip_height_to_eye_chin)
        lip_w_pct = pct(rm.lip_width_to_iod, cm.lip_width_to_iod)
        lip_chroma_diff = cm.lip_chroma_above_skin - rm.lip_chroma_above_skin
        nose_w_pct = pct(rm.nose_width_to_iod, cm.nose_width_to_iod)
        nose_len_pct = pct(rm.nose_len_to_eye_chin, cm.nose_len_to_eye_chin)
        brow_thick_pct = pct(rm.brow_thickness_to_iod, cm.brow_thickness_to_iod)
        brow_dark_diff = cm.brow_darkness_below_skin - rm.brow_darkness_below_skin
        cheek_contour_ratio = cm.cheek_contour_std / max(0.015, rm.cheek_contour_std)

        issues: List[str] = []
        if abs(eye_w_pct) > 10.5:
            issues.append(f"눈 가로폭 과장/변형 ({eye_w_pct:+.1f}%, 허용 ±10.5%)")
        if lip_h_pct > 20.0:
            issues.append(f"입술 과대/필러화 과장 ({lip_h_pct:+.1f}%, 허용 +20%)")
        elif lip_h_pct < -22.0:
            issues.append(f"입술 과축소 ({lip_h_pct:+.1f}%, 허용 -22%)")
        if abs(lip_w_pct) > 12.5:
            issues.append(f"입술 가로폭 과장/변형 ({lip_w_pct:+.1f}%, 허용 ±12.5%)")
        if lip_chroma_diff > 14.0:
            issues.append(f"립스틱/입술 채도 과장 (+{lip_chroma_diff:.1f} Lab chroma, 허용 +14)")
        if abs(nose_w_pct) > 13.5:
            issues.append(f"코 폭 과장/성형 왜곡 ({nose_w_pct:+.1f}%, 허용 ±13.5%)")
        if abs(nose_len_pct) > 11.5:
            issues.append(f"코 길이 과장/변형 ({nose_len_pct:+.1f}%, 허용 ±11.5%)")
        if brow_dark_diff > 24.0:
            issues.append(f"눈썹 과도 진하게/그린듯 과장 (+{brow_dark_diff:.1f} L contrast, 허용 +24)")

        exag_idx = (
            0.22 * abs(eye_w_pct)
            + 0.20 * abs(lip_h_pct)
            + 0.14 * abs(lip_w_pct)
            + 0.16 * abs(nose_w_pct)
            + 0.14 * abs(nose_len_pct)
            + 0.08 * max(0.0, lip_chroma_diff)
            + 0.06 * max(0.0, brow_dark_diff * 0.5)
        )
        return {
            "valid": True,
            "passed": len(issues) == 0,
            "exaggerationIndex": round(exag_idx, 2),
            "diffs": {
                "eyeWidthPct": round(eye_w_pct, 1),
                "lipHeightPct": round(lip_h_pct, 1),
                "lipWidthPct": round(lip_w_pct, 1),
                "lipChromaDiff": round(lip_chroma_diff, 1),
                "noseWidthPct": round(nose_w_pct, 1),
                "noseLenPct": round(nose_len_pct, 1),
                "browThickPct": round(brow_thick_pct, 1),
                "browDarkDiff": round(brow_dark_diff, 1),
                "cheekContourRatio": round(cheek_contour_ratio, 2),
            },
            "issues": issues,
        }

    def measure_pupil_aligned_silhouette(
        self,
        ref_gen0_bgr: np.ndarray,
        cand_bgr: np.ndarray,
        output_overlay_path: Optional[str] = None,
        max_point_err_px: float = 11.5,
        min_contour_iou: float = 0.92,
    ) -> Dict[str, object]:
        """Align Gen-0 and Candidate strictly by `(leftPupil, rightPupil)` at IOD=180px & audit exact 17-pt outer silhouette."""
        import math

        def _align(bgr: np.ndarray, size: int = 800):
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            m = self.vision.landmarks(pil)
            if not m or not m.get("contour") or not m.get("leftEye") or not m.get("rightEye"):
                return None, None
            le = np.array(m["leftEye"], dtype=np.float32)
            re = np.array(m["rightEye"], dtype=np.float32)
            lp = np.mean(np.array(m.get("leftPupil") or le, dtype=np.float32), axis=0)
            rp = np.mean(np.array(m.get("rightPupil") or re, dtype=np.float32), axis=0)
            iod = max(1.0, float(np.linalg.norm(rp - lp)))
            ang = math.degrees(math.atan2(float(rp[1] - lp[1]), float(rp[0] - lp[0])))
            mid = (0.5 * float(lp[0] + rp[0]), 0.5 * float(lp[1] + rp[1]))
            scale = 180.0 / iod
            M = cv2.getRotationMatrix2D(mid, ang, scale)
            M[0, 2] += 400.0 - mid[0]
            M[1, 2] += 340.0 - mid[1]
            warped = cv2.warpAffine(bgr, M, (size, size), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
            c = np.array(m["contour"], dtype=np.float32)
            ones = np.ones((len(c), 1), dtype=np.float32)
            c_xf = np.hstack([c, ones]) @ M.T
            return warped, c_xf

        w0, c0 = _align(ref_gen0_bgr)
        wc, cc = _align(cand_bgr)
        if w0 is None or wc is None or c0 is None or cc is None:
            return {"valid": False, "passed": True, "contourIoU": 1.0, "maxContourPointErrPx": 0.0, "width5LevelDiffPct": {}, "issues": []}

        if len(c0) != len(cc):
            t_old = np.linspace(0.0, 1.0, len(cc))
            t_new = np.linspace(0.0, 1.0, len(c0))
            cc = np.stack([np.interp(t_new, t_old, cc[:, 0]), np.interp(t_new, t_old, cc[:, 1])], axis=-1).astype(np.float32)

        pt_errs = np.linalg.norm(cc - c0, axis=1)
        max_pt_err = float(np.max(pt_errs))
        mean_pt_err = float(np.mean(pt_errs))

        m0 = np.zeros((800, 800), dtype=np.uint8)
        mc = np.zeros((800, 800), dtype=np.uint8)
        cv2.fillPoly(m0, [np.round(c0).astype(np.int32)], 255)
        cv2.fillPoly(mc, [np.round(cc).astype(np.int32)], 255)
        union = float(np.sum((m0 > 0) | (mc > 0)))
        iou = float(np.sum((m0 > 0) & (mc > 0))) / max(1.0, union)

        n = len(c0)
        w5: Dict[str, float] = {}
        for lbl, frac_i in (("temple", 0.0), ("cheek", 0.125), ("midJaw", 0.25), ("lowJaw", 0.375), ("chin", 0.4375)):
            i = int(round(frac_i * (n - 1)))
            j = (n - 1) - i
            wr = max(1.0, abs(float(c0[j, 0] - c0[i, 0])))
            wg = abs(float(cc[j, 0] - cc[i, 0]))
            w5[lbl] = round(100.0 * (wg - wr) / wr, 1)

        issues: List[str] = []
        if iou < min_contour_iou:
            issues.append(f"1:1 외곽선 겹침률(IoU) 부족 ({iou:.3f} < {min_contour_iou:.2f})")
        if max_pt_err > max_point_err_px:
            issues.append(f"외곽선 최대 점별 변형 초과 ({max_pt_err:.1f}px > {max_point_err_px:.1f}px @ IOD=180)")
        worst_w5 = max((abs(v) for v in w5.values()), default=0.0)
        if worst_w5 > 6.5:
            issues.append(f"5단계 가로폭 외곽 오차 초과 (최대 {worst_w5:.1f}%, 허용 ±6.5%)")

        saved_ov: Optional[str] = None
        if output_overlay_path:
            ov = cv2.addWeighted(w0, 0.50, wc, 0.50, 0)
            cv2.polylines(ov, [np.round(c0).astype(np.int32)], False, (0, 255, 0), 3)
            cv2.polylines(ov, [np.round(cc).astype(np.int32)], False, (0, 0, 255), 3)
            cv2.putText(
                ov,
                f"GREEN=Gen0 RED=Gen IoU={iou:.3f} MaxErr={max_pt_err:.1f}px",
                (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (255, 255, 0),
                2,
            )
            side = np.hstack([w0, ov, wc])
            cv2.imwrite(str(output_overlay_path), side, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved_ov = str(output_overlay_path)

        return {
            "valid": True,
            "passed": len(issues) == 0,
            "contourIoU": round(iou, 4),
            "maxContourPointErrPx": round(max_pt_err, 2),
            "meanContourPointErrPx": round(mean_pt_err, 2),
            "width5LevelDiffPct": w5,
            "overlayPath": saved_ov,
            "issues": issues,
        }
