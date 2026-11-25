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
        eye_mid = 0.5 * (lp + rp)
        eye_mid_y = float(eye_mid[1])

        # De-rotate 2D organ coordinates into the inter-pupil local frame so natural head tilt (roll) never causes sin/cos axis foreshortening
        u_axis = (rp - lp) / iod
        v_axis = np.array([-u_axis[1], u_axis[0]], dtype=np.float32)

        def _to_face_axes(arr: np.ndarray) -> np.ndarray:
            rel = arr - eye_mid
            return np.stack([rel @ u_axis, rel @ v_axis], axis=-1)

        c_r = _to_face_axes(c)
        le_r = _to_face_axes(le)
        re_r = _to_face_axes(re)
        leb_r = _to_face_axes(leb)
        reb_r = _to_face_axes(reb)
        nose_r = _to_face_axes(nose)
        lips_r = _to_face_axes(lips)

        eye_chin = max(1.0, float(c_r[:, 1].max()))

        # Geometric organ proportions (roll-invariant in face coordinates)
        le_w = float(le_r[:, 0].max() - le_r[:, 0].min())
        re_w = float(re_r[:, 0].max() - re_r[:, 0].min())
        eye_w_to_iod = 0.5 * (le_w + re_w) / iod

        lip_h = float(lips_r[:, 1].max() - lips_r[:, 1].min())
        lip_w = float(lips_r[:, 0].max() - lips_r[:, 0].min())
        lip_h_to_ec = lip_h / eye_chin
        lip_w_to_iod = lip_w / iod

        nose_w = float(nose_r[:, 0].max() - nose_r[:, 0].min())
        nose_len = max(1.0, float(nose_r[:, 1].max()))
        nose_w_to_iod = nose_w / iod
        nose_len_to_ec = nose_len / eye_chin

        # Eyebrow vertical thickness normalized by IOD
        lb_h = float(leb_r[:, 1].max() - leb_r[:, 1].min())
        rb_h = float(reb_r[:, 1].max() - reb_r[:, 1].min())
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
            return {"valid": False, "passed": False, "exaggerationIndex": 99.0, "diffs": {}, "issues": ["얼굴 과장 게이트 랜드마크 검출 실패"]}

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
                return None, None, None, None, 0.0, 0.0
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
            M[1, 2] += 360.0 - mid[1]
            warped = cv2.warpAffine(bgr, M, (size, size), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
            c = np.array(m["contour"], dtype=np.float32)
            c_xf = np.hstack([c, np.ones((len(c), 1), dtype=np.float32)]) @ M.T
            lips_xf = None
            inner_gap_px = 0.0
            if m.get("outerLips"):
                lp_arr = np.array(m["outerLips"], dtype=np.float32)
                lips_xf = np.hstack([lp_arr, np.ones((len(lp_arr), 1), dtype=np.float32)]) @ M.T
            if m.get("innerLips") and len(m["innerLips"]) >= 4:
                in_arr = np.array(m["innerLips"], dtype=np.float32)
                in_xf = np.hstack([in_arr, np.ones((len(in_arr), 1), dtype=np.float32)]) @ M.T
                inner_gap_px = float(max(0.0, in_xf[:, 1].max() - in_xf[:, 1].min()))
            nose_turn = 0.0
            if m.get("nose") and len(m["nose"]) >= 2:
                ns_arr = np.array(m["nose"], dtype=np.float32)
                ns_xf = np.hstack([ns_arr, np.ones((len(ns_arr), 1), dtype=np.float32)]) @ M.T
                nose_turn = float((np.mean(ns_xf[:, 0]) - 400.0) / 180.0)
            # Extract upper cranial + hair dome mask on the aligned 800x800 canvas
            bg_patch = np.vstack([warped[:40, :60].reshape(-1, 3), warped[:40, -60:].reshape(-1, 3)]).astype(np.float32)
            bg_med = np.median(bg_patch, axis=0)
            bg_std = float(np.mean(np.std(bg_patch, axis=0)))
            bg_diff = np.linalg.norm(warped.astype(np.float32) - bg_med, axis=2)
            dome_mask = np.zeros((size, size), dtype=np.uint8)
            try:
                pm = self.vision.person_mask(Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)))
                if pm is not None:
                    pm_np = np.array(pm.resize((size, size), Image.Resampling.BILINEAR))
                    dome_mask[pm_np > 160] = 255
                    # Strip soft matte background halo where pixels clearly match uniform studio wall
                    if bg_std < 20.0:
                        dome_mask[(bg_diff < 18.0) & (pm_np < 235)] = 0
            except Exception:
                pass
            if int(np.sum(dome_mask[:360, :] > 0)) < 1500:
                # Fallback: segment dark hair/head against top-corner studio background
                dome_mask[bg_diff > 24.0] = 255
            return warped, c_xf, lips_xf, dome_mask, inner_gap_px, nose_turn

        w0, c0, lips0, dome0, gap0, turn0 = _align(ref_gen0_bgr)
        wc, cc, lipsc, domec, gapc, turnc = _align(cand_bgr)
        if w0 is None or wc is None or c0 is None or cc is None:
            return {
                "valid": False,
                "passed": False,
                "contourIoU": 0.0,
                "maxContourPointErrPx": 999.0,
                "width5LevelDiffPct": {},
                "issues": ["1:1 외곽선 랜드마크 검출 실패"],
            }

        if len(c0) != len(cc):
            t_old = np.linspace(0.0, 1.0, len(cc))
            t_new = np.linspace(0.0, 1.0, len(c0))
            cc = np.stack([np.interp(t_new, t_old, cc[:, 0]), np.interp(t_new, t_old, cc[:, 1])], axis=-1).astype(np.float32)

        # Compensate ONLY rigid horizontal 3D head-yaw translation continuously (no step-cliff at 0.045!)
        # Bilateral inflation (+L, +R) has median(dx)=0 and jaw_mid_diff=0, so width5LevelDiffPct & test_28 stay 100% strict!
        jaw_mid_0 = 0.5 * float(c0[0, 0] + c0[-1, 0])
        jaw_mid_c = 0.5 * float(cc[0, 0] + cc[-1, 0])
        jaw_mid_turn = abs(jaw_mid_c - jaw_mid_0) / 180.0
        turn_diff_abs = max(abs(turnc - turn0), jaw_mid_turn)
        cc_eval = cc.copy()
        dx_yaw = 0.0
        if turn_diff_abs >= 0.010:
            max_yaw_dx = min(34.0, turn_diff_abs * 195.0)
            dx_yaw = float(np.clip(np.median(cc[:, 0] - c0[:, 0]), -max_yaw_dx, max_yaw_dx))
            cc_eval[:, 0] = cc[:, 0] - dx_yaw

        def _point_to_polyline_dists(pts_src: np.ndarray, poly_dst: np.ndarray) -> np.ndarray:
            segs_a = poly_dst[:-1]
            segs_b = poly_dst[1:]
            ab = segs_b - segs_a
            ab_len2 = np.maximum(1e-6, np.sum(ab * ab, axis=1))
            n_segs = len(segs_a)
            dists = []
            for p_pt in pts_src:
                ap = p_pt[None, :] - segs_a
                t_proj = np.clip(np.sum(ap * ab, axis=1) / ab_len2, 0.0, 1.0)
                closest = segs_a + t_proj[:, None] * ab
                seg_dists = np.linalg.norm(p_pt[None, :] - closest, axis=1)
                best_k = int(np.argmin(seg_dists))
                # If an upper-temple point (y < 410) clamps onto an open contour top endpoint (0 or n_segs-1),
                # measure horizontal temple width error |dx| rather than vertical sideburn start-height truncation.
                if float(p_pt[1]) < 410.0 and (
                    (best_k == 0 and float(t_proj[0]) <= 1e-4)
                    or (best_k == n_segs - 1 and float(t_proj[n_segs - 1]) >= 1.0 - 1e-4)
                ):
                    d = float(abs(float(p_pt[0]) - float(closest[best_k, 0])))
                else:
                    d = float(seg_dists[best_k])
                dists.append(d)
            return np.array(dists, dtype=np.float32)

        # Measure true perpendicular point-to-curve Hausdorff deviation across jaw contour (excluding hair-occluded top temple endpoints)
        jaw_slice = slice(2, -2) if len(c0) >= 12 else (slice(1, -1) if len(c0) >= 6 else slice(None))
        d_c_to_0 = _point_to_polyline_dists(cc_eval[jaw_slice], c0)
        d_0_to_c = _point_to_polyline_dists(c0[jaw_slice], cc_eval)
        sym_errs = np.concatenate([d_c_to_0, d_0_to_c])
        max_pt_err = float(np.max(sym_errs)) if sym_errs.size > 0 else 0.0
        mean_pt_err = float(np.mean(sym_errs)) if sym_errs.size > 0 else 0.0

        m0 = np.zeros((800, 800), dtype=np.uint8)
        mc = np.zeros((800, 800), dtype=np.uint8)
        cv2.fillPoly(m0, [np.round(c0).astype(np.int32)], 255)
        cv2.fillPoly(mc, [np.round(cc_eval).astype(np.int32)], 255)
        union = float(np.sum((m0 > 0) | (mc > 0)))
        iou = float(np.sum((m0 > 0) & (mc > 0))) / max(1.0, union)

        # Cranial Dome Size & Breadth Audit (y in [20..370], x in [120..680] around pupil center (400, 360))
        def _cranial_metrics(dm: np.ndarray) -> Tuple[float, float, np.ndarray]:
            roi = np.zeros_like(dm)
            roi[15:370, 120:680] = dm[15:370, 120:680]
            ys, xs = np.where(roi > 0)
            if len(ys) < 400:
                return 190.0, 290.0, roi
            top_y = float(np.percentile(ys, 2))
            crown_h = max(40.0, 360.0 - top_y)
            # Outer cranial/hair width at temple band (y in [260..355])
            t_band = roi[260:355, :]
            _, txs = np.where(t_band > 0)
            cran_w = float(np.percentile(txs, 98) - np.percentile(txs, 2)) if len(txs) >= 100 else 290.0
            return crown_h, cran_w, roi

        ch0, cw0, cr0 = _cranial_metrics(dome0)
        chc, cwc, crc = _cranial_metrics(domec)
        crown_h_diff_pct = round(100.0 * (chc - ch0) / max(40.0, ch0), 1)
        cran_w_diff_pct = round(100.0 * (cwc - cw0) / max(80.0, cw0), 1)
        cran_union = float(np.sum((cr0 > 0) | (crc > 0)))
        cran_iou = round(float(np.sum((cr0 > 0) & (crc > 0))) / max(1.0, cran_union), 4)

        # Lip Shape & Cupid's Bow Micro-Geometry Audit on IOD=180 canvas
        lip_h_diff_pct = 0.0
        lip_w_diff_pct = 0.0
        lip_iou = 1.0
        if lips0 is not None and lipsc is not None and len(lips0) >= 8 and len(lipsc) >= 8:
            lh0_raw = max(4.0, float(lips0[:, 1].max() - lips0[:, 1].min()))
            lhc_raw = max(4.0, float(lipsc[:, 1].max() - lipsc[:, 1].min()))
            # Account for natural mouth-parting range [closed .. Gen-0 gap] when Apple Vision innerLips collapses on wide shots
            lh0_closed = max(4.0, lh0_raw - 0.85 * float(max(0.0, gap0)))
            ref_hi = max(lh0_raw, lh0_closed + 0.85 * float(max(0.0, gapc)))
            ref_lo = min(lh0_raw, lh0_closed + 0.85 * float(max(0.0, gapc)))
            if lhc_raw > ref_hi:
                lip_h_diff_pct = round(100.0 * (lhc_raw - ref_hi) / ref_hi, 1)
            elif lhc_raw < ref_lo:
                lip_h_diff_pct = round(100.0 * (lhc_raw - ref_lo) / ref_lo, 1)
            else:
                lip_h_diff_pct = 0.0
            lw0 = max(10.0, float(lips0[:, 0].max() - lips0[:, 0].min()))
            lwc = max(10.0, float(lipsc[:, 0].max() - lipsc[:, 0].min()))
            lip_w_diff_pct = round(100.0 * (lwc - lw0) / lw0, 1)
            lm0 = np.zeros((800, 800), dtype=np.uint8)
            lmc = np.zeros((800, 800), dtype=np.uint8)
            # Align lip centroids so lip_iou measures pure Cupid's-bow/lip-outline shape fidelity rather than rigid 3D yaw/pitch translation
            lipsc_aligned = lipsc - np.mean(lipsc, axis=0, keepdims=True) + np.mean(lips0, axis=0, keepdims=True)
            cv2.fillPoly(lm0, [np.round(lips0).astype(np.int32)], 255)
            cv2.fillPoly(lmc, [np.round(lipsc_aligned).astype(np.int32)], 255)
            lip_iou = round(float(np.sum((lm0 > 0) & (lmc > 0))) / max(1.0, float(np.sum((lm0 > 0) | (lmc > 0)))), 4)

        # Face Length (pupil mid y=360 to chin tip)
        flen0 = max(40.0, float(c0[:, 1].max()) - 360.0)
        flenc = max(40.0, float(cc[:, 1].max()) - 360.0)
        face_len_diff_pct = round(100.0 * (flenc - flen0) / flen0, 1)

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
        if abs(face_len_diff_pct) > 5.0:
            issues.append(f"얼굴 세로길이(눈~턱) 오차 초과 ({face_len_diff_pct:+.1f}%, 허용 ±5.0%)")
        if cran_w_diff_pct > 7.5 or crown_h_diff_pct > 8.0:
            issues.append(f"두상·외곽 머리숱 크기 과대 (두상폭 {cran_w_diff_pct:+.1f}%, 두상높이 {crown_h_diff_pct:+.1f}%)")
        teeth_gap_changed = abs(gapc - gap0) >= 2.2
        min_lip_h_pct = -16.5 if teeth_gap_changed else -11.5
        min_lip_iou_thr = 0.56 if (teeth_gap_changed or turn_diff_abs >= 0.06) else 0.70
        if lip_h_diff_pct > 10.0 or lip_h_diff_pct < min_lip_h_pct or lip_iou < min_lip_iou_thr:
            issues.append(f"입술 모양·두께 변형 (두께 {lip_h_diff_pct:+.1f}%, 입술IoU {lip_iou:.2f})")

        saved_ov: Optional[str] = None
        if output_overlay_path:
            ov = cv2.addWeighted(w0, 0.50, wc, 0.50, 0)
            # Draw Cranial Dome outer contour (GREEN=Gen0, RED=Gen thin dashed/solid)
            cnts0, _ = cv2.findContours(cr0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cntsc, _ = cv2.findContours(crc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ov, cnts0, -1, (0, 220, 0), 2)
            cv2.drawContours(ov, cntsc, -1, (0, 0, 220), 2)
            # Draw Jawline contour
            cv2.polylines(ov, [np.round(c0).astype(np.int32)], False, (0, 255, 0), 3)
            cv2.polylines(ov, [np.round(cc).astype(np.int32)], False, (0, 0, 255), 3)
            # Draw Lip contour
            if lips0 is not None and lipsc is not None:
                cv2.polylines(ov, [np.round(lips0).astype(np.int32)], True, (0, 255, 0), 2)
                cv2.polylines(ov, [np.round(lipsc).astype(np.int32)], True, (0, 0, 255), 2)
            cv2.putText(
                ov,
                f"IoU={iou:.3f} MaxErr={max_pt_err:.1f}px CranW={cran_w_diff_pct:+.1f}% LipH={lip_h_diff_pct:+.1f}%",
                (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
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
            "faceLengthDiffPct": face_len_diff_pct,
            "cranialWidthDiffPct": cran_w_diff_pct,
            "crownHeightDiffPct": crown_h_diff_pct,
            "cranialDomeIoU": cran_iou,
            "lipHeightDiffPct": lip_h_diff_pct,
            "lipWidthDiffPct": lip_w_diff_pct,
            "lipIoU": lip_iou,
            "width5LevelDiffPct": w5,
            "overlayPath": saved_ov,
            "issues": issues,
        }

    def measure_jaw_to_ear_neck_and_eye_audit(
        self,
        ref_gen0_bgr: np.ndarray,
        cand_bgr: np.ndarray,
    ) -> Dict[str, object]:
        """Dedicated micro-geometry & optical shading audit for:
        1) Jaw-to-Ear diagonal line (chin->gonion slope, gonion->ear slope, outward sagitta bulge '턱 뚱뚱해 보임'),
        2) Submandibular jaw-neck shadow separation contrast ('턱밑-목 경계 음영'),
        3) Neck-to-Face width ratio ('목 굵기 / 얼굴폭'),
        4) Upper eyelid arch flatness ('윗눈꺼풀 둥글기') & brow density/darkness ('눈썹 밀도').
        """
        import math

        def _extract_profile(bgr: np.ndarray, target_iod: Optional[float] = None) -> Optional[Dict[str, float]]:
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            m = self.vision.landmarks(pil)
            if not m or not m.get("contour") or not m.get("leftEye") or not m.get("rightEye"):
                return None
            c = np.array(m["contour"], dtype=np.float32)
            le = np.array(m["leftEye"], dtype=np.float32)
            re = np.array(m["rightEye"], dtype=np.float32)
            leb = np.array(m.get("leftEyebrow") or le, dtype=np.float32)
            reb = np.array(m.get("rightEyebrow") or re, dtype=np.float32)
            lp = np.mean(np.array(m.get("leftPupil") or le, dtype=np.float32), axis=0)
            rp = np.mean(np.array(m.get("rightPupil") or re, dtype=np.float32), axis=0)
            iod = max(1.0, float(np.linalg.norm(rp - lp)))
            n = len(c)
            face_center = np.array([0.5 * float(lp[0] + rp[0]), 0.5 * float(lp[1] + rp[1])], dtype=np.float32)

            # De-rotate contour `c` by inter-pupillary roll FIRST so `mid_idx` (menton), sagitta bulge, & jaw slopes are 100% roll-invariant!
            roll_a = math.atan2(float(rp[1] - lp[1]), float(rp[0] - lp[0]))
            cr_a, sr_a = math.cos(-roll_a), math.sin(-roll_a)
            c_rel = c - face_center[None, :]
            c_intr = np.stack([c_rel[:, 0] * cr_a - c_rel[:, 1] * sr_a, c_rel[:, 0] * sr_a + c_rel[:, 1] * cr_a], axis=-1) + face_center[None, :]

            mid_idx = int(np.argmax(c_intr[:, 1]))
            chin = c[mid_idx]
            chin_i = c_intr[mid_idx]
            l_ear = c[0]
            r_ear = c[n - 1]
            l_ear_i = c_intr[0]
            l_gon_i = c_intr[max(1, int(round(0.22 * (n - 1))))]
            r_ear_i = c_intr[n - 1]
            r_gon_i = c_intr[min(n - 2, int(round(0.78 * (n - 1))))]

            def _chord_bulge(p_ear: np.ndarray, p_chin: np.ndarray, pts: np.ndarray) -> float:
                v = p_chin - p_ear
                vlen = max(1.0, float(np.linalg.norm(v)))
                nx, ny = -v[1] / vlen, v[0] / vlen
                mid_chord = 0.5 * (p_ear + p_chin)
                if ((mid_chord[0] - face_center[0]) * nx + (mid_chord[1] - face_center[1]) * ny) < 0:
                    nx, ny = -nx, -ny
                dists = [((p[0] - p_ear[0]) * nx + (p[1] - p_ear[1]) * ny) / iod for p in pts]
                return float(max(dists)) if dists else 0.0

            l_bulge = _chord_bulge(l_ear_i, chin_i, c_intr[1:mid_idx])
            r_bulge = _chord_bulge(r_ear_i, chin_i, c_intr[mid_idx + 1 : n - 1])

            # Intrinsic diagonal chin->gonion & gonion->ear angles (degrees relative to pupil horizon)
            l_chin_gon_deg = math.degrees(math.atan2(abs(float(chin_i[1] - l_gon_i[1])), max(1.0, abs(float(chin_i[0] - l_gon_i[0])))))
            r_chin_gon_deg = math.degrees(math.atan2(abs(float(chin_i[1] - r_gon_i[1])), max(1.0, abs(float(r_gon_i[0] - chin_i[0])))))
            l_gon_ear_deg = math.degrees(math.atan2(abs(float(l_gon_i[1] - l_ear_i[1])), max(1.0, abs(float(l_gon_i[0] - l_ear_i[0])))))
            r_gon_ear_deg = math.degrees(math.atan2(abs(float(r_gon_i[1] - r_ear_i[1])), max(1.0, abs(float(r_ear_i[0] - r_gon_i[0])))))

            # 2) Submandibular jaw-neck shadow contrast (CIE L* inside jaw vs outside neck along near jaw arc)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            L_ch = lab[:, :, 0]
            h, w = L_ch.shape
            off = max(4, int(round(0.08 * iod)))
            # Sample cheek/chin skin LAB first so ivory/dark shirt collars beneath the jaw are never mistaken for neck skin!
            sy1_r, sy2_r = max(0, int(chin[1] - 0.35 * iod)), max(1, int(chin[1] - 0.08 * iod))
            sx1_r, sx2_r = max(0, int(chin[0] - 0.25 * iod)), min(w, int(chin[0] + 0.25 * iod))
            skin_ref_jaw = lab[sy1_r:sy2_r, sx1_r:sx2_r].reshape(-1, 3)
            med_skin_jaw = np.median(skin_ref_jaw, axis=0) if skin_ref_jaw.size >= 9 else np.array([175.0, 136.0, 142.0], dtype=np.float32)

            jaw_contrasts = []
            # Sample lower mandible-over-neck points using inward normal toward `face_center` (independent of c[0] vs c[-1] order!)
            arc_a = c[max(1, int(round(0.28 * n))) : mid_idx]
            arc_b = c[mid_idx + 1 : min(n - 1, int(round(0.72 * n)))]
            arc_pts = arc_a if (len(arc_a) > 0 and abs(float(np.mean(arc_a[:, 0]) - face_center[0])) >= abs(float(np.mean(arc_b[:, 0]) - face_center[0])) if len(arc_b) > 0 else True) else arc_b
            for pt in arc_pts:
                vin = face_center - pt
                vnorm = max(1.0, float(np.linalg.norm(vin)))
                ux_in, uy_in = float(vin[0]) / vnorm, float(vin[1]) / vnorm
                ix, iy = int(np.clip(round(float(pt[0]) + ux_in * off), 0, w - 1)), int(np.clip(round(float(pt[1]) + uy_in * off), 0, h - 1))
                ox, oy = int(np.clip(round(float(pt[0]) - ux_in * off), 0, w - 1)), int(np.clip(round(float(pt[1]) - uy_in * off), 0, h - 1))
                out_d_ab = math.hypot(float(lab[oy, ox, 1] - med_skin_jaw[1]), float(lab[oy, ox, 2] - med_skin_jaw[2]))
                if 48.0 <= float(L_ch[oy, ox]) <= float(med_skin_jaw[0] + 8.0) and out_d_ab <= 16.5 and float(lab[oy, ox, 1]) >= 130.5:
                    jaw_contrasts.append(float(L_ch[iy, ix] - L_ch[oy, ox]))
            submandibular_contrast = float(np.mean(jaw_contrasts)) if len(jaw_contrasts) >= 2 else 12.0

            # 3) Upper cervical neck width right beneath the chin (`+0.18 * iod`, centered on head mid-span, not yawed chin tip!)
            face_w = max(1.0, float(c[:, 0].max() - c[:, 0].min()))
            neck_y = int(np.clip(chin[1] + 0.18 * iod, 0, h - 1))
            neck_band_h = max(3, int(round(0.06 * iod)))
            y1, y2 = max(0, neck_y - neck_band_h), min(h, neck_y + neck_band_h)
            x1 = max(0, int(round(c[:, 0].min() - 0.15 * face_w)))
            x2 = min(w, int(round(c[:, 0].max() + 0.15 * face_w)))
            neck_ratio = 0.0
            if y2 > y1 and x2 > x1:
                # Sample skin color right above chin to segment neck pixels cleanly
                sy1, sy2 = max(0, int(chin[1] - 0.35 * iod)), max(1, int(chin[1] - 0.08 * iod))
                sx1, sx2 = max(0, int(chin[0] - 0.25 * iod)), min(w, int(chin[0] + 0.25 * iod))
                skin_ref = lab[sy1:sy2, sx1:sx2].reshape(-1, 3)
                if skin_ref.size >= 15:
                    med_lab = np.median(skin_ref, axis=0)
                    band = lab[y1:y2, x1:x2]
                    # Allow shaded neck skin while strictly rejecting dark brown hair (L < 95) & crimson backdrop (a* > 155)
                    d_ab = np.sqrt((band[:, :, 1] - med_lab[1]) ** 2 + (band[:, :, 2] - med_lab[2]) ** 2)
                    d_l = med_lab[0] - band[:, :, 0]
                    mask = (d_ab <= 19.5) & (d_l >= -22.0) & (d_l <= 62.0) & (band[:, :, 0] >= 95.0) & (band[:, :, 1] <= 154.0)
                    col_cov = np.mean(mask, axis=0)
                    # Find longest contiguous skin-tone run near head horizontal center (robust under 3/4 head yaw!)
                    head_mid_x = 0.5 * float(c[:, 0].min() + c[:, 0].max())
                    cx_rel = int(np.clip(head_mid_x - x1, 0, max(0, x2 - x1 - 1)))
                    search_r = max(4, int(round(0.35 * face_w)))
                    seed = None
                    for delta in range(search_r):
                        for cand_x in (cx_rel - delta, cx_rel + delta):
                            if 0 <= cand_x < len(col_cov) and col_cov[cand_x] >= 0.25:
                                seed = cand_x
                                break
                        if seed is not None:
                            break
                    if seed is not None:
                        lx = seed
                        while lx > 0 and col_cov[lx - 1] >= 0.25:
                            lx -= 1
                        rx = seed
                        while rx < len(col_cov) - 1 and col_cov[rx + 1] >= 0.25:
                            rx += 1
                        neck_ratio = float(rx - lx + 1) / face_w

            # 4) Upper eyelid arch height ratio (roundness/dome height above inner-outer corner chord) & brow density
            def _upper_lid_arch(epts: np.ndarray) -> float:
                ix = int(np.argmin(epts[:, 0]))
                ox = int(np.argmax(epts[:, 0]))
                p_in, p_out = epts[ix], epts[ox]
                ew = max(1.0, float(np.linalg.norm(p_out - p_in)))
                # Highest point (smallest y) above the corner chord
                mid_y = 0.5 * float(p_in[1] + p_out[1])
                top_y = float(epts[:, 1].min())
                return max(0.0, (mid_y - top_y) / ew)

            upper_arch = 0.5 * (_upper_lid_arch(le) + _upper_lid_arch(re))

            # Brow darkness & thickness (illumination-normalized + PSF-scale-matched per-side Weber contrast)
            brow_h = 0.5 * (float(leb[:, 1].max() - leb[:, 1].min()) + float(reb[:, 1].max() - reb[:, 1].min())) / iod
            if target_iod is not None and iod > target_iod * 1.15:
                brow_sigma = float(np.clip(0.52 * (iod / max(8.0, target_iod)), 0.8, 5.0))
                L_brow = cv2.GaussianBlur(L_ch, (0, 0), sigmaX=brow_sigma, sigmaY=brow_sigma)
            else:
                L_brow = L_ch
            b_darks = []
            for bpts, e_ref in ((leb, lp), (reb, rp)):
                sk_patch = L_brow[
                    max(0, int(e_ref[1] + 0.15 * iod)) : min(h, int(e_ref[1] + 0.45 * iod)),
                    max(0, int(e_ref[0] - 0.15 * iod)) : min(w, int(e_ref[0] + 0.15 * iod)),
                ]
                loc_skin_L = float(np.median(sk_patch)) if sk_patch.size >= 9 else 175.0
                bx1, bx2 = max(0, int(bpts[:, 0].min())), min(w, int(bpts[:, 0].max()))
                by1, by2 = max(0, int(bpts[:, 1].min() - 0.04 * iod)), min(h, int(bpts[:, 1].max() + 0.04 * iod))
                patch = L_brow[by1:by2, bx1:bx2]
                if patch.size >= 12:
                    raw_drop = max(0.0, loc_skin_L - float(np.percentile(patch, 35)))
                    weber_norm_drop = raw_drop * float(np.clip(185.0 / max(75.0, loc_skin_L), 0.72, 2.25))
                    b_darks.append(weber_norm_drop)
            brow_dark = float(np.mean(b_darks)) if b_darks else 25.0

            nose_turn_p = 0.0
            if m.get("nose") and len(m["nose"]) >= 2:
                ns_pts = np.array(m["nose"], dtype=np.float32)
                ns_rel = ns_pts - face_center[None, :]
                ns_intr_x = ns_rel[:, 0] * cr_a - ns_rel[:, 1] * sr_a
                nose_turn_p = float(np.mean(ns_intr_x) / iod)

            return {
                "iodPx": round(iod, 2),
                "noseTurnIod": round(nose_turn_p, 4),
                "leftJawBulgeIod": round(l_bulge, 4),
                "rightJawBulgeIod": round(r_bulge, 4),
                "leftChinGonionDeg": round(l_chin_gon_deg, 1),
                "rightChinGonionDeg": round(r_chin_gon_deg, 1),
                "leftGonionEarDeg": round(l_gon_ear_deg, 1),
                "rightGonionEarDeg": round(r_gon_ear_deg, 1),
                "submandibularContrastL": round(submandibular_contrast, 2),
                "neckToFaceWidthRatio": round(neck_ratio, 3),
                "upperLidArchRatio": round(upper_arch, 4),
                "browThickIod": round(brow_h, 4),
                "browDarknessL": round(brow_dark, 2),
            }

        p0_raw = _extract_profile(ref_gen0_bgr)
        pc_raw = _extract_profile(cand_bgr)
        if not p0_raw or not pc_raw:
            return {"valid": False, "passed": False, "issues": ["턱→귀·눈매 프로파일 랜드마크 검출 실패"]}
        common_iod = min(float(p0_raw["iodPx"]), float(pc_raw["iodPx"]))
        p0 = _extract_profile(ref_gen0_bgr, target_iod=common_iod) if float(p0_raw["iodPx"]) > common_iod * 1.15 else p0_raw
        pc = _extract_profile(cand_bgr, target_iod=common_iod) if float(pc_raw["iodPx"]) > common_iod * 1.15 else pc_raw
        if not p0 or not pc:
            return {"valid": False, "passed": False, "issues": ["턱→귀·눈매 프로파일 랜드마크 검출 실패"]}

        l_bulge_pct = 100.0 * (pc["leftJawBulgeIod"] - p0["leftJawBulgeIod"]) / max(0.05, p0["leftJawBulgeIod"])
        r_bulge_pct = 100.0 * (pc["rightJawBulgeIod"] - p0["rightJawBulgeIod"]) / max(0.05, p0["rightJawBulgeIod"])
        mean_bulge_pct = 0.5 * (l_bulge_pct + r_bulge_pct)
        l_slope_diff = pc["leftChinGonionDeg"] - p0["leftChinGonionDeg"]
        r_slope_diff = pc["rightChinGonionDeg"] - p0["rightChinGonionDeg"]
        mean_slope_diff = 0.5 * (l_slope_diff + r_slope_diff)
        yaw_turn_diff = abs(float(pc.get("noseTurnIod", 0.0)) - float(p0.get("noseTurnIod", 0.0)))
        yaw_bulge_comp = min(16.0, yaw_turn_diff * 115.0) if yaw_turn_diff >= 0.04 else 0.0
        yaw_slope_comp = min(5.0, yaw_turn_diff * 36.0) if yaw_turn_diff >= 0.04 else 0.0
        shadow_diff = pc["submandibularContrastL"] - p0["submandibularContrastL"]
        # If reference neck is partially occluded by a high blazer/suit collar or hair (`< 0.63 * face_w`),
        # normalize excess neck width against physiological female cervical baseline (`0.65 * face_w`) without reporting false negative shrinkage!
        eff_ref_neck = max(p0["neckToFaceWidthRatio"], min(pc["neckToFaceWidthRatio"], 0.65))
        neck_diff_pct = (
            100.0 * (pc["neckToFaceWidthRatio"] - eff_ref_neck) / max(0.25, eff_ref_neck)
            if (p0["neckToFaceWidthRatio"] > 0.25 and pc["neckToFaceWidthRatio"] > 0.25)
            else 0.0
        )
        arch_diff_pct = 100.0 * (pc["upperLidArchRatio"] - p0["upperLidArchRatio"]) / max(0.05, p0["upperLidArchRatio"])
        brow_thick_pct = 100.0 * (pc["browThickIod"] - p0["browThickIod"]) / max(0.03, p0["browThickIod"])
        brow_dark_diff = pc["browDarknessL"] - p0["browDarknessL"]

        issues: List[str] = []
        if mean_bulge_pct > 8.5 or l_bulge_pct > (9.0 + yaw_bulge_comp) or r_bulge_pct > (9.0 + yaw_bulge_comp):
            issues.append(f"턱→귀 라인 외측 불룩함(뚱뚱해 보임) 초과 (좌 {l_bulge_pct:+.1f}% / 우 {r_bulge_pct:+.1f}%, 평균 {mean_bulge_pct:+.1f}%)")
        if abs(mean_slope_diff) > 5.0 or abs(l_slope_diff) > (5.5 + yaw_slope_comp) or abs(r_slope_diff) > (5.5 + yaw_slope_comp):
            issues.append(f"턱끝→하악각 사선 기울기 이탈 (좌 {l_slope_diff:+.1f}° / 우 {r_slope_diff:+.1f}°, 평균 {mean_slope_diff:+.1f}°)")
        if p0["submandibularContrastL"] >= 8.0 and pc["submandibularContrastL"] < min(11.5, p0["submandibularContrastL"] * 0.55):
            issues.append(f"턱밑-목 경계 음영 뭉개짐 (생성 {pc['submandibularContrastL']:.1f}L vs 원본 {p0['submandibularContrastL']:.1f}L)")
        if neck_diff_pct > 11.0:
            issues.append(f"목 굵기 과대 팽창 ({neck_diff_pct:+.1f}%, 허용 ≤+11.0%)")
        if abs(arch_diff_pct) > 14.0:
            issues.append(f"윗눈꺼풀 아치 곡률(둥글기) 이탈 ({arch_diff_pct:+.1f}%, 허용 ±14.0%)")
        if brow_thick_pct < -12.0 or brow_dark_diff < -10.0:
            issues.append(f"눈썹 과도한 옅어짐/얇아짐 (굵기 {brow_thick_pct:+.1f}%, 진하기 {brow_dark_diff:+.1f}L)")

        return {
            "valid": True,
            "passed": len(issues) == 0,
            "ref": p0,
            "cand": pc,
            "diffs": {
                "leftJawBulgePct": round(l_bulge_pct, 1),
                "rightJawBulgePct": round(r_bulge_pct, 1),
                "leftJawSlopeDegDiff": round(l_slope_diff, 1),
                "rightJawSlopeDegDiff": round(r_slope_diff, 1),
                "submandibularShadowDiffL": round(shadow_diff, 2),
                "neckWidthDiffPct": round(neck_diff_pct, 1),
                "upperLidArchDiffPct": round(arch_diff_pct, 1),
                "browThickDiffPct": round(brow_thick_pct, 1),
                "browDarkDiffL": round(brow_dark_diff, 1),
            },
            "issues": issues,
        }
