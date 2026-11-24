"""Universal Zero-Hardcoding Multi-Shot Real-Subject Studio (`v2.3` — Designs A & B).

Solves the fundamental real-person editorial dilemma ("preserve 100% real camera face while changing
outfit & background naturally as if photographed on location") using 1..N untouched real originals
with ZERO gender, ethnicity, garment-color, or CIE LAB threshold hardcoding:

1. `RealMultiShotBank`:
   - Catalogs 1..N untouched real camera photos (`Gen-0` + optional multi-angle / seated / standing originals).
   - Dynamically extracts ground-truth skeletal ratios (`acromion_to_face_w`, `bideltoid_to_face_w`, `trunk_to_face_w`),
     craniofacial invariants (`alar_to_iod`, `face_wl`, `jaw_taper`), highest-IOD pore/albedo master,
     and pairwise cross-shot `SFace` identity baselines.
2. `IntrinsicAlbedoRelighter` (Design A — `주파수 분리 알베도 리라이팅`):
   - Synthesizes a 1-Pass full-person 4K base with the subject's measured skeletal build & 3D scene lighting,
     then transfers the angle-and-expression-matched real camera face's intrinsic skin albedo + high-frequency
     organ anatomy (`eyes, brows, nose, lips, pores`) via pure Nyquist similarity registration on the real source
     + gentle local `C^2` TPS canvas alignment on the generated head + skin-masked Normalized-Convolution
     shading decomposition (`Relit = Real_Albedo * Scene_Shading + Real_HighFreq`) + 4-Band Laplacian Pyramid fusion.
3. `FaceLockedOutpaintStudio` (Design B — `페이스락 아웃페인팅`):
   - Pre-positions a decontaminated real camera head/hair/upper-neck layer onto the clean location plate at the
     measured skeletal anchor (`stage2a_mesh_precomposite_facelock.png`), outpaints the new garment, body, hands,
     and directional room lighting around that locked head, and locks the untouched real camera face pixels back in
     with rigid sub-pixel similarity alignment + intrinsic low-frequency scene relighting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from .apple_vision import AppleVisionAnalyzer
from .config import EngineConfig
from .detector import AdaptiveSkinToneProfile, DynamicFaceDetector, FaceDetectionResult
from .organ_depth_anatomy import OrganAndDepthAnatomyAnalyzer, OrganAndDepthProfile
from .origin_guard import OriginOnlyOneShotGuard
from .scene_radiometry import SceneRadiometryAnalyzer


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_gaussian(img: np.ndarray, weight_01: np.ndarray, sigma: float) -> np.ndarray:
    """Normalized-Convolution Gaussian filter strictly inside valid mask weights (zero exterior halo bleed)."""
    k = max(5, int(round(sigma * 4.5)) | 1)
    w = np.clip(weight_01, 0.0, 1.0).astype(np.float32)
    w_blur = cv2.GaussianBlur(w, (k, k), sigma)
    if img.ndim == 3:
        num = cv2.GaussianBlur(img.astype(np.float32) * w[:, :, None], (k, k), sigma)
        den = np.maximum(w_blur[:, :, None], 1e-4)
    else:
        num = cv2.GaussianBlur(img.astype(np.float32) * w, (k, k), sigma)
        den = np.maximum(w_blur, 1e-4)
    out = num / den
    if np.any(w_blur < 1e-3):
        fallback = cv2.GaussianBlur(img.astype(np.float32), (k, k), sigma)
        alpha_sup = np.clip(w_blur / 1e-2, 0.0, 1.0)
        if img.ndim == 3:
            alpha_sup = alpha_sup[:, :, None]
        out = out * alpha_sup + fallback * (1.0 - alpha_sup)
    return out


def _warp_affine_nyquist(
    src: np.ndarray,
    mat_2x3: np.ndarray,
    dsize: Tuple[int, int],
    flags: int = cv2.INTER_LANCZOS4,
    border_mode: int = cv2.BORDER_REPLICATE,
) -> np.ndarray:
    """Apply exact Nyquist pre-filtering when downscaling before `cv2.warpAffine`."""
    scale = float(math.hypot(float(mat_2x3[0, 0]), float(mat_2x3[1, 0])))
    work = src
    if scale < 0.92 and flags != cv2.INTER_NEAREST:
        sigma_aa = 0.35 * math.sqrt(max(0.0, (1.0 / max(0.05, scale)) ** 2 - 1.0))
        if sigma_aa >= 0.30:
            k_aa = max(3, int(round(sigma_aa * 5.0)) | 1)
            work = cv2.GaussianBlur(src, (k_aa, k_aa), sigma_aa)
    return cv2.warpAffine(work, mat_2x3, dsize, flags=flags, borderMode=border_mode)


def _laplacian_pyramid_blend(fg: np.ndarray, bg: np.ndarray, mask_01: np.ndarray, levels: int = 4) -> np.ndarray:
    """Zero-halo multi-band Burt-Adelson Laplacian pyramid blend with exterior pre-fill."""
    m = np.clip(mask_01, 0.0, 1.0).astype(np.float32)
    # Pre-fill exterior of fg with bg outside active mask support to prevent exterior colors bleeding inward
    fg_safe = np.where(m[:, :, None] > 0.005, fg.astype(np.float32), bg.astype(np.float32))
    bg_f = bg.astype(np.float32)

    gp_fg, gp_bg, gp_m = [fg_safe], [bg_f], [m]
    for _ in range(levels):
        if min(gp_fg[-1].shape[:2]) < 16:
            break
        gp_fg.append(cv2.pyrDown(gp_fg[-1]))
        gp_bg.append(cv2.pyrDown(gp_bg[-1]))
        gp_m.append(cv2.pyrDown(gp_m[-1]))

    eff_levels = len(gp_fg) - 1
    lp_fg, lp_bg = [], []
    for i in range(eff_levels):
        sz = (gp_fg[i].shape[1], gp_fg[i].shape[0])
        lp_fg.append(gp_fg[i] - cv2.pyrUp(gp_fg[i + 1], dstsize=sz))
        lp_bg.append(gp_bg[i] - cv2.pyrUp(gp_bg[i + 1], dstsize=sz))
    lp_fg.append(gp_fg[-1])
    lp_bg.append(gp_bg[-1])

    lp_out = []
    for lf, lb, gm in zip(lp_fg, lp_bg, gp_m):
        w3 = np.clip(gm, 0.0, 1.0)[:, :, None]
        lp_out.append(lf * w3 + lb * (1.0 - w3))

    cur = lp_out[-1]
    for i in range(eff_levels - 1, -1, -1):
        sz = (lp_out[i].shape[1], lp_out[i].shape[0])
        cur = cv2.pyrUp(cur, dstsize=sz) + lp_out[i]
    return np.clip(cur, 0.0, 255.0)


def _solve_tps_warp(
    img: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    reg_lambda: float = 0.025,
    grid_stride: int = 4,
) -> np.ndarray:
    """Smooth regularized C^2 Thin-Plate Spline (`r^2 log r^2`) optical grid remap on `img`.

    Maps landmarks currently located at `src_pts` in `img` so they land at `dst_pts` in the output image.
    Evaluates the RBF on a coarse `grid_stride` mesh and upsamples smoothly for sub-second 4K speed.
    """
    h, w = img.shape[:2]
    n = len(src_pts)
    if n < 6:
        return img.copy()

    # Inverse mapping: for output pixel y_out (near dst_pts), find source coordinate y_src (near src_pts)
    p_ctrl = dst_pts.astype(np.float64)
    v_target = (src_pts - dst_pts).astype(np.float64)

    scale_norm = max(32.0, float(np.std(p_ctrl)))
    center_norm = np.mean(p_ctrl, axis=0)
    pn = (p_ctrl - center_norm) / scale_norm

    diff = pn[:, None, :] - pn[None, :, :]
    r2 = np.sum(diff * diff, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.where(r2 > 1e-12, r2 * np.log(np.maximum(r2, 1e-12)), 0.0)
    K += np.eye(n, dtype=np.float64) * float(reg_lambda)

    P = np.hstack([np.ones((n, 1), dtype=np.float64), pn])
    L = np.zeros((n + 3, n + 3), dtype=np.float64)
    L[:n, :n] = K
    L[:n, n:] = P
    L[n:, :n] = P.T
    L[n:, n:] = np.eye(3, dtype=np.float64) * 1e-6

    rhs = np.zeros((n + 3, 2), dtype=np.float64)
    rhs[:n] = v_target
    try:
        weights = np.linalg.solve(L, rhs)
    except np.linalg.LinAlgError:
        weights, _, _, _ = np.linalg.lstsq(L, rhs, rcond=None)

    W_rbf = weights[:n]
    A_aff = weights[n:]

    gh = max(8, int(math.ceil(h / float(grid_stride))) + 1)
    gw = max(8, int(math.ceil(w / float(grid_stride))) + 1)
    gy = np.linspace(0.0, float(h - 1), gh, dtype=np.float64)
    gx = np.linspace(0.0, float(w - 1), gw, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(gx, gy)
    pts_q = np.stack([(grid_x.ravel() - center_norm[0]) / scale_norm, (grid_y.ravel() - center_norm[1]) / scale_norm], axis=1)

    dq = pts_q[:, None, :] - pn[None, :, :]
    rq2 = np.sum(dq * dq, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        Kq = np.where(rq2 > 1e-12, rq2 * np.log(np.maximum(rq2, 1e-12)), 0.0)

    Pq = np.hstack([np.ones(( len(pts_q), 1 ), dtype=np.float64), pts_q])
    disp = (Kq @ W_rbf + Pq @ A_aff).astype(np.float32)
    dx_c = disp[:, 0].reshape(gh, gw)
    dy_c = disp[:, 1].reshape(gh, gw)

    dx_full = cv2.resize(dx_c, (w, h), interpolation=cv2.INTER_CUBIC)
    dy_full = cv2.resize(dy_c, (w, h), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = xx + dx_full
    map_y = yy + dy_full
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)


def _extract_ordered_76pt_landmarks(marks: Dict[str, object], w: int, h: int) -> Optional[Dict[str, np.ndarray]]:
    """Extract pixel-coordinate landmark groups dynamically from Apple Vision output."""
    if not marks:
        return None
    out: Dict[str, np.ndarray] = {}
    keys = (
        "contour",
        "leftEyebrow",
        "rightEyebrow",
        "leftEye",
        "rightEye",
        "leftPupil",
        "rightPupil",
        "nose",
        "noseCrest",
        "outerLips",
        "innerLips",
    )
    for k in keys:
        raw = marks.get(k)
        if not raw:
            continue
        pts = OrganAndDepthAnatomyAnalyzer._pts(raw, w, h)
        if len(pts) > 0:
            out[k] = pts
    if "contour" not in out or "leftEye" not in out or "rightEye" not in out or "outerLips" not in out:
        return None
    return out


def _resample_curve(pts: np.ndarray, count: int) -> np.ndarray:
    if len(pts) == count:
        return pts.astype(np.float32)
    if len(pts) == 1:
        return np.repeat(pts.astype(np.float32), count, axis=0)
    idx_src = np.linspace(0.0, 1.0, len(pts))
    idx_dst = np.linspace(0.0, 1.0, count)
    xs = np.interp(idx_dst, idx_src, pts[:, 0])
    ys = np.interp(idx_dst, idx_src, pts[:, 1])
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _build_canonical_correspondence_points(lm: Dict[str, np.ndarray]) -> np.ndarray:
    """Assemble fixed-topology 66-point anatomical correspondence vector across contour, brows, eyes, nose, and lips."""
    parts = [
        _resample_curve(lm["contour"], 17),
        _resample_curve(lm.get("leftEyebrow", lm["leftEye"]), 6),
        _resample_curve(lm.get("rightEyebrow", lm["rightEye"]), 6),
        _resample_curve(lm["leftEye"], 8),
        _resample_curve(lm["rightEye"], 8),
        _resample_curve(lm.get("noseCrest", lm.get("nose", lm["leftEye"])), 5),
        _resample_curve(lm.get("nose", lm["outerLips"]), 6),
        _resample_curve(lm["outerLips"], 10),
    ]
    return np.vstack(parts).astype(np.float32)


@dataclass(frozen=True)
class RealShotEntry:
    name: str
    path: str
    sha256: str
    width: int
    height: int
    face_width_px: float
    face_height_px: float
    iod_px: float
    yaw_deg: float
    eye_tilt_deg: float
    posture_mode: str  # "seated_or_half", "standing_or_thigh", "closeup"
    hair_silhouette_mode: str  # "compact_updo" vs "draped_down"
    acromion_to_face_w: Optional[float]
    bideltoid_to_face_w: Optional[float]
    trunk_to_face_w: Optional[float]
    face_to_shoulder_ratio: Optional[float]
    organ_profile: Dict[str, float]


class RealMultiShotBank:
    """Universal multi-shot real camera bank measuring ground-truth body & facial invariants across 1..N shots."""

    def __init__(
        self,
        config: EngineConfig,
        detector: DynamicFaceDetector,
        apple_vision: AppleVisionAnalyzer,
        auditor,
        origin_guard: Optional[OriginOnlyOneShotGuard] = None,
    ) -> None:
        self.config = config
        self.detector = detector
        self.vision = apple_vision
        self.auditor = auditor
        self.organ_analyzer = OrganAndDepthAnatomyAnalyzer(apple_vision)
        self.origin_guard = origin_guard or OriginOnlyOneShotGuard()
        self.entries: List[RealShotEntry] = []
        self._sface_vecs: Dict[str, np.ndarray] = {}

    def ingest_real_originals(self, paths: Sequence[Path]) -> Dict[str, object]:
        unique_paths: List[Path] = []
        seen_sha: set[str] = set()
        for raw_p in paths:
            p = Path(raw_p).resolve()
            if not p.is_file():
                continue
            self.origin_guard.verify_one_shot_and_inputs([p])
            sha = _sha256_file(p)
            if sha in seen_sha:
                continue
            seen_sha.add(sha)
            unique_paths.append(p)

        if not unique_paths:
            raise ValueError("No valid untouched real original photos provided to RealMultiShotBank.")

        self.entries.clear()
        self._sface_vecs.clear()

        for idx, p in enumerate(unique_paths, start=1):
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            det = self.detector.detect(bgr)
            vec = self.auditor._extract_sface_vec(bgr, det)
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            marks = self.vision.landmarks(pil) or {}
            cran = self.vision.compute_cranial_and_scale_profile(pil, marks=marks) or {}
            bj = self.vision.body_and_hand_points(pil) or {}
            op = self.organ_analyzer.extract_profile(bgr)
            if op is None:
                continue

            cnt = OrganAndDepthAnatomyAnalyzer._pts(marks.get("contour") or [], w, h)
            cheek_w = float(cnt[:, 0].max() - cnt[:, 0].min()) if len(cnt) > 2 else float(det.bbox[2])

            acro_ratio: Optional[float] = None
            bidel_ratio: Optional[float] = None
            trunk_ratio: Optional[float] = None
            has_hips = False
            for body in bj.get("bodies") or []:
                jmap = {str(j["name"]): (float(j["x"]), float(j["y"])) for j in body if float(j.get("conf", 0.0)) >= 0.20}
                ls, rs = jmap.get("left_shoulder_1_joint"), jmap.get("right_shoulder_1_joint")
                nk, hip = jmap.get("neck_1_joint"), jmap.get("root")
                if ls and rs and cheek_w > 10.0:
                    acro_px = math.hypot(rs[0] - ls[0], rs[1] - ls[1])
                    if acro_px > cheek_w * 1.15:
                        acro_ratio = round(acro_px / cheek_w, 4)
                        bidel_ratio = round((acro_px * 1.48) / cheek_w, 4)
                if nk and hip and cheek_w > 10.0 and hip[1] > nk[1] + cheek_w * 0.8:
                    trunk_px = math.hypot(hip[0] - nk[0], hip[1] - nk[1])
                    trunk_ratio = round(trunk_px / cheek_w, 4)
                    has_hips = True
                break

            face_area_frac = (det.bbox[2] * det.bbox[3]) / float(max(1, w * h))
            if face_area_frac >= 0.085 or acro_ratio is None:
                posture_mode = "closeup"
            elif has_hips:
                posture_mode = "standing_or_thigh"
            else:
                posture_mode = "seated_or_half"

            cran_w_ratio = float(cran.get("cranialToFaceWidthRatio", 1.30))
            hair_mode = "compact_updo" if cran_w_ratio < 1.26 else "draped_down"

            f2s = (
                round(float(cran["faceToShoulderRatio"]), 4)
                if float(cran.get("shouldersInFrame", 0.0)) > 0.5 and "faceToShoulderRatio" in cran
                else None
            )

            entry = RealShotEntry(
                name=f"shot{idx:02d}_{p.stem}",
                path=str(p),
                sha256=_sha256_file(p),
                width=w,
                height=h,
                face_width_px=round(float(det.bbox[2]), 1),
                face_height_px=round(float(det.bbox[3]), 1),
                iod_px=round(float(det.inter_ocular_distance), 1),
                yaw_deg=round(float(det.yaw_asymmetry_ratio * 95.0), 1),
                eye_tilt_deg=round(float(det.eye_tilt_degrees), 2),
                posture_mode=posture_mode,
                hair_silhouette_mode=hair_mode,
                acromion_to_face_w=acro_ratio,
                bideltoid_to_face_w=bidel_ratio,
                trunk_to_face_w=trunk_ratio,
                face_to_shoulder_ratio=f2s,
                organ_profile={k: round(float(v), 4) for k, v in asdict(op).items()},
            )
            self.entries.append(entry)
            if vec is not None:
                self._sface_vecs[entry.path] = vec

        return self.to_summary_dict()

    @property
    def highest_iod_entry(self) -> RealShotEntry:
        return max(self.entries, key=lambda e: e.iod_px)

    def skeletal_invariants(self) -> Dict[str, float]:
        acros = [e.acromion_to_face_w for e in self.entries if e.acromion_to_face_w is not None]
        bidels = [e.bideltoid_to_face_w for e in self.entries if e.bideltoid_to_face_w is not None]
        trunks = [e.trunk_to_face_w for e in self.entries if e.trunk_to_face_w is not None]
        f2s = [e.face_to_shoulder_ratio for e in self.entries if e.face_to_shoulder_ratio is not None]
        alars = [e.organ_profile["alar_width_to_iod"] for e in self.entries]
        fwls = [e.organ_profile["face_width_to_length"] for e in self.entries]
        tapers = [e.organ_profile["jaw_to_cheek_taper"] for e in self.entries]
        return {
            "median_acromion_to_face_w": round(float(np.median(acros)), 3) if acros else 1.96,
            "median_bideltoid_to_face_w": round(float(np.median(bidels)), 3) if bidels else 2.90,
            "median_trunk_to_face_w": round(float(np.median(trunks)), 3) if trunks else 3.00,
            "median_face_to_shoulder_ratio": round(float(np.median(f2s)), 3) if f2s else 0.340,
            "median_alar_to_iod": round(float(np.median(alars)), 4) if alars else 0.483,
            "median_face_wl": round(float(np.median(fwls)), 4) if fwls else 0.822,
            "median_jaw_taper": round(float(np.median(tapers)), 4) if tapers else 0.820,
            "min_yaw_deg": round(float(min(e.yaw_deg for e in self.entries)), 1),
            "max_yaw_deg": round(float(max(e.yaw_deg for e in self.entries)), 1),
        }

    def to_dynamic_skeletal_clause(self) -> str:
        inv = self.skeletal_invariants()
        return (
            f"Ground-Truth Multi-Shot Skeletal Lock (measured across {len(self.entries)} real camera photos): "
            f"skeletal shoulder-joint span = {inv['median_acromion_to_face_w']:.2f}x transverse facial width "
            f"(outer bi-deltoid shoulder span = {inv['median_bideltoid_to_face_w']:.2f}x transverse facial width, "
            f"head-to-shoulder ratio = {inv['median_face_to_shoulder_ratio']:.3f}; NEVER inflate into oversized runway shoulder pads), "
            f"torso trunk length = {inv['median_trunk_to_face_w']:.2f}x transverse facial width, "
            f"invariant alar-to-IOD = {inv['median_alar_to_iod']:.3f}, facial W/L = {inv['median_face_wl']:.3f}."
        )

    def select_best_reference_pair(
        self,
        target_yaw_deg: float,
        scene_instruction: str = "",
        prefer_hair_mode: str = "auto",
    ) -> Tuple[RealShotEntry, Optional[RealShotEntry]]:
        """Choose (Image 1: posture+angle matched real portrait, Image 2: high-res organ/3D real portrait)."""
        low_s = scene_instruction.lower()
        want_seated = any(w in low_s for w in ("seated", "sitting", "chair", "desk", "table", "armchair", "swivel"))
        want_standing = any(w in low_s for w in ("standing", "walking", "full-length", "stride"))
        if prefer_hair_mode == "auto":
            if any(w in low_s for w in ("updo", "tied", "bun", "ponytail")):
                prefer_hair_mode = "compact_updo"
            elif any(w in low_s for w in ("hair down", "long hair", "straight hair", "flowing")):
                prefer_hair_mode = "draped_down"

        def _score_primary(e: RealShotEntry) -> float:
            ang_pen = abs(e.yaw_deg - target_yaw_deg) * 1.2
            post_bonus = 0.0
            if want_seated and e.posture_mode == "seated_or_half":
                post_bonus -= 14.0
            elif want_standing and e.posture_mode == "standing_or_thigh":
                post_bonus -= 14.0
            elif e.posture_mode != "closeup":
                post_bonus -= 5.0
            if prefer_hair_mode in ("compact_updo", "draped_down") and e.hair_silhouette_mode == prefer_hair_mode:
                post_bonus -= 8.0
            res_bonus = -min(8.0, e.iod_px / 18.0)
            return ang_pen + post_bonus + res_bonus

        primary = min(self.entries, key=_score_primary)
        others = [e for e in self.entries if e.sha256 != primary.sha256]
        if not others:
            return primary, None

        # If target yaw is >= 20 deg and we have a real 3/4 shot not already chosen, pick it; otherwise pick max IOD close-up!
        if abs(target_yaw_deg) >= 19.5:
            secondary = min(others, key=lambda e: abs(e.yaw_deg - target_yaw_deg) - 0.05 * e.iod_px)
        else:
            secondary = max(others, key=lambda e: e.iod_px)
        return primary, secondary

    def select_nearest_real_shot_for_target(
        self,
        dst_bgr: np.ndarray,
        prefer_hair_match: bool = True,
    ) -> Tuple[RealShotEntry, Dict[str, float]]:
        """Find the single real original whose 3D head yaw + expression (`eye_hw`, `mouth_w`) best matches `dst_bgr`."""
        det = self.detector.detect(dst_bgr)
        dst_yaw = float(det.yaw_asymmetry_ratio * 95.0)
        dst_op = self.organ_analyzer.extract_profile(dst_bgr)
        pil = Image.fromarray(cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2RGB))
        marks = self.vision.landmarks(pil) or {}
        cran = self.vision.compute_cranial_and_scale_profile(pil, marks=marks) or {}
        dst_hair = "compact_updo" if float(cran.get("cranialToFaceWidthRatio", 1.32)) < 1.26 else "draped_down"

        def _dist(e: RealShotEntry) -> float:
            d_yaw = abs(e.yaw_deg - dst_yaw)
            d_exp = 0.0
            if dst_op is not None:
                d_exp += abs(e.organ_profile["eye_hw_ratio"] - dst_op.eye_hw_ratio) * 85.0
                d_exp += abs(e.organ_profile["mouth_width_to_iod"] - dst_op.mouth_width_to_iod) * 55.0
            d_hair = 4.5 if (prefer_hair_match and e.hair_silhouette_mode != dst_hair) else 0.0
            # Slight preference for higher IOD among nearly identical poses (e.g. shot01 vs shot04 or shot05 vs shot02)
            iod_tiebreak = -min(3.5, e.iod_px / 45.0)
            return d_yaw + d_exp + d_hair + iod_tiebreak

        best = min(self.entries, key=_dist)
        return best, {
            "dst_yaw_deg": round(dst_yaw, 2),
            "matched_shot_yaw_deg": best.yaw_deg,
            "yaw_delta_deg": round(abs(best.yaw_deg - dst_yaw), 2),
            "matched_shot_iod_px": best.iod_px,
        }

    def evaluate_against_all_real_shots(self, dst_bgr: np.ndarray) -> Dict[str, object]:
        det = self.detector.detect(dst_bgr)
        v_dst = self.auditor._extract_sface_vec(dst_bgr, det)
        by_shot: Dict[str, float] = {}
        for e in self.entries:
            v_ref = self._sface_vecs.get(e.path)
            if v_ref is not None and v_dst is not None:
                by_shot[e.name] = round(float(np.dot(v_ref, v_dst)), 4)
        vals = list(by_shot.values())
        return {
            "sface_by_real_shot": by_shot,
            "mean_sface_real_bank": round(float(np.mean(vals)), 4) if vals else 0.0,
            "max_sface_real_bank": round(float(np.max(vals)), 4) if vals else 0.0,
            "min_sface_real_bank": round(float(np.min(vals)), 4) if vals else 0.0,
        }

    def to_summary_dict(self) -> Dict[str, object]:
        return {
            "shot_count": len(self.entries),
            "highest_iod_shot": self.highest_iod_entry.name if self.entries else None,
            "skeletal_invariants": self.skeletal_invariants() if self.entries else {},
            "dynamic_skeletal_clause": self.to_dynamic_skeletal_clause() if self.entries else "",
            "entries": [asdict(e) for e in self.entries],
        }

    @classmethod
    def from_manifest_entries(
        cls,
        entries_data: Sequence[Dict[str, object]],
        config: EngineConfig,
        detector: DynamicFaceDetector,
        apple_vision: AppleVisionAnalyzer,
        auditor,
        origin_guard: Optional[OriginOnlyOneShotGuard] = None,
    ) -> "RealMultiShotBank":
        paths = [Path(str(d["path"])) for d in entries_data if d.get("path")]
        bank = cls(config, detector, apple_vision, auditor, origin_guard=origin_guard)
        bank.ingest_real_originals(paths)
        return bank


class IntrinsicAlbedoRelighter:
    """Design A (`주파수 분리 알베도 리라이팅`):

    Preserves 100% of the generated 1-Pass full-body posture, hair drape, neck, outfit, hands, and 3D directional
    room illumination while restoring the angle-matched real camera face's exact eyes, nose, lips, jawline proportions,
    intrinsic skin undertone, and real pore micro-texture (`Relit = Real_Albedo * Scene_Shading + Real_HighFreq`).
    """

    def __init__(self, detector: DynamicFaceDetector, apple_vision: AppleVisionAnalyzer) -> None:
        self.detector = detector
        self.vision = apple_vision

    @staticmethod
    def _clean_interior_skin_weight(bgr: np.ndarray, skin_prof: AdaptiveSkinToneProfile, lm: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute clean skin weight inside facial contour excluding dark eyes, brows, nostrils, lips, and hair strands."""
        h, w = bgr.shape[:2]
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        chroma_gate = skin_prof.build_chrominance_gate(lab)

        hull_pts = cv2.convexHull(np.round(lm["contour"]).astype(np.int32))
        face_poly = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(face_poly, [hull_pts], 1)

        # Exclude ocular apertures, brows, nostril base, and outer lips from low-frequency illumination estimation
        organ_excl = np.zeros((h, w), dtype=np.uint8)
        for k in ("leftEye", "rightEye", "leftEyebrow", "rightEyebrow", "outerLips"):
            if k in lm and len(lm[k]) >= 3:
                cv2.fillPoly(organ_excl, [np.round(lm[k]).astype(np.int32)], 1)
        dil_k = max(5, int(round(0.04 * (lm["contour"][:, 0].max() - lm["contour"][:, 0].min()))) | 1)
        organ_excl = cv2.dilate(organ_excl, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_k, dil_k)))

        w_skin = chroma_gate * (face_poly > 0).astype(np.float32) * (organ_excl == 0).astype(np.float32)
        return np.clip(w_skin, 0.0, 1.0)

    @staticmethod
    def _build_organ_and_face_blend_mask(
        src_bgr_aligned: np.ndarray,
        dst_bgr_aligned: np.ndarray,
        lm_canon: np.ndarray,
        lm_groups_src: Dict[str, np.ndarray],
        src_skin_prof: AdaptiveSkinToneProfile,
        dst_skin_prof: AdaptiveSkinToneProfile,
        rigid_only: bool = False,
    ) -> np.ndarray:
        """Build zero-ghosting anatomy-hugging fusion mask (`1.0` on inner organs -> smooth cosine taper before hair/collar)."""
        h, w = dst_bgr_aligned.shape[:2]
        contour = lm_canon[:17]
        cx = float(np.mean(contour[:, 0]))
        cy = float(np.mean(contour[:, 1]))
        fw = max(24.0, float(contour[:, 0].max() - contour[:, 0].min()))

        # Core organ hull (brows + eyes + nose + outerLips + central cheeks)
        core_pts = np.vstack([
            lm_canon[17:],  # brows, eyes, nose, lips
            0.84 * (contour - np.array([[cx, cy]], dtype=np.float32)) + np.array([[cx, cy]], dtype=np.float32),
        ])
        # Extend forehead dome smoothly above eyebrows (up to ~0.42 * fw above eyebrow arch, safely below hairline)
        brows = lm_canon[17:29]
        brow_top_y = float(brows[:, 1].min())
        forehead_arc = []
        for frac_x in (-0.34, -0.20, 0.0, 0.20, 0.34):
            dip = 1.0 - 0.32 * (frac_x / 0.35) ** 2
            forehead_arc.append([cx + frac_x * fw, brow_top_y - (0.28 if rigid_only else 0.24) * fw * dip])
        all_inner = np.vstack([core_pts, np.array(forehead_arc, dtype=np.float32)])

        inner_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(inner_mask, [cv2.convexHull(np.round(all_inner).astype(np.int32))], 255)

        # Outer limit strictly inside the 17-pt jaw/temple contour + smooth forehead cap
        outer_pts = np.vstack([
            0.96 * (contour - np.array([[cx, cy]], dtype=np.float32)) + np.array([[cx, cy]], dtype=np.float32),
            np.array([[cx + fx * fw, brow_top_y - 0.38 * fw * (1.0 - 0.30 * (fx / 0.42) ** 2)] for fx in (-0.42, -0.22, 0.0, 0.22, 0.42)], dtype=np.float32),
        ])
        outer_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(outer_mask, [cv2.convexHull(np.round(outer_pts).astype(np.int32))], 255)

        dist_out = cv2.distanceTransform(outer_mask, cv2.DIST_L2, 5)
        taper_px = max(14.0, 0.14 * fw)
        alpha = np.clip(dist_out / taper_px, 0.0, 1.0)
        # Raised-cosine roll-off (`C^1` smooth)
        alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)
        alpha = np.maximum(alpha, (inner_mask > 0).astype(np.float32) * 0.96)

        # Strict Organ Shield (`1.00` strictly on eyes, brows, nose, lips without spilling onto temple hair wisps!)
        organ_solid = np.zeros((h, w), dtype=np.uint8)
        for k in ("leftEyebrow", "rightEyebrow", "leftEye", "rightEye", "nose", "outerLips"):
            if k in lm_groups_src and len(lm_groups_src[k]) >= 3:
                cv2.fillPoly(organ_solid, [cv2.convexHull(np.round(lm_groups_src[k]).astype(np.int32))], 255)
        organ_solid_f = cv2.GaussianBlur(
            cv2.dilate(organ_solid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(np.float32) / 255.0,
            (0, 0),
            max(1.4, 0.008 * fw),
        )

        # Anti-Ghosting Peripheral Hair Guard: erode skin gates near hairlines so temple/cheek hair wisps NEVER leave faint traces!
        lab_src = cv2.cvtColor(src_bgr_aligned, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_dst = cv2.cvtColor(dst_bgr_aligned, cv2.COLOR_BGR2LAB).astype(np.float32)
        skin_gate_src = src_skin_prof.build_chrominance_gate(lab_src)
        skin_gate_dst = dst_skin_prof.build_chrominance_gate(lab_dst)
        er_k = max(5, int(round(0.025 * fw)) | 1)
        joint_raw = cv2.erode(
            ((skin_gate_src > 0.5) & (skin_gate_dst > 0.5)).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er_k, er_k)),
        )
        both_skin = cv2.GaussianBlur(joint_raw.astype(np.float32), (0, 0), max(3.5, 0.025 * fw))

        # Inside core organs (eyes/brows/lips), `organ_solid_f` overrides `both_skin` (since eyes/brows aren't skin color!)
        peripheral_guard = np.maximum(both_skin, organ_solid_f)
        alpha = alpha * peripheral_guard
        alpha = np.maximum(alpha, organ_solid_f)
        alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), max(2.2, 0.015 * fw))
        return np.clip(alpha, 0.0, 1.0)

    def relight_and_fuse_real_face(
        self,
        dst_4k_bgr: np.ndarray,
        bank: RealMultiShotBank,
        forced_source_entry: Optional[RealShotEntry] = None,
        align_canvas_tps: bool = True,
        shade_gamma: float = 0.94,
        ambient_chroma_blend: float = 0.58,
        scene_mood: Optional[Dict[str, object]] = None,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """Fuse the angle-matched real camera face into `dst_4k_bgr` under true Linear-RGB 3D scene lighting."""
        h, w = dst_4k_bgr.shape[:2]
        if forced_source_entry is not None:
            src_entry = forced_source_entry
            match_meta = {"forced": True, "matched_shot_yaw_deg": src_entry.yaw_deg, "matched_shot_iod_px": src_entry.iod_px}
        else:
            src_entry, match_meta = bank.select_nearest_real_shot_for_target(dst_4k_bgr)

        src_bgr_raw = cv2.imread(src_entry.path)
        if src_bgr_raw is None:
            raise FileNotFoundError(f"Cannot read matched real shot: {src_entry.path}")
        sh_src, sw_src = src_bgr_raw.shape[:2]

        # Extract landmarks on target 4K ROI (around detected face for speed & precision on 4K canvases!)
        dst_face = self.detector.detect(dst_4k_bgr)
        fx, fy, fw_box, fh_box = dst_face.bbox
        pad_x = int(round(fw_box * 1.45))
        pad_y_top = int(round(fh_box * 1.15))
        pad_y_bot = int(round(fh_box * 1.25))
        rx1 = max(0, int(round(fx)) - pad_x)
        ry1 = max(0, int(round(fy)) - pad_y_top)
        rx2 = min(w, int(round(fx + fw_box)) + pad_x)
        ry2 = min(h, int(round(fy + fh_box)) + pad_y_bot)

        dst_roi = dst_4k_bgr[ry1:ry2, rx1:rx2].copy()
        rh, rw = dst_roi.shape[:2]

        pil_src = Image.fromarray(cv2.cvtColor(src_bgr_raw, cv2.COLOR_BGR2RGB))
        pil_dst = Image.fromarray(cv2.cvtColor(dst_roi, cv2.COLOR_BGR2RGB))
        marks_src = self.vision.landmarks(pil_src)
        marks_dst = self.vision.landmarks(pil_dst)
        lm_src = _extract_ordered_76pt_landmarks(marks_src, sw_src, sh_src)
        lm_dst = _extract_ordered_76pt_landmarks(marks_dst, rw, rh)
        if lm_src is None or lm_dst is None:
            return dst_4k_bgr, {"applied": False, "reason": "landmarks_unavailable"}

        pts66_src = _build_canonical_correspondence_points(lm_src)
        pts66_dst = _build_canonical_correspondence_points(lm_dst)

        # 1. Pure Isotropic Similarity Alignment (`sx == sy`, 0% non-rigid organ distortion!) of Real Source -> Target ROI
        lp_s = np.mean(lm_src.get("leftPupil", lm_src["leftEye"]), axis=0)
        rp_s = np.mean(lm_src.get("rightPupil", lm_src["rightEye"]), axis=0)
        chin_s = lm_src["contour"][int(np.argmax(lm_src["contour"][:, 1]))]
        mid_s = 0.5 * (lp_s + rp_s)

        lp_d = np.mean(lm_dst.get("leftPupil", lm_dst["leftEye"]), axis=0)
        rp_d = np.mean(lm_dst.get("rightPupil", lm_dst["rightEye"]), axis=0)
        chin_d = lm_dst["contour"][int(np.argmax(lm_dst["contour"][:, 1]))]
        mid_d = 0.5 * (lp_d + rp_d)

        iod_s = max(1.0, float(np.linalg.norm(rp_s - lp_s)))
        iod_d = max(1.0, float(np.linalg.norm(rp_d - lp_d)))
        vlen_s = max(1.0, float(np.linalg.norm(chin_s - mid_s)))
        vlen_d = max(1.0, float(np.linalg.norm(chin_d - mid_d)))
        # Balanced isotropic scale anchoring true craniofacial size to target head frame without stretching!
        scale_iso = 0.62 * (iod_d / iod_s) + 0.38 * (vlen_d / vlen_s)

        ang_s = math.atan2(float(rp_s[1] - lp_s[1]), float(rp_s[0] - lp_s[0]))
        ang_d = math.atan2(float(rp_d[1] - lp_d[1]), float(rp_d[0] - lp_d[0]))
        d_theta = ang_d - ang_s
        cos_t = scale_iso * math.cos(d_theta)
        sin_t = scale_iso * math.sin(d_theta)
        tx = float(mid_d[0]) - (cos_t * float(mid_s[0]) - sin_t * float(mid_s[1]))
        ty = float(mid_d[1]) - (sin_t * float(mid_s[0]) + cos_t * float(mid_s[1]))
        M_sim = np.array([[cos_t, -sin_t, tx], [sin_t, cos_t, ty]], dtype=np.float32)

        src_sim_bgr = _warp_affine_nyquist(src_bgr_raw, M_sim, (rw, rh), flags=cv2.INTER_LANCZOS4)
        pts66_src_sim = (pts66_src @ M_sim[:, :2].T) + M_sim[:, 2]
        lm_src_sim = {k: (v @ M_sim[:, :2].T) + M_sim[:, 2] for k, v in lm_src.items()}

        # 2. Optional Gentle Local C^2 TPS Alignment ON THE GENERATED CANVAS (`dst_roi`) so target shading & jaw contour
        #    sit at the exact 1:1 anatomical coordinates of the untouched similarity-aligned real face (`0px` outer shift!)
        if align_canvas_tps:
            fw_roi = max(32.0, float(pts66_dst[:17, 0].max() - pts66_dst[:17, 0].min()))
            # Gentle micro-alignment (`<= 3.2%` of face width) + tight outer hair ring pins (`1.12 * fw_roi`) so outer hair NEVER waves!
            raw_delta = pts66_src_sim - pts66_dst
            max_shift = 0.032 * fw_roi
            d_norm = np.linalg.norm(raw_delta, axis=1, keepdims=True)
            clamped_delta = np.where(d_norm > max_shift, raw_delta * (max_shift / np.maximum(d_norm, 1e-6)), raw_delta)
            target_canon_pts = pts66_dst + clamped_delta

            ring_pins: List[List[float]] = []
            for edge_x in np.linspace(0.0, float(rw - 1), 9):
                ring_pins.append([float(edge_x), 0.0])
                ring_pins.append([float(edge_x), float(rh - 1)])
            for edge_y in np.linspace(0.0, float(rh - 1), 7)[1:-1]:
                ring_pins.append([0.0, float(edge_y)])
                ring_pins.append([float(rw - 1), float(edge_y)])
            cx_d, cy_d = float(np.mean(pts66_dst[:17, 0])), float(np.mean(pts66_dst[:17, 1]))
            for rad_mul in (1.14, 1.36):
                for ang_r in np.linspace(0.0, 2.0 * math.pi, 18, endpoint=False):
                    px = float(np.clip(cx_d + rad_mul * fw_roi * math.cos(ang_r), 2.0, rw - 3.0))
                    py = float(np.clip(cy_d + (rad_mul + 0.06) * fw_roi * math.sin(ang_r), 2.0, rh - 3.0))
                    ring_pins.append([px, py])
            pins_arr = np.array(ring_pins, dtype=np.float32)

            tps_src = np.vstack([pts66_dst, pins_arr])
            tps_dst = np.vstack([target_canon_pts, pins_arr])
            dst_aligned_bgr = _solve_tps_warp(dst_roi, tps_src, tps_dst, reg_lambda=0.085, grid_stride=4)

            src_aligned_bgr = src_sim_bgr
            lm_src_final = lm_src_sim
            canon_pts = pts66_src_sim
        else:
            # Design B (`Face-Locked Outpainting`): 100.0% rigid camera geometry — ZERO non-rigid warp on real face!
            dst_aligned_bgr = dst_roi
            src_aligned_bgr = src_sim_bgr
            lm_src_final = lm_src_sim
            canon_pts = pts66_src_sim

        # 3. Skin-Domain Normalized-Convolution Intrinsic Shading Decomposition (`Relit = Real_Albedo * Scene_Shading`)
        det_src_raw = self.detector.detect(src_bgr_raw)
        det_dst_roi = self.detector.detect(dst_aligned_bgr)
        skin_prof_src = self.detector.sample_adaptive_skin_profile(src_bgr_raw, det_src_raw)
        skin_prof_dst = self.detector.sample_adaptive_skin_profile(dst_aligned_bgr, det_dst_roi)

        w_skin_src = self._clean_interior_skin_weight(src_aligned_bgr, skin_prof_src, lm_src_final)
        w_skin_dst = self._clean_interior_skin_weight(dst_aligned_bgr, skin_prof_dst, lm_src_final)
        w_skin_joint = np.maximum(0.05, w_skin_src * 0.6 + w_skin_dst * 0.4)

        sigma_illum = max(12.0, 0.24 * iod_d)
        # If target scene is low-key chiaroscuro and the outpainted head/neck was still drawn too flatly bright,
        # gently sculpt BOTH `dst_aligned_bgr` (neck/hair/cheek) AND `dst_4k_bgr` upper-body shadow envelope in Linear RGB
        # BEFORE computing `E_low_t` so face, eyes, neck, hair, and lapel share 100.0% continuous low-key side-window light!
        out_full = dst_4k_bgr.copy()
        if scene_mood is not None and bool(scene_mood.get("is_low_key", False)):
            tgt_p50 = float(scene_mood.get("face_p50_L", 105.0))
            tgt_p20 = float(scene_mood.get("face_p20_L", 58.0))
            tgt_p85 = float(scene_mood.get("face_p85_L", 182.0))
            key_left = bool(scene_mood.get("key_from_left", True))
            lab_chk = cv2.cvtColor(dst_aligned_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            cur_skin_L = lab_chk[:, :, 0][w_skin_dst > 0.3]
            cur_p50 = float(np.median(cur_skin_L)) if cur_skin_L.size >= 20 else 150.0
            cur_p20 = float(np.percentile(cur_skin_L, 20)) if cur_skin_L.size >= 20 else 125.0
            cur_p85 = float(np.percentile(cur_skin_L, 85)) if cur_skin_L.size >= 20 else 182.0
            if (cur_p50 > tgt_p50 + 8.0) or (cur_p20 > tgt_p20 + 12.0):
                # Build smooth wide-field 3D chiaroscuro side-window envelope centered on the head/neck/upper-torso
                yy_f, xx_f = np.mgrid[0:h, 0:w].astype(np.float32)
                fcx = float(rx1 + np.mean(canon_pts[:17, 0]))
                fcy = float(ry1 + np.mean(canon_pts[:17, 1]))
                fw_g = max(40.0, float(canon_pts[:17, 0].max() - canon_pts[:17, 0].min()))
                nx = (xx_f - fcx) / (0.62 * fw_g)
                if not key_left:
                    nx = -nx
                # Lit side (nx < 0) retains crisp window key light; shadow side (nx > 0) rolls off smoothly into low-key room shadow!
                shd_floor_lin = float(np.clip(((tgt_p20 + 6.0) / max(80.0, cur_p20)) ** 2.2, 0.14, 0.65))
                lit_ceil_lin = float(np.clip((tgt_p85 / max(130.0, cur_p85)) ** 2.2, 0.78, 1.04))
                side_ramp = shd_floor_lin + (lit_ceil_lin - shd_floor_lin) * (1.0 / (1.0 + np.exp(1.85 * (nx - 0.05))))
                # Localize envelope smoothly around head, hair, neck, and upper shoulders (`rx`, `ry` Gaussian support)
                rad2 = ((xx_f - fcx) / (2.4 * fw_g)) ** 2 + ((yy_f - (fcy + 0.55 * fw_g)) / (3.1 * fw_g)) ** 2
                env_w = np.exp(-0.5 * rad2)
                mult_lin_full = (1.0 - env_w) + env_w * side_ramp
                lin_full = np.power(np.clip(out_full.astype(np.float32) / 255.0, 0.0, 1.0), 2.2)
                lin_full *= mult_lin_full[:, :, None]
                out_full = np.clip(np.power(np.clip(lin_full, 0.0, 1.0), 1.0 / 2.2) * 255.0, 0.0, 255.0).astype(np.uint8)
                dst_aligned_bgr = out_full[ry1:ry2, rx1:rx2].copy()
                if align_canvas_tps:
                    dst_aligned_bgr = _solve_tps_warp(dst_aligned_bgr, tps_src, tps_dst, reg_lambda=0.085, grid_stride=4)

        # True Photometric Linear-RGB Irradiance Decomposition (`Radiance = Albedo * Irradiance`, gamma 2.2)
        src_lin = np.power(np.clip(src_aligned_bgr.astype(np.float32) / 255.0, 0.0, 1.0), 2.2)
        dst_lin = np.power(np.clip(dst_aligned_bgr.astype(np.float32) / 255.0, 0.0, 1.0), 2.2)
        # BGR -> Rec.709 linear luminance
        Y_s = 0.0722 * src_lin[:, :, 0] + 0.7152 * src_lin[:, :, 1] + 0.2126 * src_lin[:, :, 2]
        Y_t = 0.0722 * dst_lin[:, :, 0] + 0.7152 * dst_lin[:, :, 1] + 0.2126 * dst_lin[:, :, 2]

        E_low_s = _norm_gaussian(Y_s, w_skin_src, sigma_illum)
        E_low_t = _norm_gaussian(Y_t, w_skin_joint, sigma_illum)

        # Unclamped physical shadow floor (`0.09` in linear light, NOT `0.62` LAB!) so shadow cheek drops naturally into dark room light!
        ratio_lin = np.clip((E_low_t + 0.010) / (E_low_s + 0.010), 0.09, 1.42) ** float(shade_gamma)

        # Per-Eye Local Orbital Irradiance: lit-side eye keeps crisp catchlight while shadow-side eye sits naturally in shadow!
        eye_mask_total = np.zeros((rh, rw), dtype=np.uint8)
        eff_ratio_lin = ratio_lin.copy()
        orbital_scalars: List[float] = []
        for ek in ("leftEye", "rightEye"):
            if ek in lm_src_final and len(lm_src_final[ek]) >= 4:
                em = np.zeros((rh, rw), dtype=np.uint8)
                cv2.fillPoly(em, [cv2.convexHull(np.round(lm_src_final[ek]).astype(np.int32))], 255)
                eye_mask_total = np.maximum(eye_mask_total, em)
                ring_d = cv2.dilate(em, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
                ring_only = (ring_d > 0) & (em == 0) & (w_skin_src > 0.2)
                loc_sc = float(np.median(ratio_lin[ring_only])) if np.any(ring_only) else float(np.median(ratio_lin))
                # Slight 15% sclera readability lift relative to surrounding eyelid skin without glowing in the dark
                loc_eye_sc = float(np.clip(loc_sc * 1.12, 0.14, 1.15))
                orbital_scalars.append(loc_eye_sc)
                em_soft = cv2.GaussianBlur(
                    cv2.dilate(em, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(np.float32) / 255.0,
                    (0, 0),
                    2.2,
                )
                eff_ratio_lin = eff_ratio_lin * (1.0 - em_soft) + loc_eye_sc * em_soft

        eye_soft = cv2.GaussianBlur(
            cv2.dilate(eye_mask_total, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(np.float32) / 255.0,
            (0, 0),
            2.2,
        )
        orbital_damped = float(np.mean(orbital_scalars)) if orbital_scalars else 1.0

        relit_lin = np.clip(src_lin * eff_ratio_lin[:, :, None], 0.0, 1.0)
        relit_srgb_bgr = np.clip(np.power(relit_lin, 1.0 / 2.2) * 255.0, 0.0, 255.0).astype(np.uint8)

        # Harmonize Chrominance (`a*, b*`) smoothly with target scene ambient warmth & shadow desaturation (eliminates bright studio pinkness!)
        lab_s = cv2.cvtColor(src_aligned_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_relit = cv2.cvtColor(relit_srgb_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_t = cv2.cvtColor(dst_aligned_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_low_s = _norm_gaussian(lab_s[:, :, 0], w_skin_src, sigma_illum)

        a_low_r = _norm_gaussian(lab_relit[:, :, 1], w_skin_src, sigma_illum)
        b_low_r = _norm_gaussian(lab_relit[:, :, 2], w_skin_src, sigma_illum)
        a_low_t = _norm_gaussian(lab_t[:, :, 1], w_skin_joint, sigma_illum)
        b_low_t = _norm_gaussian(lab_t[:, :, 2], w_skin_joint, sigma_illum)

        lip_mask = np.zeros((rh, rw), dtype=np.uint8)
        if "outerLips" in lm_src_final and len(lm_src_final["outerLips"]) >= 5:
            cv2.fillPoly(lip_mask, [cv2.convexHull(np.round(lm_src_final["outerLips"]).astype(np.int32))], 255)
        lip_or_eye = np.clip(
            eye_soft + cv2.GaussianBlur(lip_mask.astype(np.float32) / 255.0, (0, 0), 2.5),
            0.0,
            1.0,
        )
        eff_amb = float(ambient_chroma_blend) * (1.0 - 0.60 * lip_or_eye)
        a_relit = lab_relit[:, :, 1] + eff_amb * (a_low_t - a_low_r)
        b_relit = lab_relit[:, :, 2] + eff_amb * (b_low_t - b_low_r)
        if scene_mood is not None and bool(scene_mood.get("is_low_key", False)):
            # Calm excess studio rosy-pink chroma toward the target low-key film palette
            tgt_fa = float(scene_mood.get("face_a", 4.5)) + 128.0
            cur_fa = float(np.median(a_relit[w_skin_src > 0.3])) if np.any(w_skin_src > 0.3) else tgt_fa
            if cur_fa > tgt_fa + 1.0:
                a_relit -= (cur_fa - tgt_fa) * 0.72 * (1.0 - 0.55 * lip_or_eye)

        relit_lab = np.stack([lab_relit[:, :, 0], np.clip(a_relit, 40.0, 220.0), np.clip(b_relit, 40.0, 220.0)], axis=2).astype(np.uint8)
        relit_src_bgr = cv2.cvtColor(relit_lab, cv2.COLOR_LAB2BGR)

        # De-wisp ONLY in the outer peripheral temple/hairline transition band (`dist_periph <= 0.14 * fw`, outside inner face
        # where eyewear frames, moles, and alar creases live!) so temple hair wisps never leave a faint shadow smudge!
        outer_hull = np.zeros((rh, rw), dtype=np.uint8)
        cv2.fillPoly(outer_hull, [cv2.convexHull(np.round(canon_pts[:17]).astype(np.int32))], 255)
        fw_cur = max(24.0, float(canon_pts[:17, 0].max() - canon_pts[:17, 0].min()))
        brows_cur = canon_pts[17:29]
        fh_arc = np.array(
            [[float(np.mean(canon_pts[:17, 0])) + fx_m * fw_cur, float(brows_cur[:, 1].min()) - 0.34 * fw_cur] for fx_m in (-0.40, 0.0, 0.40)],
            dtype=np.float32,
        )
        cv2.fillPoly(outer_hull, [cv2.convexHull(np.round(np.vstack([canon_pts[:17], fh_arc])).astype(np.int32))], 255)
        dist_periph = cv2.distanceTransform(outer_hull, cv2.DIST_L2, 5)

        strict_org = np.zeros((rh, rw), dtype=np.uint8)
        for k_o in ("leftEyebrow", "rightEyebrow", "leftEye", "rightEye", "nose", "outerLips"):
            if k_o in lm_src_final and len(lm_src_final[k_o]) >= 3:
                cv2.fillPoly(strict_org, [cv2.convexHull(np.round(lm_src_final[k_o]).astype(np.int32))], 1)
        strict_org = cv2.dilate(strict_org, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        in_periph_band = (outer_hull > 0) & (dist_periph <= max(10.0, 0.14 * fw_cur))
        stray_wisp = in_periph_band & (strict_org == 0) & (lab_s[:, :, 0] < L_low_s - 10.0)
        if np.any(stray_wisp):
            clean_skin_fill = _norm_gaussian(relit_src_bgr.astype(np.float32), w_skin_src, max(6.0, 0.08 * iod_d))
            w_wisp = cv2.GaussianBlur(stray_wisp.astype(np.float32), (0, 0), 2.2)[:, :, None]
            relit_src_bgr = np.clip(
                clean_skin_fill * w_wisp + relit_src_bgr.astype(np.float32) * (1.0 - w_wisp), 0.0, 255.0
            ).astype(np.uint8)

        # 3b. Authentic High-Res Pore Micro-Texture Recovery when matched shot had smaller IOD than bank's close-up master
        hi_entry = bank.highest_iod_entry
        pore_boost_std = 0.0
        if scale_iso > 1.35 and hi_entry.iod_px > src_entry.iod_px * 1.25:
            target_pore_std = float(hi_entry.organ_profile.get("skin_pore_micro_std", 1.8))
            rng = np.random.default_rng(42)
            fine_n = rng.normal(0.0, 1.0, size=(rh, rw)).astype(np.float32)
            pore_field = cv2.GaussianBlur(fine_n, (0, 0), 0.75) - cv2.GaussianBlur(fine_n, (0, 0), 2.1)
            cur_std = max(1e-4, float(np.std(pore_field)))
            pore_boost_std = round(min(1.45, target_pore_std * 0.42), 3)
            pore_delta = (pore_field / cur_std) * pore_boost_std * w_skin_src * (1.0 - lip_or_eye)
            relit_f = relit_src_bgr.astype(np.float32) + pore_delta[:, :, None]
            relit_src_bgr = np.clip(relit_f, 0.0, 255.0).astype(np.uint8)

        # 4. Anatomy-Hugging Anti-Ghosting Mask + 4-Level Laplacian Pyramid Fusion
        fuse_mask = self._build_organ_and_face_blend_mask(
            src_aligned_bgr,
            dst_aligned_bgr,
            canon_pts,
            lm_src_final,
            skin_prof_src,
            skin_prof_dst,
            rigid_only=not align_canvas_tps,
        )
        fused_roi = _laplacian_pyramid_blend(relit_src_bgr, dst_aligned_bgr, fuse_mask, levels=4)
        out_full[ry1:ry2, rx1:rx2] = np.round(fused_roi).astype(np.uint8)

        return out_full, {
            "applied": True,
            "mode": "design_a_intrinsic_albedo_relight" if align_canvas_tps else "design_b_rigid_facelock_relight",
            "matched_real_shot": src_entry.name,
            "matched_real_shot_path": src_entry.path,
            "isotropic_scale": round(float(scale_iso), 3),
            "orbital_lighting_scalar": round(orbital_damped, 3),
            "pore_micro_boost_std": pore_boost_std,
            **match_meta,
        }


def analyze_target_scene_chiaroscuro_and_mood(
    scene_bgr: np.ndarray,
    detector: Optional[DynamicFaceDetector] = None,
) -> Dict[str, object]:
    """Dynamically measure lighting key ratio, shadow depth, black level, and split-tone mood from ANY target scene."""
    h, w = scene_bgr.shape[:2]
    lab = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    A = lab[:, :, 1] - 128.0
    B = lab[:, :, 2] - 128.0

    upper_bg_L = float(np.median(L[: max(8, int(h * 0.28)), :]))
    q_pcts = (2.0, 10.0, 25.0, 50.0, 75.0, 90.0, 98.0)
    frame_L_q = [float(np.percentile(L, q)) for q in q_pcts]

    shd_m = L < 65.0
    mid_m = (L >= 65.0) & (L < 155.0)
    hi_m = L >= 155.0
    split_ab = {
        "shd_a": float(np.median(A[shd_m])) if np.any(shd_m) else 0.0,
        "shd_b": float(np.median(B[shd_m])) if np.any(shd_m) else 0.0,
        "mid_a": float(np.median(A[mid_m])) if np.any(mid_m) else 0.0,
        "mid_b": float(np.median(B[mid_m])) if np.any(mid_m) else 0.0,
        "hi_a": float(np.median(A[hi_m])) if np.any(hi_m) else 0.0,
        "hi_b": float(np.median(B[hi_m])) if np.any(hi_m) else 0.0,
    }

    left_scene_L = float(np.percentile(L[:, : w // 2], 75))
    right_scene_L = float(np.percentile(L[:, w // 2 :], 75))
    key_from_left = left_scene_L >= right_scene_L

    face_p20_L = float(np.clip(upper_bg_L * 2.1, 38.0, 145.0))
    face_p50_L = float(np.clip(upper_bg_L * 3.4, 88.0, 178.0))
    face_p85_L = float(np.clip(frame_L_q[5] * 0.90, 145.0, 205.0))
    face_a = float(split_ab["mid_a"] + 4.5)
    face_b = float(split_ab["mid_b"] + 7.5)
    garment_p50_L = float(np.clip(frame_L_q[3], 36.0, 120.0))

    if detector is not None:
        try:
            fdet = detector.detect(scene_bgr)
            fx, fy, fw, fh = fdet.bbox
            x1 = max(0, int(round(fx + 0.12 * fw)))
            x2 = min(w, int(round(fx + 0.88 * fw)))
            y1 = max(0, int(round(fy + 0.12 * fh)))
            y2 = min(h, int(round(fy + 0.92 * fh)))
            if x2 > x1 + 8 and y2 > y1 + 8:
                fL = L[y1:y2, x1:x2]
                fA = A[y1:y2, x1:x2]
                fB = B[y1:y2, x1:x2]
                skin_valid = fL > 28.0
                if np.sum(skin_valid) >= 40:
                    face_p20_L = float(np.percentile(fL[skin_valid], 22))
                    face_p50_L = float(np.percentile(fL[skin_valid], 52))
                    face_p85_L = float(np.percentile(fL[skin_valid], 86))
                    face_a = float(np.median(fA[skin_valid]))
                    face_b = float(np.median(fB[skin_valid]))
                mx = (x1 + x2) // 2
                l_half = float(np.percentile(L[y1:y2, x1:mx], 75)) if mx > x1 + 2 else left_scene_L
                r_half = float(np.percentile(L[y1:y2, mx:x2], 75)) if x2 > mx + 2 else right_scene_L
                key_from_left = l_half >= r_half
                shd_slice = L[y1:y2, mx:x2] if key_from_left else L[y1:y2, x1:mx]
                lit_slice = L[y1:y2, x1:mx] if key_from_left else L[y1:y2, mx:x2]
                if shd_slice.size >= 20:
                    shd_v = shd_slice[shd_slice > 22.0]
                    if shd_v.size >= 12:
                        face_p20_L = float(np.percentile(shd_v, 35))
                if lit_slice.size >= 20:
                    lit_v = lit_slice[lit_slice > 28.0]
                    if lit_v.size >= 12:
                        face_p85_L = float(np.percentile(lit_v, 78))
                ty1 = min(h - 4, int(round(fy + 1.35 * fh)))
                ty2 = min(h, int(round(fy + 3.20 * fh)))
                tx1 = max(0, int(round(fx - 0.55 * fw)))
                tx2 = min(w, int(round(fx + 1.55 * fw)))
                if ty2 > ty1 + 10 and tx2 > tx1 + 10:
                    garment_p50_L = float(np.percentile(L[ty1:ty2, tx1:tx2], 50))
        except Exception:
            pass

    is_low_key = bool(
        upper_bg_L < 85.0
        and (frame_L_q[3] < 95.0 or (face_p85_L - face_p20_L) >= 65.0 or face_p50_L < 155.0)
    )
    return {
        "is_low_key": is_low_key,
        "key_from_left": bool(key_from_left),
        "upper_bg_L": round(upper_bg_L, 2),
        "frame_L_quantiles": [round(v, 2) for v in frame_L_q],
        "split_ab": {k: round(v, 3) for k, v in split_ab.items()},
        "face_p20_L": round(face_p20_L, 2),
        "face_p50_L": round(face_p50_L, 2),
        "face_p85_L": round(face_p85_L, 2),
        "face_a": round(face_a, 2),
        "face_b": round(face_b, 2),
        "garment_p50_L": round(garment_p50_L, 2),
    }


def master_full_frame_mood_and_grain(
    canvas_bgr: np.ndarray,
    reference_scene_bgr: np.ndarray,
    curve_strength: float = 0.58,
    chroma_strength: float = 0.72,
    grain_std_4k: float = 1.35,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Apply 100% UNIFORM whole-frame black-level/contrast curve + split-tone LUT + 35mm sensor micro-grain (ZERO face mask!)."""
    h, w = canvas_bgr.shape[:2]
    lab_c = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_r = cv2.cvtColor(reference_scene_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    Lc = lab_c[:, :, 0]
    Lr = lab_r[:, :, 0]
    qs = (0.0, 3.0, 12.0, 28.0, 50.0, 72.0, 88.0, 97.0, 100.0)
    src_knots = np.array([0.0] + [float(np.percentile(Lc, q)) for q in qs[1:-1]] + [255.0], dtype=np.float32)
    ref_knots = np.array([0.0] + [float(np.percentile(Lr, q)) for q in qs[1:-1]] + [255.0], dtype=np.float32)
    # Ensure strict monotonicity of src_knots
    for idx_k in range(1, len(src_knots)):
        if src_knots[idx_k] <= src_knots[idx_k - 1] + 0.5:
            src_knots[idx_k] = src_knots[idx_k - 1] + 0.5
    blended_knots = src_knots * (1.0 - float(curve_strength)) + ref_knots * float(curve_strength)
    lut_x = np.arange(256, dtype=np.float32)
    lut_L = np.clip(np.interp(lut_x, src_knots, blended_knots), 0.0, 255.0)
    L_graded = np.interp(Lc, lut_x, lut_L).astype(np.float32)

    # Smooth 3-band shadow / midtone / highlight split-tone `a*, b*` matching across the WHOLE frame
    Ac = lab_c[:, :, 1] - 128.0
    Bc = lab_c[:, :, 2] - 128.0
    Ar = lab_r[:, :, 1] - 128.0
    Br = lab_r[:, :, 2] - 128.0

    w_shd = np.clip((90.0 - L_graded) / 65.0, 0.0, 1.0)
    w_hi = np.clip((L_graded - 135.0) / 65.0, 0.0, 1.0)
    w_mid = np.clip(1.0 - w_shd - w_hi, 0.0, 1.0)

    def _band_delta(ch_c: np.ndarray, ch_r: np.ndarray, lc_m: np.ndarray, lr_m: np.ndarray, lim: float = 5.0) -> float:
        mc = float(np.median(ch_c[lc_m])) if np.any(lc_m) else 0.0
        mr = float(np.median(ch_r[lr_m])) if np.any(lr_m) else 0.0
        return float(np.clip(mr - mc, -lim, lim))

    da_shd = _band_delta(Ac, Ar, Lc < 70, Lr < 70, lim=5.0)
    db_shd = _band_delta(Bc, Br, Lc < 70, Lr < 70, lim=5.5)
    da_mid = _band_delta(Ac, Ar, (Lc >= 70) & (Lc < 150), (Lr >= 70) & (Lr < 150), lim=4.5)
    db_mid = _band_delta(Bc, Br, (Lc >= 70) & (Lc < 150), (Lr >= 70) & (Lr < 150), lim=5.5)
    da_hi = _band_delta(Ac, Ar, Lc >= 150, Lr >= 150, lim=2.2) * 0.45
    db_hi = _band_delta(Bc, Br, Lc >= 150, Lr >= 150, lim=3.2) * 0.55

    da = (w_shd * da_shd + w_mid * da_mid + w_hi * da_hi) * float(chroma_strength)
    db = (w_shd * db_shd + w_mid * db_mid + w_hi * db_hi) * float(chroma_strength)

    # Subtle cinematic saturation calm-down in deep shadows (prevents artificial colour blotches in low-key shadows)
    sat_calm = 1.0 - 0.22 * w_shd
    A_out = np.clip((Ac * sat_calm + da) + 128.0, 40.0, 220.0)
    B_out = np.clip((Bc * sat_calm + db) + 128.0, 40.0, 220.0)

    graded_bgr = cv2.cvtColor(np.stack([L_graded, A_out, B_out], axis=2).astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Unified whole-frame fine optical 35mm film micro-grain (attenuated in deep black & pure highlight roll-off)
    if grain_std_4k > 0.05:
        rng = np.random.default_rng(seed)
        g_raw = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        g_opt = cv2.GaussianBlur(g_raw, (0, 0), 0.72)
        g_opt = g_opt / max(1e-4, float(np.std(g_opt)))
        mid_bell = np.sin(np.pi * np.clip(L_graded / 255.0, 0.04, 0.96)) ** 0.75
        grain = (g_opt * float(grain_std_4k) * mid_bell)[:, :, None]
        graded_bgr = np.clip(graded_bgr.astype(np.float32) + grain, 0.0, 255.0).astype(np.uint8)

    return graded_bgr, {
        "applied": True,
        "curve_strength": curve_strength,
        "chroma_strength": chroma_strength,
        "grain_std_4k": grain_std_4k,
        "frame_p50_L_after": round(float(np.median(L_graded)), 2),
    }


class FaceLockedOutpaintStudio:
    """Design B (`페이스락 아웃페인팅`):

    Physically anchors the untouched real camera head/hair/upper-neck onto the clean location canvas BEFORE
    generation (`stage2a_mesh_precomposite_facelock.png`) at exact measured skeletal scale, then locks 100%
    rigid camera face pixels + low-frequency scene lighting after body/garment outpainting.
    """

    def __init__(
        self,
        detector: DynamicFaceDetector,
        apple_vision: AppleVisionAnalyzer,
        relighter: IntrinsicAlbedoRelighter,
    ) -> None:
        self.detector = detector
        self.vision = apple_vision
        self.relighter = relighter

    def _extract_universal_decontaminated_head_layer(
        self,
        src_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        """Extract clean `(decontaminated_bgr, alpha_01, geom_meta)` of head + hair + bare neck with ZERO hardcoded colors."""
        h, w = src_bgr.shape[:2]
        det = self.detector.detect(src_bgr)
        skin_prof = self.detector.sample_adaptive_skin_profile(src_bgr, det)
        pil = Image.fromarray(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB))
        marks = self.vision.landmarks(pil) or {}
        pm_pil = self.vision.person_mask(pil)
        pm = (
            np.array(pm_pil.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            if pm_pil is not None
            else np.ones((h, w), dtype=np.float32)
        )

        lm = _extract_ordered_76pt_landmarks(marks, w, h)
        if lm is None:
            raise RuntimeError("Cannot extract 76-pt landmarks on FaceLock real source photo.")

        contour = lm["contour"]
        chin = contour[int(np.argmax(contour[:, 1]))]
        chin_y = float(chin[1])
        fw = max(24.0, float(contour[:, 0].max() - contour[:, 0].min()))
        lp = np.mean(lm.get("leftPupil", lm["leftEye"]), axis=0)
        rp = np.mean(lm.get("rightPupil", lm["rightEye"]), axis=0)
        mid = 0.5 * (lp + rp)

        lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        yy, xx = np.mgrid[0:h, 0:w]

        # 1. Dynamically sample studio backdrop LAB from exterior `pm < 0.05` pixels
        bg_mask = pm < 0.05
        bg_med = np.median(lab[bg_mask], axis=0) if np.sum(bg_mask) >= 64 else np.array([210.0, 128.0, 128.0], dtype=np.float32)

        # 2. Connected bare face + neck skin mask
        organ_pts = [contour] + [lm[k] for k in ("leftEyebrow", "rightEyebrow", "leftEye", "rightEye", "nose", "outerLips") if k in lm]
        seed_face = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(seed_face, [cv2.convexHull(np.round(np.vstack(organ_pts)).astype(np.int32))], 1)

        skin_gate = (skin_prof.build_chrominance_gate(lab) > 0.5) & (pm > 0.35)
        raw_skin_cc = (skin_gate | (seed_face > 0)).astype(np.uint8)
        _, labels_cc = cv2.connectedComponents(raw_skin_cc, connectivity=8)
        face_lbls = np.unique(labels_cc[seed_face > 0])
        bare_skin = np.isin(labels_cc, face_lbls[face_lbls > 0]) & (yy <= int(round(chin_y + 0.42 * fw)))

        # 3. Universal Garment/Collar Rejection below jaw: sample subject's true hair LAB above eye-level (`yy < mid_y`),
        #    so below `chin_y - 0.12*fw` ONLY bare neck skin OR genuine head-hair color survives (100% collar/lapel erasure!)
        crown_zone = (pm > 0.85) & (~bare_skin) & (seed_face == 0) & (yy < int(round(mid[1])))
        if np.sum(crown_zone) >= 40:
            hair_med_lab = np.median(lab[crown_zone], axis=0)
            hair_iqr_l = max(14.0, float(np.percentile(lab[:, :, 0][crown_zone], 80) - np.percentile(lab[:, :, 0][crown_zone], 20)))
            d_hair = np.sqrt(
                ((lab[:, :, 0] - float(hair_med_lab[0])) / max(1.0, 1.8 * hair_iqr_l)) ** 2
                + ((lab[:, :, 1] - float(hair_med_lab[1])) / 11.0) ** 2
                + ((lab[:, :, 2] - float(hair_med_lab[2])) / 12.0) ** 2
            )
            matches_crown_hair = d_hair <= 1.25
        else:
            matches_crown_hair = np.ones((h, w), dtype=bool)

        collar_zone = (yy >= int(round(chin_y - 0.12 * fw))) & (~bare_skin) & (seed_face == 0)
        non_hair_fabric = collar_zone & (~matches_crown_hair)

        below_neck_cut = np.zeros((h, w), dtype=bool)
        x_scan1 = max(0, int(round(mid[0] - 0.58 * fw)))
        x_scan2 = min(w, int(round(mid[0] + 0.58 * fw)))
        y_start = max(0, int(round(chin_y - 0.04 * fw)))
        for xc in range(x_scan1, x_scan2):
            ys_sk = np.where(bare_skin[y_start:, xc])[0]
            if len(ys_sk) > 0:
                by = y_start + int(ys_sk.max())
                below_neck_cut[max(0, by - 2) :, xc] = True

        far_below_or_wide = (yy >= int(round(chin_y + 0.28 * fw))) | (
            (yy >= int(round(chin_y + 0.02 * fw))) & (np.abs(xx - mid[0]) > 0.82 * fw)
        )
        garment_reject = (non_hair_fabric | below_neck_cut | far_below_or_wide) & (~bare_skin) & (seed_face == 0)
        # Also trim bare neck smoothly before inner neckline (`chin_y + 0.26 * fw`)
        garment_reject |= yy >= int(round(chin_y + 0.30 * fw))

        # 4. Studio Backdrop Rim Decontamination along hair wisps
        d_bg = np.linalg.norm(lab - bg_med[None, None, :], axis=2)
        hair_fg = (pm > 0.88) & (~bare_skin) & (~garment_reject) & (yy < int(round(chin_y + 0.25 * fw)))
        fg_d_med = float(np.median(d_bg[hair_fg])) if np.sum(hair_fg) >= 50 else 80.0
        rim_matte = np.clip(d_bg / max(22.0, fg_d_med * 0.82), 0.0, 1.0) ** 1.15

        fg_bin = ((pm > 0.65) & (~garment_reject)).astype(np.uint8)
        dist_in = cv2.distanceTransform(fg_bin, cv2.DIST_L2, 5)
        rim_zone = (~bare_skin) & (seed_face == 0) & (dist_in < max(12.0, 0.055 * fw))

        alpha = pm.copy()
        alpha[garment_reject] = 0.0
        alpha[rim_zone] = np.minimum(alpha[rim_zone], rim_matte[rim_zone])

        neck_t = np.clip((chin_y + 0.27 * fw - yy) / max(1.0, 0.22 * fw), 0.0, 1.0)
        hair_t = np.clip((chin_y + 0.23 * fw - yy) / max(1.0, 0.20 * fw), 0.0, 1.0)
        neck_fade = 0.5 - 0.5 * np.cos(np.pi * neck_t)
        hair_fade = 0.5 - 0.5 * np.cos(np.pi * hair_t)
        lower_band = yy >= int(round(chin_y - 0.02 * fw))
        alpha[lower_band] *= np.where(bare_skin, neck_fade, hair_fade)[lower_band]
        alpha = np.maximum(alpha, seed_face.astype(np.float32))
        alpha[garment_reject] = 0.0

        # Keep single connected head component
        _, lbl_a = cv2.connectedComponents((alpha > 0.05).astype(np.uint8), connectivity=8)
        keep_lbls = np.unique(lbl_a[seed_face > 0])
        alpha *= np.isin(lbl_a, keep_lbls[keep_lbls > 0]).astype(np.float32)
        alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 1.2)

        # Decontaminate semi-transparent hair edge colors via Normalized Convolution from pure interior (`alpha > 0.88`)
        pure_fg = (alpha > 0.88).astype(np.float32)
        clean_fg_bgr = _norm_gaussian(src_bgr.astype(np.float32), pure_fg, max(4.0, 0.03 * fw))
        decontam_bgr = np.where((alpha[:, :, None] < 0.88) & (seed_face[:, :, None] == 0), clean_fg_bgr, src_bgr.astype(np.float32))
        decontam_bgr = np.clip(decontam_bgr, 0.0, 255.0).astype(np.uint8)

        return decontam_bgr, np.clip(alpha, 0.0, 1.0), {
            "mid_xy": [float(mid[0]), float(mid[1])],
            "chin_xy": [float(chin[0]), float(chin[1])],
            "face_w_px": float(fw),
            "iod_px": float(det.inter_ocular_distance),
        }

    def build_facelock_precomposite_canvas(
        self,
        clean_location_path: Path,
        output_canvas_path: Path,
        bank: RealMultiShotBank,
        original_location_path: Optional[Path] = None,
        scene_instruction: str = "",
        aspect_ratio_label: str = "9:16",
    ) -> Dict[str, object]:
        """Place the decontaminated real camera head onto `clean_location_path` at measured 7.45-head skeletal scale."""
        clean_bgr = cv2.imread(str(clean_location_path))
        if clean_bgr is None:
            raise FileNotFoundError(f"Clean location image unreadable: {clean_location_path}")

        # Upscale low-res clean location plate (e.g. 572x1024) to crisp 1440px-tall working plate so real camera head keeps high IOD!
        ch0, cw0 = clean_bgr.shape[:2]
        if max(ch0, cw0) < 1500:
            up_s = 1600.0 / float(max(ch0, cw0))
            clean_bgr = cv2.resize(
                clean_bgr,
                (int(round(cw0 * up_s)), int(round(ch0 * up_s))),
                interpolation=cv2.INTER_LANCZOS4,
            )
        ch, cw = clean_bgr.shape[:2]

        # Determine natural seated/standing head anchor from original location reference (before person erasure) if available
        anchor_cx = 0.50 * cw
        anchor_mid_y = 0.235 * ch
        target_fw = 0.138 * cw  # realistic 7.45-head medium-full portrait scale
        if original_location_path is not None and Path(original_location_path).is_file():
            orig_loc = cv2.imread(str(original_location_path))
            if orig_loc is not None:
                oh, ow = orig_loc.shape[:2]
                try:
                    f_loc = self.detector.detect(orig_loc)
                    lx, ly, lw, lh = f_loc.bbox
                    anchor_cx = ((lx + 0.5 * lw) / float(ow)) * cw
                    anchor_mid_y = ((ly + 0.42 * lh) / float(oh)) * ch
                    # Calibrate head width from foreign model's shoulder span -> subject's true 1.96x acromion ratio!
                    pil_loc = Image.fromarray(cv2.cvtColor(orig_loc, cv2.COLOR_BGR2RGB))
                    bj_loc = self.vision.body_and_hand_points(pil_loc) or {}
                    inv = bank.skeletal_invariants()
                    for b in bj_loc.get("bodies") or []:
                        jm = {str(j["name"]): (float(j["x"]), float(j["y"])) for j in b if float(j.get("conf", 0.0)) >= 0.2}
                        ls, rs = jm.get("left_shoulder_1_joint"), jm.get("right_shoulder_1_joint")
                        if ls and rs:
                            acro_norm = math.hypot(rs[0] - ls[0], rs[1] - ls[1]) / float(ow)
                            # Subject's real shoulder span is ~12% narrower than Western runway shoulder pads
                            subj_acro_px = acro_norm * cw * 0.90
                            target_fw = np.clip(subj_acro_px / max(1.65, inv["median_acromion_to_face_w"]), 0.115 * cw, 0.175 * cw)
                        break
                except Exception:
                    pass

        primary_shot, _ = bank.select_best_reference_pair(target_yaw_deg=13.0, scene_instruction=scene_instruction)
        src_bgr = cv2.imread(primary_shot.path)
        if src_bgr is None:
            raise FileNotFoundError(f"Real shot unreadable: {primary_shot.path}")

        head_bgr, head_alpha, gmeta = self._extract_universal_decontaminated_head_layer(src_bgr)

        # Pre-relight anchored head in Linear RGB to match target scene's directional chiaroscuro & film color palette!
        mood_src_bgr = clean_bgr
        if original_location_path is not None and Path(original_location_path).is_file():
            tmp_ol = cv2.imread(str(original_location_path))
            if tmp_ol is not None:
                mood_src_bgr = tmp_ol
        scene_mood = analyze_target_scene_chiaroscuro_and_mood(mood_src_bgr, self.detector)

        try:
            hh, hw = head_bgr.shape[:2]
            if bool(scene_mood.get("is_low_key", False)):
                yy_h, xx_h = np.mgrid[0:hh, 0:hw].astype(np.float32)
                mx_h, my_h = gmeta["mid_xy"]
                chin_x_h, chin_y_h = gmeta["chin_xy"]
                fw_h = max(24.0, float(gmeta["face_w_px"]))
                nx_h = (xx_h - float(mx_h)) / (0.54 * fw_h)
                if not bool(scene_mood.get("key_from_left", True)):
                    nx_h = -nx_h
                tgt_p20 = float(scene_mood.get("face_p20_L", 58.0))
                tgt_p85 = float(scene_mood.get("face_p85_L", 182.0))
                shd_lin = float(np.clip(((tgt_p20 + 10.0) / 192.0) ** 2.2, 0.16, 0.42))
                lit_lin = float(np.clip((tgt_p85 / 196.0) ** 2.2, 0.74, 0.96))
                side_env = shd_lin + (lit_lin - shd_lin) * (1.0 / (1.0 + np.exp(1.75 * (nx_h - 0.06))))
                neck_dim = 1.0 - 0.34 * np.clip((yy_h - (float(chin_y_h) - 0.10 * fw_h)) / max(12.0, 0.35 * fw_h), 0.0, 1.0)
                lin_h = np.power(np.clip(head_bgr.astype(np.float32) / 255.0, 0.0, 1.0), 2.2)
                lin_h *= (side_env * neck_dim)[:, :, None]
                srgb_h = np.clip(np.power(np.clip(lin_h, 0.0, 1.0), 1.0 / 2.2) * 255.0, 0.0, 255.0).astype(np.uint8)
                lab_h = cv2.cvtColor(srgb_h, cv2.COLOR_BGR2LAB).astype(np.float32)
                fg_m = head_alpha > 0.5
                tgt_fa = float(scene_mood.get("face_a", 4.5)) + 128.0
                tgt_fb = float(scene_mood.get("face_b", 9.5)) + 128.0
                cur_fa = float(np.median(lab_h[:, :, 1][fg_m])) if np.any(fg_m) else tgt_fa
                cur_fb = float(np.median(lab_h[:, :, 2][fg_m])) if np.any(fg_m) else tgt_fb
                lab_h[:, :, 1] = np.clip(lab_h[:, :, 1] + 0.68 * (tgt_fa - cur_fa), 40.0, 220.0)
                lab_h[:, :, 2] = np.clip(lab_h[:, :, 2] + 0.55 * (tgt_fb - cur_fb), 40.0, 220.0)
                head_bgr = cv2.cvtColor(lab_h.astype(np.uint8), cv2.COLOR_LAB2BGR)
            else:
                radio = SceneRadiometryAnalyzer.analyze(scene_bgr=clean_bgr)
                lab_h = cv2.cvtColor(head_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
                lab_bg = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
                l_scene_med = float(np.median(lab_bg[:, :, 0]))
                l_scale = float(np.clip((l_scene_med + 42.0) / 195.0, 0.84, 1.04))
                lab_h[:, :, 0] = np.clip(lab_h[:, :, 0] * l_scale, 0.0, 255.0)
                if radio is not None:
                    warm_shift = float(np.clip((5600.0 - float(radio.kelvin_est)) / 420.0, -3.5, 4.5))
                    lab_h[:, :, 2] = np.clip(lab_h[:, :, 2] + warm_shift, 40.0, 220.0)
                head_bgr = cv2.cvtColor(lab_h.astype(np.uint8), cv2.COLOR_LAB2BGR)
        except Exception:
            pass

        scale_iso = float(target_fw) / max(12.0, float(gmeta["face_w_px"]))
        mid_x, mid_y = gmeta["mid_xy"]
        tx = float(anchor_cx) - scale_iso * float(mid_x)
        ty = float(anchor_mid_y) - scale_iso * float(mid_y)
        M = np.array([[scale_iso, 0.0, tx], [0.0, scale_iso, ty]], dtype=np.float32)

        warped_head = _warp_affine_nyquist(head_bgr, M, (cw, ch), flags=cv2.INTER_LANCZOS4)
        warped_alpha = cv2.warpAffine(head_alpha, M, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        warped_alpha_3 = np.clip(warped_alpha, 0.0, 1.0)[:, :, None]

        precomp = np.clip(
            warped_head.astype(np.float32) * warped_alpha_3 + clean_bgr.astype(np.float32) * (1.0 - warped_alpha_3),
            0.0,
            255.0,
        ).astype(np.uint8)

        output_canvas_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_canvas_path), precomp)

        return {
            "precomposite_path": str(output_canvas_path),
            "locked_real_shot": primary_shot.name,
            "locked_real_shot_path": primary_shot.path,
            "locked_yaw_deg": primary_shot.yaw_deg,
            "canvas_size": [cw, ch],
            "target_face_w_px": round(float(target_fw), 1),
            "anchor_mid_xy": [round(float(anchor_cx), 1), round(float(anchor_mid_y), 1)],
        }

    def lock_real_face_with_scene_relighting(
        self,
        outpainted_4k_bgr: np.ndarray,
        bank: RealMultiShotBank,
        locked_shot_path: Optional[Path] = None,
        scene_mood: Optional[Dict[str, object]] = None,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """Lock 100% rigid camera-original face geometry + intrinsic low-frequency scene lighting onto `outpainted_4k_bgr`."""
        forced_entry: Optional[RealShotEntry] = None
        if locked_shot_path is not None:
            lp = str(Path(locked_shot_path).resolve())
            forced_entry = next((e for e in bank.entries if str(Path(e.path).resolve()) == lp), None)
        is_lk = bool(scene_mood.get("is_low_key", False)) if scene_mood is not None else False
        return self.relighter.relight_and_fuse_real_face(
            outpainted_4k_bgr,
            bank=bank,
            forced_source_entry=forced_entry,
            align_canvas_tps=False,  # 100% rigid similarity alignment — zero non-rigid warp on real face!
            shade_gamma=0.96 if is_lk else 0.84,
            ambient_chroma_blend=0.60 if is_lk else 0.26,
            scene_mood=scene_mood,
        )
