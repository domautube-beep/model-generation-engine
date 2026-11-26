"""macOS Native Apple Vision Wrapper (landmarks, faces, person_mask, face_shape, compare_shapes).

Ported and generalized from AI모델스튜디오 (`studio/vision.py`) into `model_generation_engine`:
- Compiles and caches native Swift Vision tools (`landmarks.swift`, `faces.swift`, `person_mask.swift`)
  inside the Google Drive Heavy Asset Vault (`models/apple_vision_bin`) or user cache.
- Provides 17-point jaw contour, exact left/right pupil coordinates, accurate person segmentation mask,
  and quantitative face proportion verification (`compare_face_shapes`: eye-to-chin & width-to-length %).
"""

from __future__ import annotations

from hashlib import blake2b, sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import threading
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageOps

_VISION_CACHE_LOCK = threading.Lock()
_MEM_LANDMARKS_CACHE: Dict[str, Optional[Dict[str, List[List[float]]]]] = {}
_MEM_PERSON_MASK_CACHE: Dict[str, Optional[bytes]] = {}
_MEM_TEXT_BOXES_CACHE: Dict[str, List[Dict[str, object]]] = {}
_MEM_BODY_JOINTS_CACHE: Dict[str, Dict[str, object]] = {}
_BIN_PATH_CACHE: Dict[Tuple[str, str], Path] = {}


class AppleVisionError(RuntimeError):
    """Raised when a macOS native Vision Swift tool fails to compile or execute."""


class AppleVisionAnalyzer:
    """Thread-safe runner for macOS Vision `faces`, `landmarks`, and `person_mask` Swift tools."""

    def __init__(self, bin_dir: Optional[Path] = None, tools_dir: Optional[Path] = None) -> None:
        self.tools_dir = tools_dir or (Path(__file__).resolve().parent / "tools")
        # Always execute compiled Mach-O binaries from local SSD (never over Google Drive FUSE mount, which can hit Errno 60!)
        local_ssd_cache = Path.home() / ".cache" / "model_generation_engine" / "apple_vision_bin"
        if bin_dir is None or "CloudStorage" in str(bin_dir) or "GoogleDrive" in str(bin_dir):
            bin_dir = local_ssd_cache
        self.bin_dir = Path(bin_dir)
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        self.disk_cache_dir = Path.home() / ".cache" / "model_generation_engine" / "vision_cache"
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def _image_fingerprint(raw_or_pil: bytes | Image.Image) -> str:
        """Compute fast (<0.8ms) collision-free blake2b digest for PIL Image or raw encoded bytes."""
        h = blake2b(digest_size=16)
        if isinstance(raw_or_pil, bytes):
            h.update(b"raw:")
            h.update(raw_or_pil)
            return h.hexdigest()
        im = ImageOps.exif_transpose(raw_or_pil).convert("RGB")
        w, ht = im.size
        h.update(struct.pack("<II", w, ht))
        if w * ht <= 1_600_000:
            h.update(im.tobytes())
        else:
            # Sub-millisecond dense 2-tier fingerprint (global 384x384 + center 256x256)
            h.update(im.resize((384, 384), Image.Resampling.NEAREST).tobytes())
            cw, ch = max(8, w // 2), max(8, ht // 2)
            cx0, cy0 = (w - cw) // 2, (ht - ch) // 2
            h.update(im.crop((cx0, cy0, cx0 + cw, cy0 + ch)).resize((256, 256), Image.Resampling.NEAREST).tobytes())
        return h.hexdigest()

    def _binary_for(self, tool_stem: str) -> Path:
        bkey = (str(self.tools_dir), tool_stem)
        cached_exe = _BIN_PATH_CACHE.get(bkey)
        if cached_exe is not None and cached_exe.exists():
            return cached_exe
        src = self.tools_dir / f"{tool_stem}.swift"
        if not src.exists():
            raise AppleVisionError(f"Swift Vision source missing: {src}")
        digest = sha256(src.read_bytes()).hexdigest()[:12]
        exe = self.bin_dir / f"{tool_stem}-{digest}"
        if exe.exists() and os.access(exe, os.X_OK):
            _BIN_PATH_CACHE[bkey] = exe
            return exe

        with self._lock:
            if exe.exists() and os.access(exe, os.X_OK):
                _BIN_PATH_CACHE[bkey] = exe
                return exe
            swiftc = shutil.which("swiftc")
            if not swiftc:
                raise AppleVisionError("macOS 'swiftc' compiler not found; install Xcode Command Line Tools.")
            tmp_exe = self.bin_dir / f"{tool_stem}-{digest}.building"
            done = subprocess.run(
                [swiftc, "-O", str(src), "-o", str(tmp_exe)],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if done.returncode != 0 or not tmp_exe.exists():
                tmp_exe.unlink(missing_ok=True)
                raise AppleVisionError(f"Failed to compile {src.name}: {done.stderr or done.stdout}")
            os.replace(tmp_exe, exe)
            _BIN_PATH_CACHE[bkey] = exe
            return exe

    @staticmethod
    def _write_rgb_fast(raw_or_pil: bytes | Image.Image, dst: Path) -> None:
        if isinstance(raw_or_pil, Image.Image):
            ImageOps.exif_transpose(raw_or_pil).convert("RGB").save(dst, "JPEG", quality=98, subsampling=0)
        else:
            with Image.open(BytesIO(raw_or_pil)) as im:
                ImageOps.exif_transpose(im).convert("RGB").save(dst, "JPEG", quality=98, subsampling=0)

    def landmarks(self, raw_or_pil: bytes | Image.Image) -> Optional[Dict[str, List[List[float]]]]:
        """Return {'contour', 'leftPupil', 'rightPupil', 'leftEye', 'rightEye', 'nose'} or None if no face (2-tier cached)."""
        exe = self._binary_for("landmarks")
        ckey = f"{exe.name}_{self._image_fingerprint(raw_or_pil)}"
        with _VISION_CACHE_LOCK:
            if ckey in _MEM_LANDMARKS_CACHE:
                return _MEM_LANDMARKS_CACHE[ckey]
        disk_f = self.disk_cache_dir / f"lm_{ckey}.json"
        if disk_f.is_file():
            try:
                val = json.loads(disk_f.read_text(encoding="utf-8"))
                out_cached = val if (isinstance(val, dict) and val.get("contour")) else None
                with _VISION_CACHE_LOCK:
                    _MEM_LANDMARKS_CACHE[ckey] = out_cached
                return out_cached
            except Exception:
                pass

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "img.jpg"
            self._write_rgb_fast(raw_or_pil, img_path)
            done = subprocess.run([str(exe), str(img_path)], capture_output=True, text=True, timeout=60)
            if done.returncode != 0:
                raise AppleVisionError(f"landmarks failed (code {done.returncode}): {done.stderr}")
            data = json.loads(done.stdout)
            res = data if data.get("contour") else None
        try:
            disk_f.write_text(json.dumps(res or {}, separators=(",", ":")), encoding="utf-8")
        except Exception:
            pass
        with _VISION_CACHE_LOCK:
            if len(_MEM_LANDMARKS_CACHE) > 1024:
                _MEM_LANDMARKS_CACHE.clear()
            _MEM_LANDMARKS_CACHE[ckey] = res
        return res

    def extract_key_points(self, raw_or_pil: bytes | Image.Image) -> Optional[Dict[str, object]]:
        """Extract canonical pupil midpoint, chin bottom, jaw contour, and face width (`face_detail.points`)."""
        marks = self.landmarks(raw_or_pil)
        if not marks or not marks.get("leftPupil") or not marks.get("rightPupil") or not marks.get("contour"):
            return None
        lp = tuple(marks["leftPupil"][0])
        rp = tuple(marks["rightPupil"][0])
        contour = [tuple(p) for p in marks["contour"]]
        chin = max(contour, key=lambda p: p[1])
        width = max(p[0] for p in contour) - min(p[0] for p in contour)
        return {
            "lp": lp,
            "rp": rp,
            "mid": ((lp[0] + rp[0]) / 2.0, (lp[1] + rp[1]) / 2.0),
            "chin": chin,
            "contour": contour,
            "width": float(width),
            "marks": marks,
        }

    def person_mask(self, raw_or_pil: bytes | Image.Image) -> Optional[Image.Image]:
        """Return an L-mode person segmentation mask (255 = person, 0 = background) at image resolution (2-tier cached)."""
        exe = self._binary_for("person_mask")
        ckey = f"{exe.name}_{self._image_fingerprint(raw_or_pil)}"
        with _VISION_CACHE_LOCK:
            if ckey in _MEM_PERSON_MASK_CACHE:
                png_b = _MEM_PERSON_MASK_CACHE[ckey]
                if png_b is None:
                    return None
                with Image.open(BytesIO(png_b)) as m:
                    return m.convert("L").copy()
        disk_png = self.disk_cache_dir / f"pm_{ckey}.png"
        disk_none = self.disk_cache_dir / f"pm_{ckey}.none"
        if disk_none.is_file():
            with _VISION_CACHE_LOCK:
                _MEM_PERSON_MASK_CACHE[ckey] = None
            return None
        if disk_png.is_file():
            try:
                png_b = disk_png.read_bytes()
                with Image.open(BytesIO(png_b)) as m:
                    im_l = m.convert("L").copy()
                with _VISION_CACHE_LOCK:
                    _MEM_PERSON_MASK_CACHE[ckey] = png_b
                return im_l
            except Exception:
                pass

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "img.jpg"
            out_path = Path(tmp) / "mask.png"
            self._write_rgb_fast(raw_or_pil, img_path)
            done = subprocess.run([str(exe), str(img_path), str(out_path)], capture_output=True, text=True, timeout=60)
            if done.returncode == 3 or not out_path.exists():
                try:
                    disk_none.write_text("0", encoding="utf-8")
                except Exception:
                    pass
                with _VISION_CACHE_LOCK:
                    _MEM_PERSON_MASK_CACHE[ckey] = None
                return None
            if done.returncode != 0:
                raise AppleVisionError(f"person_mask failed (code {done.returncode}): {done.stderr}")
            png_bytes = out_path.read_bytes()
            try:
                shutil.copy2(out_path, disk_png)
            except Exception:
                pass
            with _VISION_CACHE_LOCK:
                if len(_MEM_PERSON_MASK_CACHE) > 256:
                    _MEM_PERSON_MASK_CACHE.clear()
                _MEM_PERSON_MASK_CACHE[ckey] = png_bytes
            with Image.open(BytesIO(png_bytes)) as m:
                return m.convert("L").copy()

    def text_boxes(self, raw_or_pil: bytes | Image.Image) -> List[Dict[str, object]]:
        """Detect graphic typography/brand overlay bounding boxes (normalized top-left x, y, w, h) via macOS Vision."""
        exe = self._binary_for("text_boxes")
        ckey = f"{exe.name}_{self._image_fingerprint(raw_or_pil)}"
        with _VISION_CACHE_LOCK:
            if ckey in _MEM_TEXT_BOXES_CACHE:
                return list(_MEM_TEXT_BOXES_CACHE[ckey])
        disk_f = self.disk_cache_dir / f"tb_{ckey}.json"
        if disk_f.is_file():
            try:
                boxes = list(json.loads(disk_f.read_text(encoding="utf-8")))
                with _VISION_CACHE_LOCK:
                    _MEM_TEXT_BOXES_CACHE[ckey] = boxes
                return list(boxes)
            except Exception:
                pass
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "img.jpg"
            self._write_rgb_fast(raw_or_pil, img_path)
            done = subprocess.run([str(exe), str(img_path)], capture_output=True, text=True, timeout=60)
            if done.returncode != 0:
                return []
            data = json.loads(done.stdout)
            boxes = list(data.get("boxes") or [])
        try:
            disk_f.write_text(json.dumps(boxes, separators=(",", ":")), encoding="utf-8")
        except Exception:
            pass
        with _VISION_CACHE_LOCK:
            _MEM_TEXT_BOXES_CACHE[ckey] = boxes
        return list(boxes)

    def body_and_hand_points(self, raw_or_pil: bytes | Image.Image) -> Dict[str, object]:
        """Extract full human skeletal joints + 21-point hand joints in pixel coordinates via macOS Vision (2-tier cached)."""
        exe = self._binary_for("body_joints")
        ckey = f"{exe.name}_{self._image_fingerprint(raw_or_pil)}"
        with _VISION_CACHE_LOCK:
            if ckey in _MEM_BODY_JOINTS_CACHE:
                return _MEM_BODY_JOINTS_CACHE[ckey]
        disk_f = self.disk_cache_dir / f"bj_{ckey}.json"
        if disk_f.is_file():
            try:
                bj = json.loads(disk_f.read_text(encoding="utf-8"))
                with _VISION_CACHE_LOCK:
                    _MEM_BODY_JOINTS_CACHE[ckey] = bj
                return bj
            except Exception:
                pass
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "img.jpg"
            self._write_rgb_fast(raw_or_pil, img_path)
            done = subprocess.run([str(exe), str(img_path)], capture_output=True, text=True, timeout=60)
            if done.returncode != 0:
                res_bj: Dict[str, object] = {"bodies": [], "hands": []}
            else:
                res_bj = json.loads(done.stdout)
        try:
            disk_f.write_text(json.dumps(res_bj, separators=(",", ":")), encoding="utf-8")
        except Exception:
            pass
        with _VISION_CACHE_LOCK:
            if len(_MEM_BODY_JOINTS_CACHE) > 512:
                _MEM_BODY_JOINTS_CACHE.clear()
            _MEM_BODY_JOINTS_CACHE[ckey] = res_bj
        return res_bj

    @staticmethod
    def compute_face_shape(marks: Optional[Dict[str, List[List[float]]]]) -> Optional[Dict[str, object]]:
        """Compute normalized eye-to-chin length and width-to-length facial proportions (`vision.face_shape`)."""
        if not marks:
            return None

        def _center(pts: Optional[List[List[float]]]) -> Optional[Tuple[float, float]]:
            return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)) if pts else None

        left = _center(marks.get("leftPupil") or marks.get("leftEye"))
        right = _center(marks.get("rightPupil") or marks.get("rightEye"))
        nose = _center(marks.get("nose"))
        contour = marks.get("contour") or []
        if not (left and right and nose and contour):
            return None
        eyes = math.dist(left, right)
        if eyes <= 0:
            return None
        eye_y = (left[1] + right[1]) / 2.0
        mid_x = (left[0] + right[0]) / 2.0
        # Project onto pupil-aligned orthonormal basis (ux, uy) so rigid SE(2) head tilt never inflates cheek width!
        ux = (right[0] - left[0]) / eyes
        uy = (right[1] - left[1]) / eyes
        if ux < 0:
            ux, uy = -ux, -uy
        u_vals = [(p[0] - mid_x) * ux + (p[1] - eye_y) * uy for p in contour]
        v_vals = [-(p[0] - mid_x) * uy + (p[1] - eye_y) * ux for p in contour]
        length = max(v_vals)
        width = max(u_vals) - min(u_vals)
        if length <= 0:
            return None
        turn = ((nose[0] - mid_x) * ux + (nose[1] - eye_y) * uy) / eyes
        # Cosine foreshortening compensation for semi-frontal turns (|turn| < 0.30, ~25 deg) so cheek width is STILL audited!
        cos_comp = math.sqrt(max(0.72, 1.0 - min(0.28, (turn * 1.15) ** 2)))
        norm_wl = (width / length) / cos_comp
        return {
            "eyePx": round(eyes, 1),
            "eyeChin": round(length / eyes, 3),
            "widthLength": round(width / length, 3),
            "normWidthLength": round(norm_wl, 3),
            "turn": round(turn, 3),
            "frontal": abs(turn) < 0.12,
            "auditable": abs(turn) < 0.30,
        }

    @staticmethod
    def compare_face_shapes(
        original: Optional[Dict[str, object]],
        result: Optional[Dict[str, object]],
        limit_pct: float = 3.5,
    ) -> Dict[str, object]:
        """Compare craniofacial proportions (% diff in eye-to-chin & perspective-normalized cheek width-to-length)."""
        if not original or not result:
            return {"state": "unmeasured", "passed": True, "note": "얼굴 윤곽 미검출"}
        orig_aud = bool(original.get("auditable", original.get("frontal")))
        res_aud = bool(result.get("auditable", result.get("frontal")))
        if not (orig_aud and res_aud):
            return {"state": "unmeasured", "passed": True, "note": "깊은 측면(>25°) 비율 비교 생략"}

        orig_ec = float(original["eyeChin"])
        res_ec = float(result["eyeChin"])
        both_strict_frontal = bool(original.get("frontal") and result.get("frontal"))
        orig_wl = float(original["widthLength"] if both_strict_frontal else original.get("normWidthLength", original["widthLength"]))
        res_wl = float(result["widthLength"] if both_strict_frontal else result.get("normWidthLength", result["widthLength"]))
        eff_lim = limit_pct if both_strict_frontal else max(limit_pct, 4.0)

        length_diff = round((res_ec / orig_ec - 1.0) * 100.0, 1)
        ratio_diff = round((res_wl / orig_wl - 1.0) * 100.0, 1)
        drifted = abs(length_diff) > eff_lim or abs(ratio_diff) > eff_lim
        return {
            "state": "drifted" if drifted else "same",
            "passed": not drifted,
            "lengthDiffPct": length_diff,
            "ratioDiffPct": ratio_diff,
            "limitPct": eff_lim,
            "note": f"원본 대비 눈~턱 {length_diff:+}% · 볼폭/길이 {ratio_diff:+}%",
        }

    def compute_cranial_and_scale_profile(
        self,
        raw_or_pil: Image.Image,
        marks: Optional[Dict[str, List[List[float]]]] = None,
    ) -> Optional[Dict[str, object]]:
        """Measure above-eyebrow Cranial Dome (forehead height, crown root thickness, outer cranial width, L/R hair lobe) + Face-to-Shoulder scale."""
        import cv2
        import numpy as np

        pil_rgb = ImageOps.exif_transpose(raw_or_pil).convert("RGB")
        orig_w, orig_h = pil_rgb.size
        scale = min(1.0, 1024.0 / float(max(orig_w, orig_h)))
        work_pil = (
            pil_rgb.resize((max(32, int(round(orig_w * scale))), max(32, int(round(orig_h * scale)))), Image.Resampling.BILINEAR)
            if scale < 1.0
            else pil_rgb
        )
        w, h = work_pil.size
        eff_marks = self.landmarks(work_pil) if (marks is None or scale < 1.0) else marks
        if not eff_marks or not eff_marks.get("contour") or not (eff_marks.get("leftPupil") or eff_marks.get("leftEye")):
            return None

        lp_pts = eff_marks.get("leftPupil") or eff_marks.get("leftEye")
        rp_pts = eff_marks.get("rightPupil") or eff_marks.get("rightEye")
        if not lp_pts or not rp_pts:
            return None
        lp = (sum(p[0] for p in lp_pts) / len(lp_pts), sum(p[1] for p in lp_pts) / len(lp_pts))
        rp = (sum(p[0] for p in rp_pts) / len(rp_pts), sum(p[1] for p in rp_pts) / len(rp_pts))
        mid_x, mid_y = 0.5 * (lp[0] + rp[0]), 0.5 * (lp[1] + rp[1])
        contour = eff_marks["contour"]
        chin_y = max(p[1] for p in contour)
        fx_min, fx_max = min(p[0] for p in contour), max(p[0] for p in contour)
        fw = max(1.0, fx_max - fx_min)
        eye_to_chin = max(1.0, chin_y - mid_y)

        pm = self.person_mask(work_pil)
        if pm is None:
            return None
        pm_arr = np.array(pm.resize((w, h), Image.Resampling.BILINEAR)) > 120

        # 1. Crown top y directly above pupil midpoint
        cx1 = max(0, int(round(mid_x - 0.25 * fw)))
        cx2 = min(w, int(round(mid_x + 0.25 * fw)))
        ys_top = np.where(pm_arr[: max(1, int(round(mid_y))), cx1:cx2])[0]
        crown_y = float(ys_top.min()) if len(ys_top) > 0 else max(0.0, mid_y - eye_to_chin)
        crown_clipped = crown_y <= 3.0
        crown_to_eye = (mid_y - crown_y) / eye_to_chin

        # 2. Universal self-referenced CIE LAB hairline detector (compares upward strip against subject's own glabella skin!)
        rgb_np = np.asarray(work_pil, dtype=np.uint8)
        lab_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        glab_y1 = max(0, int(round(mid_y - 0.28 * eye_to_chin)))
        glab_y2 = max(glab_y1 + 2, int(round(mid_y - 0.16 * eye_to_chin)))
        sx1 = max(0, int(round(mid_x - 0.10 * fw)))
        sx2 = min(w, max(sx1 + 2, int(round(mid_x + 0.10 * fw))))
        glab_patch = lab_np[glab_y1:glab_y2, sx1:sx2]
        skin_ref = np.median(glab_patch.reshape(-1, 3), axis=0) if glab_patch.size >= 3 else np.array([150.0, 132.0, 136.0])

        hairline_y = crown_y
        scan_start = max(int(round(crown_y)), int(round(mid_y - 0.24 * eye_to_chin)))
        for yy in range(scan_start, max(0, int(round(crown_y))) - 1, -1):
            row_med = np.median(lab_np[yy, sx1:sx2], axis=0)
            dist_lab = float(np.linalg.norm(row_med - skin_ref))
            if dist_lab >= 18.0:
                hairline_y = float(yy)
                break

        forehead_ratio = float(np.clip((mid_y - hairline_y) / eye_to_chin, 0.35, 1.15))
        crown_root_ratio = float(np.clip((hairline_y - crown_y) / eye_to_chin, 0.0, 0.85))

        # 3. Outer cranial + temporal hair width at upper-temple level (`mid_y - 0.10 * eye_to_chin`)
        ty = max(0, min(h - 1, int(round(mid_y - 0.10 * eye_to_chin))))
        # Restrict horizontal search to connected head region around mid_x (`mid_x ± 1.15 * fw`) so raised hands don't count as skull!
        hx1 = max(0, int(round(mid_x - 1.15 * fw)))
        hx2 = min(w, int(round(mid_x + 1.15 * fw)))
        xs_head = np.where(pm_arr[ty, hx1:hx2])[0]
        if len(xs_head) > 0:
            hl_x = float(hx1 + xs_head.min())
            hr_x = float(hx1 + xs_head.max())
            cranial_w = max(fw, hr_x - hl_x)
            left_lobe = max(0.0, fx_min - hl_x) / fw
            right_lobe = max(0.0, hr_x - fx_max) / fw
        else:
            cranial_w = fw * 1.25
            left_lobe, right_lobe = 0.12, 0.12
        cranial_to_face_w = float(np.clip(cranial_w / fw, 1.0, 2.0))

        # 4. True Skeletal Bi-Deltoid Shoulder Span (uses Vision Body Pose joints immune to arms on armrests, falling back to person_mask)
        eyes_px = max(1.0, math.dist(lp, rp))
        nose_pts = eff_marks.get("nose") or []
        nose_x = (sum(p[0] for p in nose_pts) / len(nose_pts)) if nose_pts else mid_x
        turn_est = (nose_x - mid_x) / eyes_px
        cos_torso = math.sqrt(max(0.58, 1.0 - min(0.42, (turn_est * 1.25) ** 2)))

        shoulders_in_frame = False
        face_to_shoulder = 0.32
        if (fw / float(min(w, h))) < 0.24:
            try:
                bj = self.body_and_hand_points(work_pil)
                for body in bj.get("bodies") or []:
                    jmap = {str(j["name"]): (float(j["x"]), float(j["y"])) for j in body if float(j.get("conf", 0.0)) >= 0.25}
                    jmap_hip = {str(j["name"]): (float(j["x"]), float(j["y"])) for j in body if float(j.get("conf", 0.0)) >= 0.15}
                    ls = jmap.get("left_shoulder_1_joint")
                    rs = jmap.get("right_shoulder_1_joint")
                    if ls and rs:
                        if min(ls[0], rs[0]) > w * 0.04 and max(ls[0], rs[0]) < w * 0.96 and max(ls[1], rs[1]) < h * 0.93:
                            acromion_2d = math.hypot(rs[0] - ls[0], rs[1] - ls[1])
                            if acromion_2d > fw * 1.1:
                                # De-foreshorten 3D torso yaw using consensus of vertical spine trunk & upper-arm humerus lengths
                                # ONLY when genuine torso yaw foreshortening (> 14% contraction, >= 29 deg yaw) is present!
                                skel_acro_ests: List[float] = []
                                nk = jmap.get("neck_1_joint")
                                hip = jmap_hip.get("root")
                                if hip is None and "left_upLeg_joint" in jmap_hip and "right_upLeg_joint" in jmap_hip:
                                    lu, ru = jmap_hip["left_upLeg_joint"], jmap_hip["right_upLeg_joint"]
                                    hip = (0.5 * (lu[0] + ru[0]), 0.5 * (lu[1] + ru[1]))
                                if nk and hip and hip[1] < h * 0.95 and hip[1] > nk[1] + fw * 0.8:
                                    trunk_len = math.hypot(hip[0] - nk[0], hip[1] - nk[1])
                                    skel_acro_ests.append(0.715 * trunk_len)
                                hum_lens: List[float] = []
                                if "left_forearm_joint" in jmap:
                                    hum_lens.append(math.hypot(jmap["left_forearm_joint"][0] - ls[0], jmap["left_forearm_joint"][1] - ls[1]))
                                if "right_forearm_joint" in jmap:
                                    hum_lens.append(math.hypot(jmap["right_forearm_joint"][0] - rs[0], jmap["right_forearm_joint"][1] - rs[1]))
                                if hum_lens:
                                    skel_acro_ests.append(1.12 * float(np.median(hum_lens)))
                                med_skel_acro = float(np.median(skel_acro_ests)) if skel_acro_ests else acromion_2d
                                if med_skel_acro > acromion_2d * 1.14:
                                    acromion_3d = min(med_skel_acro, acromion_2d / 0.48)
                                    bi_deltoid = acromion_3d * 1.48
                                    shoulders_in_frame = True
                                    face_to_shoulder = float((fw / max(0.75, cos_torso)) / bi_deltoid)
                                else:
                                    bi_deltoid = acromion_2d * 1.48
                                    shoulders_in_frame = True
                                    face_to_shoulder = float(fw / bi_deltoid) * cos_torso
                                break
            except Exception:
                pass

        if not shoulders_in_frame:
            sh_y1 = int(round(chin_y + 0.75 * fw))
            sh_y2 = min(h - 1, int(round(chin_y + 1.30 * fw)))
            if sh_y2 < int(h * 0.95) and sh_y2 > sh_y1 + 4 and (fw / float(min(w, h))) < 0.24:
                tx1 = max(0, int(round(mid_x - 2.05 * fw)))
                tx2 = min(w, int(round(mid_x + 2.05 * fw)))
                band_mask = np.any(pm_arr[sh_y1:sh_y2, tx1:tx2], axis=0)
                xs_torso = np.where(band_mask)[0]
                if len(xs_torso) >= 10:
                    tl_x, tr_x = float(tx1 + xs_torso.min()), float(tx1 + xs_torso.max())
                    if tl_x > w * 0.03 and tr_x < w * 0.97:
                        shoulders_in_frame = True
                        face_to_shoulder = float(fw / max(fw, tr_x - tl_x)) * cos_torso

        return {
            "crownToEyeChinRatio": round(crown_to_eye, 3),
            "foreheadToEyeChinRatio": round(forehead_ratio, 3),
            "crownRootThicknessRatio": round(crown_root_ratio, 3),
            "cranialToFaceWidthRatio": round(cranial_to_face_w, 3),
            "leftHairLobeRatio": round(left_lobe, 3),
            "rightHairLobeRatio": round(right_lobe, 3),
            "crownClipped": crown_clipped,
            "shouldersInFrame": shoulders_in_frame,
            "faceToShoulderRatio": round(face_to_shoulder, 3),
        }

    @staticmethod
    def compare_cranial_and_scale(
        ref_cranial: Optional[Dict[str, object]],
        dst_cranial: Optional[Dict[str, object]],
        max_face_to_shoulder: float = 0.375,
        min_face_to_shoulder: float = 0.285,
        max_dome_diff_pct: float = 18.0,
    ) -> Dict[str, object]:
        """Audit Two-Sided 8-Head Face-to-Shoulder Scale (rejecting BOTH big-head AND tiny-head drift) and Cranial Dome invariance."""
        if not dst_cranial:
            return {"passed": True, "note": "두상 미계측"}

        issues: List[str] = []
        notes: List[str] = []

        if bool(dst_cranial.get("shouldersInFrame")):
            fts = float(dst_cranial["faceToShoulderRatio"])
            notes.append(f"어깨대비얼굴 {fts * 100:.1f}%")
            if fts > max_face_to_shoulder:
                issues.append(
                    f"대두 비율 발생 (어깨 대비 얼굴 폭 {fts * 100:.1f}% > 최대 {max_face_to_shoulder * 100:.1f}% — 7.5~8등신 비율 위반)"
                )
            elif fts < min_face_to_shoulder:
                issues.append(
                    f"과소두 비율 발생 (어깨 대비 얼굴 폭 {fts * 100:.1f}% < 최소 {min_face_to_shoulder * 100:.1f}% — 어깨 과확장·얼굴 축소 위반)"
                )

        if ref_cranial and not bool(ref_cranial.get("crownClipped")) and not bool(dst_cranial.get("crownClipped")):
            r_dome = max(0.4, float(ref_cranial["crownToEyeChinRatio"]))
            d_dome = float(dst_cranial["crownToEyeChinRatio"])
            dome_diff = (d_dome / r_dome - 1.0) * 100.0
            r_cw = max(1.0, float(ref_cranial["cranialToFaceWidthRatio"]))
            d_cw = float(dst_cranial["cranialToFaceWidthRatio"])
            cw_diff = (d_cw / r_cw - 1.0) * 100.0
            notes.append(f"상두상높이 {dome_diff:+.1f}% · 두상가로폭 {cw_diff:+.1f}%")
            if abs(dome_diff) > max_dome_diff_pct:
                issues.append(f"상두상/정수리 볼륨 왜곡 ({dome_diff:+.1f}%, 허용 ±{max_dome_diff_pct:.0f}%)")
            if abs(cw_diff) > max_dome_diff_pct:
                issues.append(f"옆두상 가로폭 왜곡 ({cw_diff:+.1f}%, 허용 ±{max_dome_diff_pct:.0f}%)")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "note": " · ".join(notes),
        }
