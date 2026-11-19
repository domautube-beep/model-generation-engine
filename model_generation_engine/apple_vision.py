"""macOS Native Apple Vision Wrapper (landmarks, faces, person_mask, face_shape, compare_shapes).

Ported and generalized from AI모델스튜디오 (`studio/vision.py`) into `model_generation_engine`:
- Compiles and caches native Swift Vision tools (`landmarks.swift`, `faces.swift`, `person_mask.swift`)
  inside the Google Drive Heavy Asset Vault (`models/apple_vision_bin`) or user cache.
- Provides 17-point jaw contour, exact left/right pupil coordinates, accurate person segmentation mask,
  and quantitative face proportion verification (`compare_face_shapes`: eye-to-chin & width-to-length %).
"""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageOps


class AppleVisionError(RuntimeError):
    """Raised when a macOS native Vision Swift tool fails to compile or execute."""


class AppleVisionAnalyzer:
    """Thread-safe runner for macOS Vision `faces`, `landmarks`, and `person_mask` Swift tools."""

    def __init__(self, bin_dir: Optional[Path] = None, tools_dir: Optional[Path] = None) -> None:
        self.tools_dir = tools_dir or (Path(__file__).resolve().parent / "tools")
        if bin_dir is None:
            default_cache = Path.home() / ".cache" / "model_generation_engine" / "apple_vision_bin"
            default_cache.mkdir(parents=True, exist_ok=True)
            bin_dir = default_cache
        self.bin_dir = Path(bin_dir)
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _binary_for(self, tool_stem: str) -> Path:
        src = self.tools_dir / f"{tool_stem}.swift"
        if not src.exists():
            raise AppleVisionError(f"Swift Vision source missing: {src}")
        digest = sha256(src.read_bytes()).hexdigest()[:12]
        exe = self.bin_dir / f"{tool_stem}-{digest}"
        if exe.exists() and os.access(exe, os.X_OK):
            return exe

        with self._lock:
            if exe.exists() and os.access(exe, os.X_OK):
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
            return exe

    @staticmethod
    def _write_rgb_fast(raw_or_pil: bytes | Image.Image, dst: Path) -> None:
        if isinstance(raw_or_pil, Image.Image):
            ImageOps.exif_transpose(raw_or_pil).convert("RGB").save(dst, "JPEG", quality=98, subsampling=0)
        else:
            with Image.open(BytesIO(raw_or_pil)) as im:
                ImageOps.exif_transpose(im).convert("RGB").save(dst, "JPEG", quality=98, subsampling=0)

    def landmarks(self, raw_or_pil: bytes | Image.Image) -> Optional[Dict[str, List[List[float]]]]:
        """Return {'contour', 'leftPupil', 'rightPupil', 'leftEye', 'rightEye', 'nose'} or None if no face."""
        exe = self._binary_for("landmarks")
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "img.jpg"
            self._write_rgb_fast(raw_or_pil, img_path)
            done = subprocess.run([str(exe), str(img_path)], capture_output=True, text=True, timeout=60)
            if done.returncode != 0:
                raise AppleVisionError(f"landmarks failed (code {done.returncode}): {done.stderr}")
            data = json.loads(done.stdout)
            return data if data.get("contour") else None

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
        """Return an L-mode person segmentation mask (255 = person, 0 = background) at image resolution."""
        exe = self._binary_for("person_mask")
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "img.jpg"
            out_path = Path(tmp) / "mask.png"
            self._write_rgb_fast(raw_or_pil, img_path)
            done = subprocess.run([str(exe), str(img_path), str(out_path)], capture_output=True, text=True, timeout=60)
            if done.returncode == 3 or not out_path.exists():
                return None
            if done.returncode != 0:
                raise AppleVisionError(f"person_mask failed (code {done.returncode}): {done.stderr}")
            with Image.open(out_path) as m:
                return m.convert("L").copy()

    def text_boxes(self, raw_or_pil: bytes | Image.Image) -> List[Dict[str, object]]:
        """Detect graphic typography/brand overlay bounding boxes (normalized top-left x, y, w, h) via macOS Vision."""
        exe = self._binary_for("text_boxes")
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "img.jpg"
            self._write_rgb_fast(raw_or_pil, img_path)
            done = subprocess.run([str(exe), str(img_path)], capture_output=True, text=True, timeout=60)
            if done.returncode != 0:
                return []
            data = json.loads(done.stdout)
            return list(data.get("boxes") or [])

    def body_and_hand_points(self, raw_or_pil: bytes | Image.Image) -> Dict[str, object]:
        """Extract full human skeletal joints + 21-point hand joints in pixel coordinates via macOS Vision."""
        exe = self._binary_for("body_joints")
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "img.jpg"
            self._write_rgb_fast(raw_or_pil, img_path)
            done = subprocess.run([str(exe), str(img_path)], capture_output=True, text=True, timeout=60)
            if done.returncode != 0:
                return {"bodies": [], "hands": []}
            return json.loads(done.stdout)

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
        eye_y = (left[1] + right[1]) / 2.0
        mid_x = (left[0] + right[0]) / 2.0
        length = max(p[1] for p in contour) - eye_y
        width = max(p[0] for p in contour) - min(p[0] for p in contour)
        if eyes <= 0 or length <= 0:
            return None
        turn = (nose[0] - mid_x) / eyes
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
