"""Universal 3D Articulated Previz Pose Rig & Structural Guide Renderer (`previz_pose.py`).

Allows users and automated pipelines to define any 3D human posture, head/torso azimuth (`0°..360°`),
camera elevation, focal length, and frame placement—rendering a clean 3D articulated mannequin guide
and automatically mapping head orientation to the closest of the 8 Canonical Turnaround Views.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .identity_turnaround import resolve_8angle_view_from_azimuth

POSE_PRESETS: Dict[str, Dict[str, object]] = {
    "standing_editorial": {
        "label_ko": "스탠딩 에디토리얼 (정면·여유)",
        "shot_scale": "three-quarter portrait shot",
        "pose_text": "standing naturally with relaxed balanced shoulders and calm composed posture",
        "torso_azimuth_deg": 0.0,
        "head_azimuth_deg": 0.0,
        "joints": {
            "pelvis": (0.0, 0.0, 0.0),
            "neck": (0.0, 0.56, 0.0),
            "head": (0.0, 0.78, 0.0),
            "l_shoulder": (-0.22, 0.52, 0.0),
            "r_shoulder": (0.22, 0.52, 0.0),
            "l_elbow": (-0.26, 0.24, 0.03),
            "r_elbow": (0.26, 0.24, 0.03),
            "l_wrist": (-0.24, -0.04, 0.06),
            "r_wrist": (0.24, -0.04, 0.06),
            "l_hip": (-0.12, -0.02, 0.0),
            "r_hip": (0.12, -0.02, 0.0),
            "l_knee": (-0.13, -0.44, 0.02),
            "r_knee": (0.13, -0.44, -0.02),
            "l_ankle": (-0.14, -0.86, 0.0),
            "r_ankle": (0.14, -0.86, 0.0),
        },
    },
    "seated_armchair": {
        "label_ko": "라운지 체어 착석 (우측 여백·지적 무드)",
        "shot_scale": "three-quarter seated portrait shot",
        "pose_text": "seated comfortably in an armchair with torso gently angled, hands resting calmly on lap/knee",
        "torso_azimuth_deg": 20.0,
        "head_azimuth_deg": 0.0,
        "joints": {
            "pelvis": (0.0, -0.10, 0.0),
            "neck": (0.0, 0.44, -0.03),
            "head": (0.0, 0.66, -0.02),
            "l_shoulder": (-0.22, 0.40, -0.02),
            "r_shoulder": (0.22, 0.40, -0.02),
            "l_elbow": (-0.25, 0.14, 0.08),
            "r_elbow": (0.25, 0.14, 0.08),
            "l_wrist": (-0.06, 0.02, 0.24),
            "r_wrist": (0.06, 0.02, 0.24),
            "l_hip": (-0.12, -0.12, 0.0),
            "r_hip": (0.12, -0.12, 0.0),
            "l_knee": (-0.10, -0.14, 0.38),
            "r_knee": (0.12, -0.18, 0.36),
            "l_ankle": (-0.08, -0.62, 0.36),
            "r_ankle": (0.12, -0.62, 0.34),
        },
    },
    "standing_left_45": {
        "label_ko": "좌 45° 회전 스탠딩 화보",
        "shot_scale": "medium three-quarter standing portrait shot",
        "pose_text": "standing in a composed 45-degree left-facing three-quarter turn with calm forward-left gaze",
        "torso_azimuth_deg": 45.0,
        "head_azimuth_deg": 45.0,
        "joints": {
            "pelvis": (0.0, 0.0, 0.0),
            "neck": (0.0, 0.56, 0.0),
            "head": (0.0, 0.78, 0.0),
            "l_shoulder": (-0.22, 0.52, 0.0),
            "r_shoulder": (0.22, 0.52, 0.0),
            "l_elbow": (-0.25, 0.24, 0.04),
            "r_elbow": (0.24, 0.24, 0.08),
            "l_wrist": (-0.22, -0.02, 0.08),
            "r_wrist": (0.16, 0.02, 0.16),
            "l_hip": (-0.12, -0.02, 0.0),
            "r_hip": (0.12, -0.02, 0.0),
            "l_knee": (-0.13, -0.44, 0.04),
            "r_knee": (0.13, -0.44, -0.02),
            "l_ankle": (-0.14, -0.86, 0.04),
            "r_ankle": (0.14, -0.86, -0.02),
        },
    },
    "standing_right_45": {
        "label_ko": "우 45° 회전 스탠딩 화보",
        "shot_scale": "medium three-quarter standing portrait shot",
        "pose_text": "standing in a composed 45-degree right-facing three-quarter turn with calm forward-right gaze",
        "torso_azimuth_deg": 315.0,
        "head_azimuth_deg": 315.0,
        "joints": {
            "pelvis": (0.0, 0.0, 0.0),
            "neck": (0.0, 0.56, 0.0),
            "head": (0.0, 0.78, 0.0),
            "l_shoulder": (-0.22, 0.52, 0.0),
            "r_shoulder": (0.22, 0.52, 0.0),
            "l_elbow": (-0.24, 0.24, 0.08),
            "r_elbow": (0.25, 0.24, 0.04),
            "l_wrist": (-0.16, 0.02, 0.16),
            "r_wrist": (0.22, -0.02, 0.08),
            "l_hip": (-0.12, -0.02, 0.0),
            "r_hip": (0.12, -0.02, 0.0),
            "l_knee": (-0.13, -0.44, -0.02),
            "r_knee": (0.13, -0.44, 0.04),
            "l_ankle": (-0.14, -0.86, -0.02),
            "r_ankle": (0.14, -0.86, 0.04),
        },
    },
    "walking_stride": {
        "label_ko": "풀바디 워킹 스트라이드 (전신 도보)",
        "shot_scale": "full-body editorial walking shot",
        "pose_text": "walking forward in a confident mid-stride with natural arm swing and balanced posture",
        "torso_azimuth_deg": 12.0,
        "head_azimuth_deg": 0.0,
        "joints": {
            "pelvis": (0.0, 0.0, 0.0),
            "neck": (0.0, 0.56, 0.02),
            "head": (0.0, 0.78, 0.03),
            "l_shoulder": (-0.22, 0.52, 0.0),
            "r_shoulder": (0.22, 0.52, 0.0),
            "l_elbow": (-0.25, 0.25, 0.12),
            "r_elbow": (0.25, 0.25, -0.10),
            "l_wrist": (-0.20, 0.02, 0.24),
            "r_wrist": (0.22, -0.02, -0.18),
            "l_hip": (-0.12, -0.02, 0.02),
            "r_hip": (0.12, -0.02, -0.02),
            "l_knee": (-0.12, -0.42, 0.20),
            "r_knee": (0.12, -0.44, -0.14),
            "l_ankle": (-0.12, -0.84, 0.26),
            "r_ankle": (0.12, -0.86, -0.22),
        },
    },
    "arms_crossed": {
        "label_ko": "팔짱 프로필 (전문가·권위 무드)",
        "shot_scale": "medium three-quarter portrait shot",
        "pose_text": "standing with arms neatly folded across chest, relaxed confident shoulders",
        "torso_azimuth_deg": 20.0,
        "head_azimuth_deg": 0.0,
        "joints": {
            "pelvis": (0.0, 0.0, 0.0),
            "neck": (0.0, 0.56, 0.0),
            "head": (0.0, 0.78, 0.0),
            "l_shoulder": (-0.22, 0.52, 0.0),
            "r_shoulder": (0.22, 0.52, 0.0),
            "l_elbow": (-0.24, 0.24, 0.14),
            "r_elbow": (0.24, 0.24, 0.14),
            "l_wrist": (0.10, 0.26, 0.20),
            "r_wrist": (-0.10, 0.23, 0.22),
            "l_hip": (-0.12, -0.02, 0.0),
            "r_hip": (0.12, -0.02, 0.0),
            "l_knee": (-0.13, -0.44, 0.0),
            "r_knee": (0.13, -0.44, 0.0),
            "l_ankle": (-0.14, -0.86, 0.0),
            "r_ankle": (0.14, -0.86, 0.0),
        },
    },
    "hand_in_pocket": {
        "label_ko": "한손 포켓 캐주얼 테일러드",
        "shot_scale": "three-quarter standing portrait shot",
        "pose_text": "standing with one hand tucked casually into trouser pocket and the other arm relaxed",
        "torso_azimuth_deg": 335.0,
        "head_azimuth_deg": 345.0,
        "joints": {
            "pelvis": (0.0, 0.0, 0.0),
            "neck": (0.0, 0.56, 0.0),
            "head": (0.0, 0.78, 0.0),
            "l_shoulder": (-0.22, 0.52, 0.0),
            "r_shoulder": (0.22, 0.52, 0.0),
            "l_elbow": (-0.28, 0.28, -0.04),
            "r_elbow": (0.25, 0.24, 0.04),
            "l_wrist": (-0.15, 0.02, 0.08),
            "r_wrist": (0.23, -0.04, 0.06),
            "l_hip": (-0.12, -0.02, 0.0),
            "r_hip": (0.12, -0.02, 0.0),
            "l_knee": (-0.13, -0.44, 0.02),
            "r_knee": (0.14, -0.44, -0.02),
            "l_ankle": (-0.13, -0.86, 0.02),
            "r_ankle": (0.15, -0.86, -0.02),
        },
    },
    "over_shoulder_back": {
        "label_ko": "오버숄더 백뷰 (135° 후방 시선)",
        "shot_scale": "medium three-quarter rear-angle editorial shot",
        "pose_text": "turned 135 degrees away showing back-three-quarter silhouette with subtle over-the-shoulder head angle",
        "torso_azimuth_deg": 135.0,
        "head_azimuth_deg": 115.0,
        "joints": {
            "pelvis": (0.0, 0.0, 0.0),
            "neck": (0.0, 0.56, 0.0),
            "head": (0.0, 0.78, 0.0),
            "l_shoulder": (-0.22, 0.52, 0.0),
            "r_shoulder": (0.22, 0.52, 0.0),
            "l_elbow": (-0.25, 0.24, 0.02),
            "r_elbow": (0.25, 0.24, 0.02),
            "l_wrist": (-0.22, -0.04, 0.04),
            "r_wrist": (0.22, -0.04, 0.04),
            "l_hip": (-0.12, -0.02, 0.0),
            "r_hip": (0.12, -0.02, 0.0),
            "l_knee": (-0.13, -0.44, 0.0),
            "r_knee": (0.13, -0.44, 0.0),
            "l_ankle": (-0.14, -0.86, 0.0),
            "r_ankle": (0.14, -0.86, 0.0),
        },
    },
}

BONE_EDGES: Tuple[Tuple[str, str, Tuple[int, int, int]], ...] = (
    ("pelvis", "neck", (215, 215, 220)),
    ("neck", "head", (240, 240, 245)),
    ("neck", "l_shoulder", (90, 180, 255)),
    ("l_shoulder", "l_elbow", (70, 150, 245)),
    ("l_elbow", "l_wrist", (50, 125, 235)),
    ("neck", "r_shoulder", (255, 165, 80)),
    ("r_shoulder", "r_elbow", (245, 135, 65)),
    ("r_elbow", "r_wrist", (235, 110, 50)),
    ("pelvis", "l_hip", (110, 220, 150)),
    ("l_hip", "l_knee", (85, 195, 130)),
    ("l_knee", "l_ankle", (65, 170, 110)),
    ("pelvis", "r_hip", (215, 120, 240)),
    ("r_hip", "r_knee", (190, 95, 220)),
    ("r_knee", "r_ankle", (165, 75, 195)),
)


@dataclass
class PosePrevizSpec:
    """Universal 3D Previz Pose & Camera Rig Specification."""

    preset_id: str = "standing_editorial"
    head_azimuth_deg: float = 0.0
    head_pitch_deg: float = 0.0
    torso_azimuth_deg: float = 0.0
    camera_elevation_deg: float = 0.0
    focal_length_mm: int = 85
    subject_x_norm: float = 0.58
    shot_scale: str = "three-quarter portrait shot"
    custom_pose_note: str = ""
    joints_override: Optional[Dict[str, Tuple[float, float, float]]] = None

    @property
    def matched_8angle_view(self) -> str:
        return resolve_8angle_view_from_azimuth(self.head_azimuth_deg)

    def camera_angle_clause(self) -> str:
        if self.camera_elevation_deg <= -10.0:
            return f"subtle low-angle perspective ({abs(int(round(self.camera_elevation_deg)))} deg looking slightly upward)"
        if self.camera_elevation_deg >= 10.0:
            return f"elevated high-angle perspective ({int(round(self.camera_elevation_deg))} deg looking gently downward)"
        return "eye-level camera angle"

    def placement_clause(self) -> str:
        if self.subject_x_norm >= 0.56:
            return "subject positioned right-of-center leaving clean negative space on the left side of the frame"
        if self.subject_x_norm <= 0.44:
            return "subject positioned left-of-center leaving clean negative space on the right side of the frame"
        return "balanced centered editorial framing"

    def to_prompt_clause(self) -> str:
        preset = POSE_PRESETS.get(self.preset_id, POSE_PRESETS["standing_editorial"])
        base_pose = self.custom_pose_note.strip() or str(preset["pose_text"])
        scale_txt = self.shot_scale or str(preset["shot_scale"])
        return (
            f"{scale_txt} captured from an {self.camera_angle_clause()} with an {self.focal_length_mm}mm prime lens; "
            f"pose: {base_pose} (torso azimuth {self.torso_azimuth_deg:.0f} deg, head azimuth {self.head_azimuth_deg:.0f} deg, "
            f"head pitch {self.head_pitch_deg:+.0f} deg); composition: {self.placement_clause()}"
        )


class Previz3DRenderer:
    """Renders a clean 3D articulated mannequin guide image for visual previz and pose-conditioned generation."""

    @staticmethod
    def _rotate_y(pt: np.ndarray, deg: float) -> np.ndarray:
        rad = math.radians(deg)
        c, s = math.cos(rad), math.sin(rad)
        # Positive azimuth = turning toward image-left (-X)
        return np.array([-s * pt[2] + c * pt[0], pt[1], c * pt[2] + s * pt[0]], dtype=np.float32)

    @staticmethod
    def _rotate_x(pt: np.ndarray, deg: float) -> np.ndarray:
        rad = math.radians(deg)
        c, s = math.cos(rad), math.sin(rad)
        return np.array([pt[0], c * pt[1] - s * pt[2], s * pt[1] + c * pt[2]], dtype=np.float32)

    @classmethod
    def render_pose_guide(
        cls,
        spec: PosePrevizSpec,
        canvas_size: Tuple[int, int] = (1280, 720),
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """Render the 3D Previz mannequin skeleton + head orientation vector onto a dark neutral studio grid."""
        w, h = canvas_size
        img = np.full((h, w, 3), (28, 31, 36), dtype=np.uint8)

        preset = POSE_PRESETS.get(spec.preset_id, POSE_PRESETS["standing_editorial"])
        raw_joints: Dict[str, Tuple[float, float, float]] = dict(preset["joints"])  # type: ignore[arg-type]
        if spec.joints_override:
            raw_joints.update(spec.joints_override)

        world_joints: Dict[str, np.ndarray] = {}
        neck_base = np.array(raw_joints["neck"], dtype=np.float32)
        for name, xyz in raw_joints.items():
            p = np.array(xyz, dtype=np.float32)
            if name == "head":
                rel = p - neck_base
                rel = cls._rotate_x(rel, -spec.head_pitch_deg)
                rel = cls._rotate_y(rel, spec.head_azimuth_deg)
                base_neck_rot = cls._rotate_y(neck_base, spec.torso_azimuth_deg)
                world_joints[name] = base_neck_rot + rel
            else:
                world_joints[name] = cls._rotate_y(p, spec.torso_azimuth_deg)

        # Camera projection
        cam_dist = 2.5 if "medium" in spec.shot_scale else (2.9 if "three-quarter" in spec.shot_scale else 3.6)
        fov_scale = (float(spec.focal_length_mm) / 85.0) * (h * 0.78)
        cx_px = float(np.clip(spec.subject_x_norm, 0.15, 0.85)) * w
        cy_px = h * (0.58 if "three-quarter" in spec.shot_scale or "medium" in spec.shot_scale else 0.52)

        def project(pt3: np.ndarray) -> Tuple[int, int, float]:
            p_cam = cls._rotate_x(pt3 - np.array([0.0, 0.12, 0.0], dtype=np.float32), spec.camera_elevation_deg)
            z = max(0.4, cam_dist - float(p_cam[2]))
            px = int(round(cx_px + float(p_cam[0]) * fov_scale / z))
            py = int(round(cy_px - float(p_cam[1]) * fov_scale / z))
            return px, py, z

        # Subtle perspective floor grid
        for gx in np.linspace(-1.2, 1.2, 9):
            p1 = project(np.array([gx, -0.88, -0.8], dtype=np.float32))
            p2 = project(np.array([gx, -0.88, 0.8], dtype=np.float32))
            cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), (46, 50, 58), 1, cv2.LINE_AA)
        for gz in np.linspace(-0.8, 0.8, 7):
            p1 = project(np.array([-1.2, -0.88, gz], dtype=np.float32))
            p2 = project(np.array([1.2, -0.88, gz], dtype=np.float32))
            cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), (46, 50, 58), 1, cv2.LINE_AA)

        proj: Dict[str, Tuple[int, int, float]] = {k: project(v) for k, v in world_joints.items()}

        # Depth-sorted bones
        sorted_bones = sorted(
            BONE_EDGES,
            key=lambda b: 0.5 * (proj[b[0]][2] + proj[b[1]][2]),
            reverse=True,
        )
        for j1, j2, color_bgr in sorted_bones:
            x1, y1, z1 = proj[j1]
            x2, y2, z2 = proj[j2]
            thick = max(3, int(round(16.0 / max(1.0, 0.5 * (z1 + z2)))))
            cv2.line(img, (x1, y1), (x2, y2), color_bgr, thick, cv2.LINE_AA)

        for name, (jx, jy, jz) in proj.items():
            if name == "head":
                r_head = max(14, int(round(0.115 * fov_scale / jz)))
                cv2.circle(img, (jx, jy), r_head, (235, 238, 245), -1, cv2.LINE_AA)
                cv2.circle(img, (jx, jy), r_head, (90, 150, 255), 2, cv2.LINE_AA)
                # Nose / gaze direction vector on head sphere
                nose_local = cls._rotate_x(np.array([0.0, 0.0, 0.14], dtype=np.float32), -spec.head_pitch_deg)
                nose_world = world_joints["head"] + cls._rotate_y(nose_local, spec.head_azimuth_deg)
                nx, ny, nz = project(nose_world)
                if nz <= jz + 0.02:
                    cv2.arrowedLine(img, (jx, jy), (nx, ny), (60, 210, 255), 3, cv2.LINE_AA, tipLength=0.35)
            else:
                r_j = max(4, int(round(11.0 / max(1.0, jz))))
                cv2.circle(img, (jx, jy), r_j, (250, 250, 252), -1, cv2.LINE_AA)

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return img
