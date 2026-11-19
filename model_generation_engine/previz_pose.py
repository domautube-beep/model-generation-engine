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

from .identity_turnaround import normalize_frontal_yaw_deg, resolve_8angle_view_from_azimuth

POSE_PRESETS: Dict[str, Dict[str, object]] = {
    "standing_editorial": {
        "label_ko": "스탠딩 에디토리얼 (0° 정면)",
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
        "label_ko": "라운지 체어 착석 (-22.5° 좌 미세)",
        "shot_scale": "three-quarter seated portrait shot",
        "pose_text": "seated comfortably in an armchair with torso gently angled, hands resting calmly on lap/knee",
        "torso_azimuth_deg": -22.5,
        "head_azimuth_deg": -22.5,
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
        "label_ko": "좌 표준 ¾ (-45° 스탠딩)",
        "shot_scale": "medium three-quarter standing portrait shot",
        "pose_text": "standing in a composed 45-degree left-facing three-quarter turn with calm forward-left gaze",
        "torso_azimuth_deg": -45.0,
        "head_azimuth_deg": -45.0,
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
        "label_ko": "우 표준 ¾ (+45° 스탠딩)",
        "shot_scale": "medium three-quarter standing portrait shot",
        "pose_text": "standing in a composed 45-degree right-facing three-quarter turn with calm forward-right gaze",
        "torso_azimuth_deg": 45.0,
        "head_azimuth_deg": 45.0,
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
        "label_ko": "풀바디 워킹 (+22.5° 우 미세)",
        "shot_scale": "full-body editorial walking shot",
        "pose_text": "walking forward in a confident mid-stride with natural arm swing and balanced posture",
        "torso_azimuth_deg": 22.5,
        "head_azimuth_deg": 22.5,
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
        "label_ko": "팔짱 프로필 (-67.5° 좌 깊은¾)",
        "shot_scale": "medium three-quarter portrait shot",
        "pose_text": "standing with arms neatly folded across chest, relaxed confident shoulders",
        "torso_azimuth_deg": -55.0,
        "head_azimuth_deg": -67.5,
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
        "label_ko": "한손 포켓 (+67.5° 우 깊은¾)",
        "shot_scale": "three-quarter standing portrait shot",
        "pose_text": "standing with one hand tucked casually into trouser pocket and the other arm relaxed",
        "torso_azimuth_deg": 55.0,
        "head_azimuth_deg": 67.5,
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
    "full_profile_editorial": {
        "label_ko": "완전 측면 실루엣 (-90° / +90°)",
        "shot_scale": "medium side-profile editorial shot",
        "pose_text": "standing in a clean side-profile silhouette with composed horizontal gaze",
        "torso_azimuth_deg": -90.0,
        "head_azimuth_deg": -90.0,
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
    "seated_desk_focus": {
        "label_ko": "데스크·태블릿 몰입 착석 (+45° 우¾ · 하방 시선)",
        "shot_scale": "medium waist-up seated candid editorial shot",
        "pose_text": (
            "seated at a table leaning naturally forward from the waist with forearms resting comfortably on the tabletop, "
            "one hand holding a pen/stylus in active focus while the other hand steadies a tablet/notebook on the surface, "
            "shoulders relaxed with authentic weight grounded onto the table"
        ),
        "torso_azimuth_deg": 35.0,
        "head_azimuth_deg": 45.0,
        "head_pitch_deg": -32.0,
        "head_roll_deg": 12.0,
        "gaze_mode": "downward_task_focus",
        "joints": {
            "pelvis": (0.0, -0.14, -0.04),
            "neck": (0.04, 0.42, 0.10),
            "head": (0.08, 0.61, 0.18),
            "l_shoulder": (-0.18, 0.39, 0.08),
            "r_shoulder": (0.24, 0.37, 0.08),
            "l_elbow": (-0.12, 0.14, 0.24),
            "r_elbow": (0.28, 0.12, 0.22),
            "l_wrist": (0.04, 0.06, 0.36),
            "r_wrist": (0.24, 0.05, 0.36),
            "l_hip": (-0.12, -0.16, -0.04),
            "r_hip": (0.12, -0.16, -0.04),
            "l_knee": (-0.10, -0.20, 0.34),
            "r_knee": (0.14, -0.20, 0.34),
            "l_ankle": (-0.10, -0.64, 0.34),
            "r_ankle": (0.14, -0.64, 0.34),
        },
    },
}

GAZE_MODES: Dict[str, str] = {
    "auto": "",
    "downward_task_focus": (
        "natural candid downward gaze directed at the hands/tablet/object below with relaxed semi-lowered upper eyelids "
        "and calm unposed concentration—ZERO camera eye contact, ZERO stiff horizontal stare"
    ),
    "off_camera_natural": (
        "relaxed off-camera candid gaze looking naturally along the head's turn direction toward the side/distance "
        "with unforced eyelids—not staring into the camera lens"
    ),
    "camera_eye_contact": (
        "calm, lifelike direct eye contact toward the camera lens with natural relaxed eyelids and subtle presence"
    ),
    "side_window_contemplative": (
        "quiet contemplative gaze directed toward the natural window light source with serene relaxed expression"
    ),
    "upward_contemplative": (
        "gentle upward-angled thoughtful gaze with natural neck extension and relaxed brow"
    ),
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
    head_roll_deg: float = 0.0
    gaze_mode: str = "auto"  # "auto" | "downward_task_focus" | "off_camera_natural" | "camera_eye_contact" | "side_window_contemplative" | "upward_contemplative"
    torso_azimuth_deg: float = 0.0
    camera_elevation_deg: float = 0.0
    focal_length_mm: int = 85
    subject_x_norm: float = 0.58
    shot_scale: str = "three-quarter portrait shot"
    custom_pose_note: str = ""
    joints_override: Optional[Dict[str, Tuple[float, float, float]]] = None
    bag_carry_mode: str = ""  # e.g. "hand_right", "shoulder_left", "crossbody_l2r", "placed_beside"
    bag_size: str = "medium_standard"  # "micro_mini" | "small_compact" | "medium_standard" | "large_tote_weekender"

    @property
    def matched_8angle_view(self) -> str:
        return resolve_8angle_view_from_azimuth(self.head_azimuth_deg)

    def resolved_gaze_mode(self) -> str:
        if self.gaze_mode in GAZE_MODES and self.gaze_mode != "auto":
            return self.gaze_mode
        preset = POSE_PRESETS.get(self.preset_id, {})
        if preset.get("gaze_mode") and self.head_pitch_deg <= -12.0:
            return str(preset["gaze_mode"])
        if self.head_pitch_deg <= -14.0:
            return "downward_task_focus"
        if self.head_pitch_deg >= 14.0:
            return "upward_contemplative"
        if abs(self.head_azimuth_deg) >= 20.0:
            return "off_camera_natural"
        return "camera_eye_contact"

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
        pitch_note = ""
        if self.head_pitch_deg <= -12.0:
            pitch_note = f", head bowed naturally downward {abs(self.head_pitch_deg):.0f} deg from the cervical neck"
        elif self.head_pitch_deg >= 12.0:
            pitch_note = f", chin tilted gently upward {self.head_pitch_deg:.0f} deg"
        roll_note = (
            f", subtle natural head tilt {self.head_roll_deg:+.0f} deg"
            if abs(self.head_roll_deg) >= 4.0
            else ""
        )
        gaze_txt = GAZE_MODES.get(self.resolved_gaze_mode(), GAZE_MODES["camera_eye_contact"])
        return (
            f"{scale_txt} captured from an {self.camera_angle_clause()} with an {self.focal_length_mm}mm prime lens; "
            f"pose: {base_pose} (torso azimuth {self.torso_azimuth_deg:.0f} deg, head azimuth {self.head_azimuth_deg:.0f} deg"
            f"{pitch_note}{roll_note}); gaze & eyelid kinematics: {gaze_txt}; composition: {self.placement_clause()}"
        )


class Previz3DRenderer:
    """Renders a clean 3D articulated mannequin guide image for visual previz and pose-conditioned generation."""

    @staticmethod
    def _rotate_y(pt: np.ndarray, deg: float) -> np.ndarray:
        rad = math.radians(normalize_frontal_yaw_deg(deg))
        c, s = math.cos(rad), math.sin(rad)
        # Signed frontal yaw: negative (-90..0) turns toward image-left (-X), positive (0..+90) turns toward image-right (+X)
        return np.array([c * pt[0] + s * pt[2], pt[1], -s * pt[0] + c * pt[2]], dtype=np.float32)

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

        # Auto-harmonize hand/elbow posture when carrying a bag by hand or crook of arm (unless manually overridden)
        if spec.bag_carry_mode == "hand_right":
            raw_joints["r_elbow"] = (0.25, 0.23, 0.04)
            raw_joints["r_wrist"] = (0.25, -0.05, 0.06)
        elif spec.bag_carry_mode == "hand_left":
            raw_joints["l_elbow"] = (-0.25, 0.23, 0.04)
            raw_joints["l_wrist"] = (-0.25, -0.05, 0.06)
        elif spec.bag_carry_mode == "crook_right":
            raw_joints["r_elbow"] = (0.25, 0.20, 0.06)
            raw_joints["r_wrist"] = (0.14, 0.24, 0.20)
        elif spec.bag_carry_mode == "crook_left":
            raw_joints["l_elbow"] = (-0.25, 0.20, 0.06)
            raw_joints["l_wrist"] = (-0.14, 0.24, 0.20)

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

        # Optional proportional 3D Bag Prop indicator on the Previz guide
        if spec.bag_carry_mode:
            sz_mul = {
                "micro_mini": 0.52,
                "small_compact": 0.74,
                "medium_standard": 1.0,
                "large_tote_weekender": 1.32,
            }.get(spec.bag_size, 1.0)
            bw_3d = 0.24 * sz_mul
            bh_3d = 0.20 * sz_mul
            anchor_pt: Optional[np.ndarray] = None
            strap_from: Optional[np.ndarray] = None

            if spec.bag_carry_mode == "hand_right":
                anchor_pt = world_joints["r_wrist"] + np.array([0.0, -0.08 - bh_3d * 0.5, 0.0], dtype=np.float32)
                strap_from = world_joints["r_wrist"]
            elif spec.bag_carry_mode == "hand_left":
                anchor_pt = world_joints["l_wrist"] + np.array([0.0, -0.08 - bh_3d * 0.5, 0.0], dtype=np.float32)
                strap_from = world_joints["l_wrist"]
            elif spec.bag_carry_mode == "crook_right":
                anchor_pt = world_joints["r_elbow"] + np.array([0.02, -0.10 - bh_3d * 0.5, 0.04], dtype=np.float32)
                strap_from = world_joints["r_elbow"]
            elif spec.bag_carry_mode == "crook_left":
                anchor_pt = world_joints["l_elbow"] + np.array([-0.02, -0.10 - bh_3d * 0.5, 0.04], dtype=np.float32)
                strap_from = world_joints["l_elbow"]
            elif spec.bag_carry_mode == "shoulder_right":
                anchor_pt = world_joints["r_hip"] + np.array([0.10, 0.06, 0.0], dtype=np.float32)
                strap_from = world_joints["r_shoulder"]
            elif spec.bag_carry_mode == "shoulder_left":
                anchor_pt = world_joints["l_hip"] + np.array([-0.10, 0.06, 0.0], dtype=np.float32)
                strap_from = world_joints["l_shoulder"]
            elif spec.bag_carry_mode == "crossbody_l2r":
                anchor_pt = world_joints["r_hip"] + np.array([0.10, 0.02, 0.04], dtype=np.float32)
                strap_from = world_joints["l_shoulder"]
            elif spec.bag_carry_mode == "crossbody_r2l":
                anchor_pt = world_joints["l_hip"] + np.array([-0.10, 0.02, 0.04], dtype=np.float32)
                strap_from = world_joints["r_shoulder"]
            elif spec.bag_carry_mode == "underarm_clutch":
                anchor_pt = world_joints["r_elbow"] + np.array([-0.04, 0.04, 0.06], dtype=np.float32)
            elif spec.bag_carry_mode == "placed_beside":
                anchor_pt = np.array([0.44, -0.88 + bh_3d * 0.5, 0.08], dtype=np.float32)

            if anchor_pt is not None:
                bx, by, bz = project(anchor_pt)
                pw = max(12, int(round((bw_3d * fov_scale) / bz)))
                ph = max(10, int(round((bh_3d * fov_scale) / bz)))
                if strap_from is not None:
                    sx, sy, _ = project(strap_from)
                    cv2.line(img, (sx, sy), (bx, by - ph // 2), (70, 190, 245), 2, cv2.LINE_AA)
                cv2.rectangle(
                    img,
                    (bx - pw // 2, by - ph // 2),
                    (bx + pw // 2, by + ph // 2),
                    (55, 165, 235),
                    2,
                    cv2.LINE_AA,
                )

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return img
