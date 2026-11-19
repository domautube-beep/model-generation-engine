"""
모델생성 (Universal Model Generation & 5K Super-Resolution Engine v2.0)
Zero-Hardcoding Universal Model Synthesis across ANY Subject, Wardrobe, Location, and Camera Angle.
"""

from .config import (
    CameraRigSpec,
    CanvasSpec,
    EngineConfig,
    LocationSpec,
    SubjectProfile,
    SubjectSpec,
    WardrobeSpec,
    resolve_gdrive_vault,
)
from .prompt_builder import PromptContaminationError, PromptFirewall
from .detector import AdaptiveSkinToneProfile, DynamicFaceDetector, FaceDetectionResult
from .skin_processor import MatteSkinProcessor, SkinQualityMetrics
from .workflow import (
    GenerationalDegradationError,
    ReferenceIsolationError,
    SinglePassGenerationGuard,
    UniversalVariationPlanner,
    VariationMatrixPlanner,
    VariationSpec,
)
from .upscaler import SuperResolutionEngine
from .compositor import Seamless5KCompositor
from .apple_vision import AppleVisionAnalyzer, AppleVisionError
from .face_reattach import FaceLockCompositor, HeadRedrawCompositor
from .identity_turnaround import (
    TURNAROUND_5_VIEWS,
    TURNAROUND_8_VIEWS,
    TURNAROUND_FULL_SWEEP,
    TURNAROUND_VIEWS,
    AnatomicalBlueprint,
    RealPersonTurnaroundEngine,
    normalize_frontal_yaw_deg,
    resolve_8angle_view_from_azimuth,
)
from .origin_guard import OriginLineageViolationError, OriginOnlyOneShotGuard
from .previz_pose import GAZE_MODES, POSE_PRESETS, PosePrevizSpec, Previz3DRenderer
from .scene_radiometry import (
    APERTURE_DOF_DESCRIPTIONS,
    APERTURE_MODES,
    AutonomousDoPDirector,
    CAMERA_PROFILES,
    COHESION_MASTERING_MODES,
    FILM_PROFILES,
    LIGHTING_MODES,
    PostGenCohesionMasterer,
    SceneRadiometryAnalyzer,
    SingleLensRadiometricSpec,
)
from .multi_asset_composer import (
    AttachmentItem,
    ComposedSceneJob,
    MultiSceneBatchComposer,
    SceneLocationInput,
)
from .quality_gate import QualityGateAuditor, QualityReport
from .pipeline import ModelGenerationPipeline, PortraitIdentityPipeline

__all__ = [
    "EngineConfig",
    "SubjectProfile",
    "SubjectSpec",
    "WardrobeSpec",
    "LocationSpec",
    "CameraRigSpec",
    "CanvasSpec",
    "resolve_gdrive_vault",
    "PromptFirewall",
    "PromptContaminationError",
    "AdaptiveSkinToneProfile",
    "DynamicFaceDetector",
    "FaceDetectionResult",
    "MatteSkinProcessor",
    "SkinQualityMetrics",
    "GenerationalDegradationError",
    "ReferenceIsolationError",
    "SinglePassGenerationGuard",
    "UniversalVariationPlanner",
    "VariationMatrixPlanner",
    "VariationSpec",
    "SuperResolutionEngine",
    "Seamless5KCompositor",
    "AppleVisionAnalyzer",
    "AppleVisionError",
    "FaceLockCompositor",
    "HeadRedrawCompositor",
    "TURNAROUND_5_VIEWS",
    "TURNAROUND_8_VIEWS",
    "TURNAROUND_FULL_SWEEP",
    "TURNAROUND_VIEWS",
    "AnatomicalBlueprint",
    "RealPersonTurnaroundEngine",
    "normalize_frontal_yaw_deg",
    "resolve_8angle_view_from_azimuth",
    "OriginOnlyOneShotGuard",
    "OriginLineageViolationError",
    "GAZE_MODES",
    "POSE_PRESETS",
    "PosePrevizSpec",
    "Previz3DRenderer",
    "LIGHTING_MODES",
    "APERTURE_MODES",
    "APERTURE_DOF_DESCRIPTIONS",
    "CAMERA_PROFILES",
    "FILM_PROFILES",
    "COHESION_MASTERING_MODES",
    "AutonomousDoPDirector",
    "SingleLensRadiometricSpec",
    "SceneRadiometryAnalyzer",
    "PostGenCohesionMasterer",
    "AttachmentItem",
    "SceneLocationInput",
    "ComposedSceneJob",
    "MultiSceneBatchComposer",
    "QualityGateAuditor",
    "QualityReport",
    "ModelGenerationPipeline",
    "PortraitIdentityPipeline",
]
__version__ = "2.0.0"
