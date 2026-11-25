"""
릴프로필 (프로필 / 모델생성 — 정본 v1.0.0)
Zero-Hardcoding 1-Pass Full-Person 3D Synthesis, Mandatory 1-Hop 8-Dir Bank, Context-Adaptive 5-Lens x 5-Lighting Matrix, and 4K Optical Pore Mastering.
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
from .face_reattach import FaceLockCompositor, HeadRedrawCompositor, RelitOpticalFaceIntegrator
from .proportion_calibrator import (
    MicroProportionCalibrator,
    compare_14pt_proportions,
    extract_14pt_proportion_vector,
)
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
from .origin_guard import (
    ChainedGenerationViolationError,
    Incomplete8DirectionBankError,
    Mandatory8DirBankVerifier,
    OriginLineageViolationError,
    OriginOnlyOneShotGuard,
)
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
from .ocular_anatomy import OcularAnatomyAnalyzer, OcularAnatomyProfile
from .face_exaggeration_guard import FaceExaggerationGuard, FacialOrganExaggerationMetrics
from .pipeline import ModelGenerationPipeline, PortraitIdentityPipeline
from .fresh_origin_workflow import CandidateScore, FreshOriginEditorialWorkflow
from .headless_mesh_composer import HeadlessMeshComposer
from .version_guard import (
    CANONICAL_PROJECT_NAME,
    PROJECT_ALIASES,
    SHORT_PROJECT_NAME,
    VersionIntegrityReport,
    check_version_integrity,
    load_version_ledger,
    resolve_project_alias,
    sync_or_bump_version,
)

__all__ = [
    "CANONICAL_PROJECT_NAME",
    "SHORT_PROJECT_NAME",
    "PROJECT_ALIASES",
    "VersionIntegrityReport",
    "check_version_integrity",
    "load_version_ledger",
    "resolve_project_alias",
    "sync_or_bump_version",
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
    "RelitOpticalFaceIntegrator",
    "MicroProportionCalibrator",
    "extract_14pt_proportion_vector",
    "compare_14pt_proportions",
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
    "Incomplete8DirectionBankError",
    "ChainedGenerationViolationError",
    "Mandatory8DirBankVerifier",
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
    "OcularAnatomyAnalyzer",
    "OcularAnatomyProfile",
    "FaceExaggerationGuard",
    "FacialOrganExaggerationMetrics",
    "ModelGenerationPipeline",
    "PortraitIdentityPipeline",
    "CandidateScore",
    "FreshOriginEditorialWorkflow",
    "HeadlessMeshComposer",
]
__project_name__ = "릴프로필"
__project_aliases__ = ("릴프로필", "프로필", "모델생성")
__version__ = "1.0.10"