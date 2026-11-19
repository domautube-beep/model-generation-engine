# Project Rule: 모델생성 (`model-generation-engine` v2.0)

## 1. 2-Tier Hybrid Cloud Architecture
- **Tier 1 (Code, Tests, Rules, Skills, Workflows — GitHub & iCloud `< 50KB`)**:
  - Local Path: `/Volumes/DOMA/Project/모델생성` (Symlink: `/Users/doma/Projects/모델생성`)
  - GitHub Repository: `https://github.com/domautube-beep/model-generation-engine`
  - Contains ONLY modular Python source code (`model_generation_engine/`), automated tests (`tests/`), rules (`GEMINI.md`), workflows (`.agents/workflows/`), and lightweight skills (`.agents/skills/`).
- **Tier 2 (Heavy Binary Models, Reference Media, 5K Deliverables — Google Drive)**:
  - Local Google Drive Vault: `/Users/doma/Library/CloudStorage/GoogleDrive-domautube@gmail.com/내 드라이브/Antigravity_Assets/모델생성`
  - Subdirectories: `models/` (`realesrgan-ncnn-vulkan`, `4xNomos8kSC.*`, `yunet.onnx`, `codeformer.onnx`), `resources/`, and `outputs/`.
  - Never commit `.onnx`, `.bin`, `.param`, binaries, or large images to Git or iCloud Drive.

## 2. Universal Zero-Hardcoding Mandates (5 Axes)
1. **Universal Model Axis (`SubjectProfile`)**:
   - Never hardcode gender pronouns (`"man"`, `"woman"`, `"his"`, `"her"`), specific nationalities, specific glasses (`"browline glasses"`), or fixed height/weight (`175cm, 80kg`) inside core classes or prompt templates.
   - Always build prompts dynamically from `SubjectProfile` fields (`identity_tag`, optional `height_cm`/`weight_kg`/`build_descriptor`, optional `hairstyle`, optional `eyewear`, `expression`).
2. **Universal Outfit, Location, and Angle Axes (`WardrobeSpec`, `LocationSpec`, `CameraRigSpec`)**:
   - Never hardcode a specific clothing item (`"black jacket"`), background (`"doorway"` or `"terracotta wall"`), or seated pose inside the prompt builder or pipeline.
   - Support arbitrary user-provided `WardrobeSpec`, `LocationSpec`, and `CameraRigSpec` instances as well as multi-domain catalogs (`UniversalVariationPlanner`).
3. **Dynamic Landmark & Adaptive Skin Tone Axis (`DynamicFaceDetector` & `AdaptiveSkinToneProfile`)**:
   - Never hardcode pixel coordinates `(x1, y1, x2, y2)` or fixed CIE LAB skin-color thresholds (`L > 125, A > 131, B > 133`).
   - Always detect 5-point landmarks via `YuNet`, estimate 3D head pose (`yaw_asymmetry_ratio`), and sample the subject's actual skin CIE LAB median and IQR dynamically (`sample_adaptive_skin_profile`).
4. **Single-Pass From-Scratch Lineage Axis (`SinglePassGenerationGuard`)**:
   - Never feed an AI-generated image back into `image-generator` (`generation_depth == 0`).
   - Never pass a mood/composition reference photo containing another person's face alongside the identity reference photo.
5. **Halo-Free Matte Skin & Auto-Balanced 5K Canvas Axis (`MatteSkinProcessor` & `Seamless5KCompositor`)**:
   - Use scalar percentile (`P88`) soft-knee specular roll-off on skin pixels (never spatial Gaussian blur across dark hairlines) + scale-adaptive pore micro-texture (`inter_ocular_distance / 110.0`).
   - When `pad_left_ratio=None`, automatically detect subject horizontal placement via `YuNet` and expand to `5040x2160` (`21:9`) with a 240px smooth cosine ramp.
