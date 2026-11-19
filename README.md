# 모델생성 (`model-generation-engine` v2.0)

> **100% Universal, Zero-Hardcoding Model Synthesis & 5K Editorial Hero Engine**  
> Designed to work seamlessly for **any new model (`SubjectProfile`)**, **any new outfit (`WardrobeSpec`)**, **any new location (`LocationSpec`)**, and **any new camera angle & framing (`CameraRigSpec`)**.

---

## 1. Core Design Philosophy (Zero Hardcoding Across 5 Axes)

| Axis | Legacy Hardcoding Pitfall | Universal `모델생성` v2.0 Architecture |
| :--- | :--- | :--- |
| **1. Model (`SubjectProfile`)** | Hardcoded gender (`"Korean male"`, `"his"`), hardcoded glasses (`"browline glasses"`), or fixed `175cm / 80kg` build | Gender-neutral default (`"human subject in the reference portrait"`), optional `height_cm` / `weight_kg`, dynamic BMI-based physique descriptor, and 1:1 reference inheritance for eyewear & hairstyle unless explicitly overridden |
| **2. Outfit (`WardrobeSpec`)** | Hardcoded `"black jacket"` or single preset dictionary | Composable `WardrobeSpec` supporting arbitrary custom outfits or multi-category wardrobe catalogs (`tailored_editorial`, `smart_casual`, `architectural_outerwear`) |
| **3. Location (`LocationSpec`)** | Hardcoded `"doorway"` or `"terracotta wall"` | Composable `LocationSpec` supporting any indoor/outdoor environment, props, lighting, and surface textures (`studio_editorial`, `architectural_interior`, `outdoor_natural`, `urban_cinematic`, or custom) |
| **4. Camera & Angle (`CameraRigSpec`)** | Fixed frontal three-quarter seated framing | Composable `CameraRigSpec` supporting any camera angle (eye-level, low-angle, high-angle, 3/4 profile, side-profile), shot scale, pose, and lens focal length |
| **5. Skin & Canvas (`AdaptiveSkinToneProfile` & `Seamless5KCompositor`)** | Hardcoded LAB skin thresholds (`L>125, A>131, B>133`) and fixed `pad_left_ratio=0.65` | Dynamic landmark-sampled CIE LAB skin tone distribution (`sample_adaptive_skin_profile`), 3D head-yaw-aware skin masking, scale-adaptive pore micro-texture (`iod / 110.0`), and auto-detected horizontal padding (`pad_left_ratio=None`) |

---

## 2. The 7-Stage Universal Progression Workflow

1. **Stage 1 — Reference Isolation (`SinglePassGenerationGuard.validate_reference_lineage`)**:
   - Inspects the input identity reference photo via `YuNet` (`DynamicFaceDetector`).
   - Blocks any mood/style reference photo containing another person's face from being passed alongside the identity reference (`ReferenceIsolationError`).
   - Blocks any AI-generated output from being re-fed into generation (`GenerationalDegradationError`, `generation_depth == 0`).
2. **Stage 2 — Universal Multi-Axis Matrix Planning (`UniversalVariationPlanner.plan_variations`)**:
   - Generates a structured matrix of `VariationBrief` objects combining any `SubjectProfile` with any custom or catalog `WardrobeSpec`, `LocationSpec`, and `CameraRigSpec`.
3. **Stage 3 — Universal Prompt Firewall (`PromptFirewall.build_universal_single_pass_prompt`)**:
   - Dynamically constructs a single-pass from-scratch prompt from the 4 axes (`subject`, `wardrobe`, `location`, `camera`) with zero hardcoded gender, glasses, outfit, or location assumptions.
   - Enforces the **Body-vs-Face Separation Firewall** (physique specs apply strictly to torso/shoulders, never widening cheeks/jawline) and **Natural Matte Skin Expression Firewall** (eliminates plastic/oily specular gloss while preserving subtle studio pore micro-texture without forced wrinkles).
4. **Stage 4 — Single-Pass From-Scratch Generation**:
   - Invokes the image generator in a single pass from scratch (`generation_depth == 0`) using ONLY the original identity reference image.
5. **Stage 5 — 4x Native Super-Resolution (`SuperResolutionEngine`)**:
   - Runs `realesrgan-ncnn-vulkan` with `4xNomos8kSC` at 4x native scale without intermediate JPEG re-encoding.
6. **Stage 6 — Adaptive Halo-Free Matte Skin & Scale-Adaptive Pore Micro-Texture (`MatteSkinProcessor`)**:
   - Samples the subject's actual skin CIE LAB distribution from the detected cheek and nose-bridge landmarks (`AdaptiveSkinToneProfile`).
   - Builds a 3D-yaw-adaptive skin mask excluding glasses/eyes, mouth, hair, and background.
   - Applies a **halo-free scalar percentile (`P88`) soft-knee specular roll-off** (preventing dark boundary halos near dark hairlines or collars), recovers warm skin chrominance on de-glossed highlights, and injects subtle scale-adaptive pore micro-texture (`inter_ocular_distance / 110.0`).
7. **Stage 7 — Auto-Balanced 5K (`5040x2160`) 21:9 Outpainting & 6 Quality Gates (`Seamless5KCompositor` & `QualityGateAuditor`)**:
   - Auto-detects the subject's horizontal placement via `YuNet` when `pad_left_ratio=None` and expands the canvas to `5040x2160` (`21:9`) with a 240px smooth cosine vignette ramp.
   - Verifies all 6 automated quality gates (`Resolution 5040x2160`, `Face Landmarks`, `Sharpness`, `Seam Continuity`, `Natural Skin Texture`, `Zero Plastic Gloss`).

---

## 3. CLI Usage Examples

```bash
# 1. Plan variations across any model, outfit, location, and camera angle
PYTHONPATH=. uv run --no-project --with opencv-python-headless --with onnxruntime \
  python3 -m model_generation_engine.cli plan \
  --reference-face "/Users/doma/Library/CloudStorage/GoogleDrive-domautube@gmail.com/내 드라이브/Antigravity_Assets/모델생성/resources/ref_person_face.jpg" \
  --identity-tag "Female model in her late 20s" \
  --height-cm 168 --weight-kg 52 \
  --hairstyle "sleek shoulder-length dark brown hair" \
  --eyewear "no glasses, bare eyes" \
  --wardrobe-category smart_casual \
  --location-category architectural_interior \
  --camera-category multi_angle_portfolio \
  --count 4

# 2. Run the full post-generation 5K pipeline on a single-pass raw image
PYTHONPATH=. uv run --no-project --with opencv-python-headless --with onnxruntime \
  python3 -m model_generation_engine.cli run \
  --input raw_single_pass.jpg \
  --output "/Users/doma/Library/CloudStorage/GoogleDrive-domautube@gmail.com/내 드라이브/Antigravity_Assets/모델생성/outputs/hero_5k.jpg" \
  --verify-crop "/Users/doma/Library/CloudStorage/GoogleDrive-domautube@gmail.com/내 드라이브/Antigravity_Assets/모델생성/outputs/closeup_verify.jpg" \
  --reference-face "/Users/doma/Library/CloudStorage/GoogleDrive-domautube@gmail.com/내 드라이브/Antigravity_Assets/모델생성/resources/ref_person_face.jpg"
```
