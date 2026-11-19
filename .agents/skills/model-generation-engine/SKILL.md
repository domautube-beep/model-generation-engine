---
name: model-generation-engine
description: Use this skill immediately whenever the user talks about creating, generating, editing, or upscaling model/portrait/editorial hero images ('모델생성' 프로젝트, /Volumes/DOMA/Project/모델생성 및 /Users/doma/Projects/모델생성, GitHub domautube-beep/model-generation-engine), or asks to generate a person/model with any new face, new outfit, new indoor/outdoor location, new props, or new camera angle. Enforces the 100% Universal Zero-Hardcoding 7-Stage Progression Workflow: Reference Isolation (single identity photo only), Universal 4-Axis Matrix Planning (SubjectProfile x WardrobeSpec x LocationSpec x CameraRigSpec), Universal Prompt Firewall (Body-vs-Face & Natural Matte Skin), Single-Pass From-Scratch Generation (generation_depth == 0, Zero Re-Feeding), 4x Real-ESRGAN (4xNomos8kSC) Super-Resolution, Adaptive Skin-Tone & 3D-Yaw Halo-Free P88 Matte Skin Specular Roll-Off + Scale-Adaptive Pore Micro-Texture, Auto-Balanced 240px Cosine-Ramp 5K (5040x2160) 21:9 Outpainting, and 6 Automated Quality Gates.
---

# Universal Model Generation Engine (`모델생성` — `model-generation-engine` v2.0)

## 1. Hybrid Cloud Paths & Environment
- **Active Workspace**: `/Volumes/DOMA/Project/모델생성` (Symlink: `/Users/doma/Projects/모델생성`)
- **GitHub Repository**: `https://github.com/domautube-beep/model-generation-engine`
- **Google Drive Heavy Asset Vault**: `/Users/doma/Library/CloudStorage/GoogleDrive-domautube@gmail.com/내 드라이브/Antigravity_Assets/모델생성`
  - `models/`: `realesrgan-ncnn-vulkan`, `4xNomos8kSC.param`, `4xNomos8kSC.bin`, `yunet.onnx`, `codeformer.onnx`
  - `resources/`: Reference portraits (`ref_person_face.jpg`, etc.)
  - `outputs/`: 5K `5040x2160` 21:9 hero images and close-up verification crops
- **Execution Command Rule**: Because `/Volumes/DOMA` is an external volume that blocks `.venv` creation (`os error 1`), always execute via `uv run --no-project`:
  ```bash
  PYTHONPATH="/Volumes/DOMA/Project/모델생성" /Users/doma/.local/bin/uv run --no-project --with opencv-python-headless --with onnxruntime python3 -m model_generation_engine.cli ...
  ```

---

## 2. Universal Zero-Hardcoding Architecture (4 Input Axes + 2 Dynamic Processing Axes)

1. **Model Axis (`SubjectProfile`)**:
   - Works for **any person** (any gender, age, ethnicity, eyewear state, hairstyle, or body build).
   - Never hardcode `"Korean male"`, `"his"`, `"browline glasses"`, or `175cm / 80kg` in prompt templates.
   - `height_cm` and `weight_kg` are optional; if provided, BMI dynamically determines the torso/shoulder build descriptor (`slender`, `balanced`, `sturdy`, `broad`). If omitted, body proportions follow a natural editorial build.
   - `eyewear` and `hairstyle` are optional; if omitted, the prompt explicitly inherits eyewear presence/absence and hairstyle 1:1 from the reference portrait.
2. **Outfit Axis (`WardrobeSpec`)**:
   - Composable `outerwear`, `inner_layer`, `trousers`, `accessories`, `fabric_notes`.
   - Supports arbitrary custom outfits or built-in catalog categories (`tailored_editorial`, `smart_casual`, `architectural_outerwear`).
3. **Location Axis (`LocationSpec`)**:
   - Composable `setting_type`, `backdrop`, `architecture_or_furniture`, `props`, `lighting`, `floor_or_ground`.
   - Supports arbitrary custom locations or built-in catalog categories (`studio_editorial`, `architectural_interior`, `outdoor_natural`, `urban_cinematic`).
4. **Camera & Angle Axis (`CameraRigSpec`)**:
   - Composable `shot_scale`, `camera_angle`, `subject_pose`, `composition_placement`, `lens_mm`, `depth_of_field`.
   - Supports arbitrary custom angles (eye-level, low-angle, high-angle, 3/4 profile, side-profile) or built-in catalog categories (`seated_editorial`, `multi_angle_portfolio`).
5. **Adaptive Skin-Tone & 3D-Pose Axis (`DynamicFaceDetector` & `AdaptiveSkinToneProfile`)**:
   - Never hardcode pixel coordinates `(x1, y1, x2, y2)` or fixed CIE LAB skin-color thresholds (`L>125, A>131, B>133`).
   - Dynamically detects 5-point landmarks via `YuNet`, computes 3D head yaw (`yaw_asymmetry_ratio`), samples the subject's actual skin CIE LAB median & IQR from cheek/nose landmarks (`sample_adaptive_skin_profile`), and scales pore micro-texture wavelengths by `inter_ocular_distance / 110.0`.
6. **Auto-Balanced 5K Composition Axis (`Seamless5KCompositor`)**:
   - When `pad_left_ratio=None` (default), automatically detects the subject's horizontal placement via `YuNet` (`norm_x > 0.52 -> 0.65`, `norm_x < 0.48 -> 0.35`, else `0.50`) and expands to `5040x2160` (`21:9`) with a 240px smooth cosine vignette ramp.

---

## 3. Mandatory 7-Stage Progression Workflow

### Stage 1: Reference Isolation (`SinglePassGenerationGuard.validate_reference_lineage`)
- Verify the identity reference portrait contains a valid face (`YuNet` 5-point landmarks) and is **not** a previously generated output (`generation_depth == 0`).
- If the user uploads a mood/pose reference photo containing another person, extract only its text attributes (`LocationSpec`, `WardrobeSpec`, `CameraRigSpec`) and **never pass the mood photo alongside the identity photo** to `image-generator`.

### Stage 2: Universal Multi-Axis Matrix Planning (`UniversalVariationPlanner.plan_variations`)
- Generate a structured matrix of `VariationBrief` objects combining `SubjectProfile` with custom or catalog `WardrobeSpec`, `LocationSpec`, and `CameraRigSpec`.

### Stage 3: Universal Prompt Firewall (`PromptFirewall.build_universal_single_pass_prompt`)
- Every single-pass generation prompt must pass `PromptFirewall.validate_prompt(prompt)`:
  - **Body-vs-Face Firewall**: Any physique spec applies strictly to torso, shoulders, chest, and clothing drape, never inflating cheeks or jawline.
  - **Natural Matte Skin Expression Firewall**: Enforces natural studio-powdered matte skin with subtle real skin pore micro-texture while forbidding plastic/oily specular gloss and forced wrinkles.

### Stage 4: Single-Pass From-Scratch Generation (`generation_depth == 0`)
- Invoke `image-generator` in a **single pass from scratch**, passing ONLY the original identity reference image (`ref_person_face.jpg`). Never re-feed a generated image back into `image-generator`.

### Stage 5: 4x Native Super-Resolution & Scale-Adaptive Anti-Smudge Focus Recovery (`SuperResolutionEngine`)
- Upscale the single-pass raw image 4x using `realesrgan-ncnn-vulkan` (`4xNomos8kSC`).
- **Scale-Adaptive Anti-Smudge Rescue**:
  - For medium/close-up shots (`inter_ocular_distance >= 90px` on 5K and high per-organ edge energy), preserve the pure 4x Real-ESRGAN detail directly without down-warping through a `512x512` bottleneck.
  - For distant/full-body wide shots (`inter_ocular_distance < 90px` on 5K or weak organ edge energy), automatically execute `restore_face_focus_5pt` with **Ultra-High-Frequency Residual Recovery (`warped_back + 0.65 * (patch - warped_orig_back)`)** so zero double-interpolation blur occurs.
  - **3D Yaw Anti-Smudge Guard**: When head yaw is steep 3/4 or side-profile (`abs(yaw_asymmetry_ratio) > 0.28`), bypass frontal `FFHQ_TEMPLATE_512` warping (which stretches/crushes angled faces) and apply pose-invariant multi-scale anatomical crisping.

### Stage 6: Adaptive Halo-Free Matte Skin, Organ Exclusion & Organ Micro-Crisping (`MatteSkinProcessor.process`)
- Dynamically sample the subject's skin CIE LAB profile (`sample_adaptive_skin_profile`), build a 3D-yaw-adaptive skin mask that **explicitly excludes eyes/eyelids/eyewear, eyebrow arches, nostrils/alar base, and lips** so 3D facial organ shading never flattens.
- Apply scalar `P88` percentile soft-knee specular roll-off (preventing forehead/hairline shadow halos), recover warm chrominance, blend scale-adaptive pore micro-texture (`iod / 110.0`), and apply **Organ-Selective Micro-Crisping (`build_anatomical_organ_mask`)** strictly on the eyes, eyebrows, nostrils, and lip contour.

### Stage 7: Auto-Balanced 5K (`5040x2160`) 21:9 Outpainting & 7 Anti-Smudge Quality Gates (`Seamless5KCompositor` & `QualityGateAuditor`)
- Expand to `5040x2160` (`21:9`) (or face-centered crop for `9:16` / `1:1`), verify all 7 automated quality gates including **Per-Organ High-Frequency Edge Energy (`measure_per_organ_sharpness` on left eye, right eye, nose, and mouth)** and **Anatomical Landmark Geometry Validation (`validate_anatomical_geometry`)**, and visually inspect both the 5K output and 1:1 close-up face verification crop via `read_file`.
