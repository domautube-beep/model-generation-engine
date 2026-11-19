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

## 3. Mandatory 8-Stage Progression Workflow (Including Real-Person 3D Turnaround Registration)

### Stage 0: Real-Person 3D Turnaround Registration (`RealPersonTurnaroundEngine` — Solves Real-Photo vs AI-Model Angle Gap)
- **Why Real Photos Drifted Across Angles While AI Models Stayed Consistent**:
  1. AI models (`제니`, `이소`) start from a tight `85mm` neutral studio head-and-shoulders anchor, carry an explicit verbal DNA (`identityPrompt`), and pre-build a **5-Angle Turnaround (`정면 0° · 좌¾ 45° · 우¾ 45° · 좌측면 90° · 우측면 90°`)** using 2-Hop Progressive Rotation (`0° -> ±45° -> ±90°` with `[0° + ±45°]` dual anchors), then attach the angle-matched face anchor to every scene!
  2. Raw user photos often have small face area (`~5%` of pixels surrounded by furniture/suits/legs), zero `identityPrompt` (`photo-v1` empty lead/look), and zero side/3Q depth ground-truth—forcing the generator to hallucinate generic average faces when turning `>20°`.
- **Mandatory Real-Photo Registration Steps (`model_generation_engine/identity_turnaround.py`)**:
  1. `prepare_canonical_identity_anchor`: Normalize the uploaded photo into a crisp `4:5` (`1200x1500`) `85mm` head-and-shoulders anchor (preserving real `7.5~8등신` shoulder context while maximizing facial ViT patch tokens).
  2. `extract_anatomical_blueprint`: Measure `Apple Vision` + `YuNet` craniofacial ratios (`eye_to_chin`, `width_to_length`, `iod_to_bbox_width`, `mouth_to_eye_width`) and inject `to_3d_invariance_clause()`.
  3. `plan_progressive_5angle_turnaround` & `assemble_master_turnaround_sheet`: Generate **Hop 1 (`좌¾ 45°`, `우¾ 45°`)** from `0°`, then **Hop 2 (`좌측면 90°`, `우측면 90°`)** passing **BOTH `[0° Front + ±45° Same-Side 3Q]`**, and assemble the 5-angle Master Turnaround Sheet.
  4. `select_angle_matched_references`: Whenever generating or refining an angled scene, automatically attach `[Canonical Front + Angle-Matched Turnaround View]`!

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

### Stage 5: Pure Optical Lanczos-4 Super-Resolution (`upscale_mode="optical_lanczos"`, `Real-ESRGAN` Disabled by Default)
- **HARD BAN ON `Real-ESRGAN` (`4xNomos8kSC`) FOR GLASSES & SKIN**:
  - `Real-ESRGAN` corrodes `2px` tortoiseshell/metal glasses rims into moth-eaten artifacts and smears fine hair/skin grain (`Tenengrad` drops `-63% ~ -88%`), matching `AI모델스튜디오` Rule 26 (`Real-ESRGAN 금지`).
  - Default `EngineConfig.upscale_mode = "optical_lanczos"` performs pure `INTER_LANCZOS4` scaling + gentle clamped `[-7, +7]` luminance micro-clarity with zero GAN hallucination.

### Stage 6: Zero-Blotch / Zero-Synthetic-Noise Optical Skin Preservation (`MatteSkinProcessor`)
- **HARD BAN ON SYNTHETIC NOISE & LAB HIGHLIGHT CHROMINANCE INJECTION**:
  - `micro_texture_amplitude = 0.0` (synthetic Gaussian noise creates sandpaper speckles on 5K faces) and `organ_crisp_strength = 0.0` (prevents white ringing halos on glasses rims).
  - `percentile_knee = 99.2` rolls off ONLY extreme clipped white glare (`> P99.2`) in luminance `L` ONLY—never shifting `A/B` chrominance inside highlights (which previously caused orange/peach blotches beside the philtrum).

### Stage 7: Tone-Matched 3D In-Place Upper-Body Synthesis (`run_tonematched_5k_synthesis`), 5K (`5040x2160`) 21:9 Outpainting & 9 Quality Gates
- **HARD BAN ON 2D OVAL PIXEL PASTING (`facelock` PROHIBITED)**:
  - Never paste flat 2D reference face pixels through an oval mask (`cx ± 0.45*fw`). 2D pixel pasting cuts off eyewear temple arms (`안경 다리`) in mid-air, skews 2D facial axes against 3D rotated heads, crams mismatched scales into the skull, and forces white-studio LAB skin tones that clash with the room's lighting and mood (`합성 취소` rule).
- **Mandatory Tone-Matched 3D In-Place `16:9` Upper-Body Redraw (`ModelGenerationPipeline.run_tonematched_5k_synthesis`)**:
  1. Extract the dynamic `16:9` (`1280x720`) upper-body crop via `extract_dynamic_upper_body_crop`.
  2. Refine in-place with `image-generator` passing **Image 1 = full upper-body identity reference portrait** (preserving real head-to-shoulder ratio) + **Image 2 = `16:9` upper-body crop** using `PromptFirewall.build_tonematched_3d_refinement_prompt`:
     - **True 3D Identity & Balanced 7.5–8-Head Proportions**: Render the exact reference identity naturally rotated in 3D matching Image 2's head angle and balanced head-to-shoulder proportions.
     - **Continuous Eyewear Temple Integrity (`안경 다리 필수`)**: Whenever eyewear is worn, render both temple arms continuously and uninterruptedly from the outer front frame hinges back over both ears.
     - **Scene Tone & Manner Harmony (`전체 톤앤매너·색감·조명 일치`)**: Relight face and matte skin naturally to match the target room's directional key light, shadow falloff, color temperature, and mood.
  3. Upscale both full-body scene and refined crop 4x (`4xNomos8kSC`), integrate seamlessly via `Seamless5KCompositor.integrate_tonematched_crop` (skipping CodeFormer and `facelock` so 3D anatomy, glasses temples, and scene tone & manner remain 100% intact), and apply gentle `P88` specular gloss roll-off (`reference_skin_profile=None`).
- **9 Automated Quality Gates (`QualityGateAuditor`)**:
  1. Resolution (`5040x2160`), 2. Face Detection, 3. Global Laplacian Sharpness, 4. Outpaint & Crop Seam Continuity (`seam_l1_diff <= 6.0`), 5. Natural Skin Micro-Texture, 6. Zero Plastic Specular Glare (`<= 0.04`), 7. Per-Organ Anti-Smudge Edge Energy (`left_eye, right_eye, nose, mouth`) & Landmark Geometry, 8. Minimum Face Scale (`min_inter_ocular_5k_px`), 9. **`SFace` 128-D Neural Identity Cosine Similarity (`>= 0.65`) & macOS Apple Vision 3D-Foreshortening-Aware Facial Proportion Lock (`compare_face_shapes <= 5.5%` on `eyeChin` and `widthLength`)**.
