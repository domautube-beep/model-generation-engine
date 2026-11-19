---
name: model-generation-engine
description: Use this skill immediately whenever the user talks about creating, generating, editing, or upscaling model/portrait/editorial hero images ('모델생성' 프로젝트, /Volumes/DOMA/Project/모델생성 및 /Users/doma/Projects/모델생성, GitHub domautube-beep/model-generation-engine), launching '모델생성 스튜디오' (http://127.0.0.1:18830), or asks to generate a real person/model with 8 turnaround angles, custom clothes, accessories, bags, shoes, multiple locations/style references, or 3D Previz poses. Enforces 100% Universal Zero-Hardcoding v3.0: Pure Optical Canonical Front + 3-Hop 8-Angle (0°..315°) Turnaround Bank, Foreign-Face Grey Masking on Outfit/Location References, Interactive 3D Previz Pose Rig, Multi-Location N-to-N Parallel Batching, 16-Point Scene Tone & Manner Harmony, Pure Optical Lanczos-4 5K (Real-ESRGAN & 2D facelock disabled), and 9 Automated Quality Gates.
---

# Universal Model Generation Engine & Studio (`모델생성` — `model-generation-engine` v3.0)

## 1. Hybrid Cloud Paths, Studio Launcher & Environment
- **Active Workspace**: `/Volumes/DOMA/Project/모델생성` (Symlink: `/Users/doma/Projects/모델생성`)
- **GitHub Repository**: `https://github.com/domautube-beep/model-generation-engine`
- **Standalone Studio Web App (`모델생성 스튜디오 v3.0`)**:
  - Local URL: `http://127.0.0.1:18830` (isolated from port `18820`)
  - Double-click macOS Launcher: `/Volumes/DOMA/Project/모델생성/모델생성 스튜디오 실행.command`
  - CLI / Daemon Launch:
    ```bash
    PYTHONPATH="/Volumes/DOMA/Project/모델생성" /Users/doma/.local/bin/uv run --no-project --with fastapi --with uvicorn --with python-multipart --with httpx --with opencv-python-headless --with onnxruntime --with pillow python3 /Volumes/DOMA/Project/모델생성/launch_studio.py --no-browser
    ```
- **Google Drive Heavy Asset Vault**: `/Users/doma/Library/CloudStorage/GoogleDrive-domautube@gmail.com/내 드라이브/Antigravity_Assets/모델생성`
  - `models/`: `yunet.onnx`, `sface.onnx`, `codeformer.onnx`, `realesrgan-ncnn-vulkan` (legacy opt-in only)
  - `resources/`: Untouched real reference portraits (`ref_person_face.jpg`, `ref_teacher_02.jpg`, etc.)
  - `outputs/`: 8-Angle Master Turnaround Sheets, 5K `5040x2160` / 4K `3840x2160` hero deliverables, and macro verification crops

---

## 2. 100% Universal Zero-Hardcoding Rules (Mandatory Across ALL Projects)
1. **Zero Hardcoded Attributes**: Never hardcode gender (`Korean male`, `his`), eyewear (`browline glasses`), height/weight (`175cm, 80kg`), fixed outfits, fixed locations, pixel coordinates `(x1, y1, x2, y2)`, or fixed CIE LAB skin thresholds (`L>125, A>131, B>133`) anywhere in code or prompts. Static regex audit `test_12` enforces this across `model_generation_engine/**/*.py`.
2. **Untouched `Gen-0` Primary Authority (Zero Re-Feeding Degradation)**:
   - Always pass the **untouched real photograph (`Gen-0` pure optical canonical front)** as **Image 1** in every turnaround hop and every scene generation job.
   - Never pass an AI-generated image (`Gen-1`) alone without `Gen-0` Image 1—otherwise faint AR lens coatings amplify into green irises and shadow noise turns into purple eyelid bruises on small wide-shot faces.
3. **Pure Optical `Lanczos-4` Super-Resolution (`Real-ESRGAN` Disabled by Default)**:
   - `EngineConfig.upscale_mode = "optical_lanczos"` is the mandatory default. `Real-ESRGAN (4xNomos8kSC)` corrodes `2px` glasses rims and drops `Tenengrad` sharpness by `63~88%`.
4. **Zero Synthetic Skin Noise & Zero Highlight Chrominance Shift (`MatteSkinProcessor`)**:
   - `micro_texture_amplitude = 0.0`, `organ_crisp_strength = 0.0`, and `percentile_knee = 99.2` luminance-only roll-off (zero `A/B` color shift inside highlights, preventing philtrum orange blotches and white sandpaper speckles).
5. **Hard Ban on 2D Oval Pixel Pasting (`facelock` Disabled)**:
   - `auto_facelock_below_iod_5k_px = 0.0`. Never paste 2D oval face crops onto rotated bodies (which severs eyewear temple arms and breaks scene lighting). Use multi-reference scene generation or `run_tonematched_5k_synthesis` (`16:9` upper-body 3D redraw).

---

## 3. End-to-End 5-Module Studio & Engine Workflow (`v3.0`)

### Step 1: Real-Person 8-Angle 360° Turnaround Registration (`identity_turnaround.py`)
- `prepare_canonical_identity_anchor`: Normalizes any real face/portrait photo into a pure optical `85mm` `4:5` (`1200x1500`) head-and-shoulders anchor with zero GAN pre-smoothing.
- `extract_anatomical_blueprint`: Measures `Apple Vision` + `YuNet` 3D Euclidean craniofacial ratios (`eye_to_chin`, `width_to_length`, `iod_to_bbox_width`, `mouth_to_eye_width`).
- `plan_progressive_8angle_turnaround` & `assemble_master_turnaround_sheet`: Builds all **8 canonical compass angles** at `45°` intervals:
  - `0°` `front` (① 정면) — Untouched `Gen-0` optical source
  - `45°` `three_quarter_left` (② 좌전방 ¾) & `315°` `three_quarter_right` (⑧ 우전방 ¾) — **Hop 1** from `[0°]`
  - `90°` `left_profile` (③ 좌측면) & `270°` `right_profile` (⑦ 우측면) — **Hop 2** bridged by `[0° + ±45°]`
  - `135°` `back_left_3q` (④ 좌후방 ¾), `180°` `back` (⑤ 완전 후면), `225°` `back_right_3q` (⑥ 우후방 ¾) — **Hop 3** bridged by `[0° + ±90°]`

### Step 2: Multi-Category Styling Attachments & Foreign-Face Grey Masking (`multi_asset_composer.py`)
- Supports multiple optional image/text attachments across 4 dedicated slots:
  1. **Clothes (`garment`)**: Outerwear, tops, bottoms, suits, dresses
  2. **Accessories (`accessory`)**: Eyewear, watches, ties, scarves, hats, jewelry, belts
  3. **Bags (`bag`)**: Briefcases, totes, shoulder bags, backpacks, clutches
  4. **Shoes (`shoe`)**: Dress shoes, loafers, sneakers, boots, heels
- `mask_foreign_faces_in_reference`: Every uploaded garment, accessory, bag, shoe, and location/style photo is scanned with `YuNet` and any foreign human face is covered with a neutral `#8C8C8C` grey box so foreign models never contaminate the subject's identity.

### Step 3: Interactive 3D Previz Pose Director (`previz_pose.py`)
- `PosePrevizSpec` & `Previz3DRenderer.render_pose_guide`: Renders a 14-bone depth-sorted 3D articulated mannequin with perspective floor grid, camera focal-length projection (`35/50/85/105mm`), camera elevation (`low / eye / high`), horizontal composition placement, custom joint overrides, and a head gaze vector.
- `resolve_8angle_view_from_azimuth(head_azimuth_deg)`: Automatically snaps any `0°..360°` Previz head angle to the closest of the 8 turnaround views and attaches `[Image 1: Gen-0 Front, Image 2: Matched 8-Angle View]`.

### Step 4: Multi-Location & Reference Style N-to-N Parallel Batching (`MultiSceneBatchComposer`)
- Uploading **N** location/style reference photos (and/or **M** text mood lines) automatically expands into **N + M** parallel scene jobs (`여러 장 넣으면 생성도 여러 장`).
- Supports both `scene_and_mood` (place subject inside the uploaded architectural space + match lighting/tone) and `style_mood_only` (extract 16-point brightness, key-light direction, color temperature, shadow falloff, and editorial mood into a fresh space).
- Every composed prompt is validated through `PromptFirewall.validate_prompt` (Body-vs-Face Firewall + Natural Matte Skin Firewall + Continuous Eyewear Temple Integrity + Authentic Iris Color Lock).

### Step 5: Pure Optical 5K Composition & 9 Automated Quality Gates (`auditor.py`)
- Outputs `5040x2160` (`21:9`), `3840x2160` (`16:9`), `2160x3840` (`9:16`), `2400x3200` (`3:4`), or `2048x2048` (`1:1`) via `Seamless5KCompositor` + pure optical `Lanczos-4` + 9 Quality Gates (`Resolution`, `YuNet Face Detection`, `Global Sharpness`, `Seam L1 <= 6.0`, `Natural Skin Texture`, `Zero Specular Glare <= 0.04`, `Per-Organ Anti-Smudge`, `Min Face Scale`, and `SFace >= 0.65 + Apple Vision 3D Proportion Lock`).
