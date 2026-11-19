---
name: model-generation-engine
description: Use this skill immediately whenever the user talks about creating, generating, editing, or upscaling model/portrait/editorial hero images ('모델생성' 프로젝트, /Volumes/DOMA/Project/모델생성 및 /Users/doma/Projects/모델생성, GitHub domautube-beep/model-generation-engine), launching '모델생성 스튜디오' (http://127.0.0.1:18830), or asks to generate a real person/model with 8 detailed frontal-arc turnaround directions (-90°..+90°), custom clothes, accessories, bags, shoes, multiple locations/style references, or 3D Previz poses. Enforces 100% Universal Zero-Hardcoding v3.4: Pure Optical Canonical Front (0°) + 4-Hop Detailed 8-Direction (-90°..+90°) Frontal-Arc Bank, Ironclad 1원본들→1생성 Hook & Guard, 3-Axis Pose + Natural Gaze Auto-Extraction, Pre-Gen Single-Lens Chiaroscuro Relighting + 3-Plane Shallow DOF (f/1.8) + Post-Gen Optical Cohesion Mastering (PostGenCohesionMasterer), Pure Optical Lanczos-4 4K/5K, and 9 Automated Quality Gates.
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
2. **Ironclad `1원본들 → 1생성` (One-Originals-Set to One-Generation — Enforced by Global Hook `원본1생성게이트.py` & `OriginOnlyOneShotGuard`)**:
   - **HARD BAN ON RE-FEEDING (`Depth >= 2` PROHIBITED)**: Never feed a generated scene output (`live_test_`, `hero_`, `5k_21x9_`, `raw_batch_`, etc.) back into any image generator (`v1 -> v2` refinement is strictly blocked by `OriginOnlyOneShotGuard` SHA-256 blacklist AND `~/.gemini/config/hooks.json` `PreToolUse` hook).
   - **8-Direction Face Anchors ALWAYS Attached**: Every scene generation call MUST attach `[Image 1: Exact Angle-Matched 8-Direction Face, Image 2: Untouched 0° Original Face, Image 3: Complete -90°..+90° 8-Direction Master Turnaround Sheet]` alongside the **raw user-uploaded original assets** (`원본 옷`, `원본 악세사리`, `원본 가방`, `원본 신발`, `원본 장소`).
   - **ZERO Synthetic Boards / Wireframes in `--image`**: Never pass `composite_styling_board.jpg` or synthetic `previz_*.jpg` wireframes into `--image`; 3D Previz is strictly for UI pose/angle control (auto face direction by default, manual override when changed).
   - Use direct native `4K` (`3840x2160`) multi-image `gti` (`StudioImageRuntime`) rather than 3-image-capped subagent tools whenever attaching >3 originals.
3. **Pure Optical `Lanczos-4` Super-Resolution (`Real-ESRGAN` Disabled by Default)**:
   - `EngineConfig.upscale_mode = "optical_lanczos"` is the mandatory default. `Real-ESRGAN (4xNomos8kSC)` corrodes `2px` glasses rims and drops `Tenengrad` sharpness by `63~88%`.
4. **Zero Synthetic Skin Noise & Zero Highlight Chrominance Shift (`MatteSkinProcessor`)**:
   - `micro_texture_amplitude = 0.0`, `organ_crisp_strength = 0.0`, and `percentile_knee = 99.2` luminance-only roll-off (zero `A/B` color shift inside highlights, preventing philtrum orange blotches and white sandpaper speckles).
5. **Hard Ban on 2D Oval Pixel Pasting (`facelock` Disabled)**:
   - `auto_facelock_below_iod_5k_px = 0.0`. Never paste 2D oval face crops onto rotated bodies (which severs eyewear temple arms and breaks scene lighting). Use multi-reference scene generation or `run_tonematched_5k_synthesis` (`16:9` upper-body 3D redraw).

---

## 3. End-to-End 5-Module Studio & Engine Workflow (`v3.0`)

### Step 1: Real-Person Frontal-Arc Detailed 8-Direction Registration (`identity_turnaround.py` — `완전좌 -90° ~ 완전우 +90° 디테일 8방향`)
- `prepare_canonical_identity_anchor`: Normalizes any real face/portrait photo into a pure optical `85mm` `4:5` (`1200x1500`) head-and-shoulders center origin (`0°` `Gen-0`) with zero GAN pre-smoothing.
- `extract_anatomical_blueprint`: Measures `Apple Vision` + `YuNet` 3D Euclidean craniofacial ratios (`eye_to_chin`, `width_to_length`, `iod_to_bbox_width`, `mouth_to_eye_width`).
- `plan_progressive_8angle_turnaround` & `assemble_master_turnaround_sheet`: Builds **8 detailed frontal-hemisphere directions** at exact **`22.5°` micro-intervals** between **Complete Left (`-90°`)** and **Complete Right (`+90°`)** around the untouched **`0°` Original** (`<= 11.25°` max angular error across the entire frontal arc, ZERO wasted rear-head slots):
  - **Center Origin (`0°`)**: `front` (`기준 원본 0°`) — Untouched `Gen-0` optical source
  - **Hop 1 (`±22.5°` 세미프론트)**: `left_22` (`④ 좌 미세 -22.5°`) & `right_22` (`⑤ 우 미세 +22.5°`) from `[0°]`
  - **Hop 2 (`±45.0°` 표준 ¾)**: `left_45` (`③ 좌 표준¾ -45°`) & `right_45` (`⑥ 우 표준¾ +45°`) bridged by `[0° + ±22.5°]`
  - **Hop 3 (`±67.5°` 깊은 반측면)**: `left_67` (`② 좌 깊은¾ -67.5°`) & `right_67` (`⑦ 우 깊은¾ +67.5°`) bridged by `[0° + ±45.0°]`
  - **Hop 4 (`±90.0°` 완전 측면)**: `left_90` (`① 완전 좌 -90°`) & `right_90` (`⑧ 완전 우 +90°`) bridged by `[0° + ±67.5°]`

### Step 2: Hierarchical Garment Layering, Anthropometric Scale Lock & Bag Kinematics (`multi_asset_composer.py` v3.2)
- **Hierarchical Garment Z-Order (`GARMENT_SLOTS` + `WEARING_STYLES`)**:
  - `inner_top` (속 티셔츠/셔츠) -> `mid_layer` (니트/베스트) -> `outerwear` (자켓/코트) + `bottom` (하의) / `one_piece`.
  - When `inner_top` or `mid_layer` is paired with `outerwear` (`wearing_style="auto"`), the engine automatically sets `outerwear` to `open_layered` so inner tops are never hidden behind buttoned jackets unless explicitly set to `closed_buttoned`.
- **Anthropometric Scale & Proportion Lock (`ACCESSORY_SLOTS` + `_build_proportional_styling_board`)**:
  - Prevents standalone e-commerce macro photos from inflating watches or mini-bags:
    - `watch_left` / `watch_right`: Locked to `38–42mm` (~68% of human wrist width) + exposes `~1.5cm` past the sleeve cuff.
    - `eyewear`: `1:1` cranial temple span + continuous temple arms over ears.
    - `jewelry_fine`: `1.0–2.5cm` earlobe/finger scale; `belt`: `3.0–3.5cm` loop width.
  - Automatically builds a **Proportional Composite Styling Board (`composite_styling_board.jpg`)** where accessories/mini-bags occupy smaller cell ratios than coats/trousers.
- **Bag Size Classes (`BAG_SIZES`) & 11 Carry/Placement Kinematics (`BAG_CARRY_MODES`)**:
  - Size classes: `micro_mini (~20cm)`, `small_compact (~26cm)`, `medium_standard (~36cm)`, `large_tote_weekender (~48cm)`, plus optional exact `size_cm`.
  - Carry/Position modes: `hand_right`, `hand_left`, `crook_right`, `crook_left`, `shoulder_right`, `shoulder_left`, `crossbody_l2r`, `crossbody_r2l`, `underarm_clutch`, `backpack`, and **`placed_beside` (not held—grounded upright beside chair/feet)**—synchronized directly onto the 3D Previz Mannequin!
- **Foreign-Face Grey Masking (`mask_foreign_faces_in_reference`)**: Every uploaded item or location photo is scanned with `YuNet` and any foreign human face is covered with `#8C8C8C` neutral grey.

### Step 3: 3-Axis Pose (`Yaw + Pitch + Roll`), Natural Gaze & Interactive 3D Previz (`previz_pose.py`)
- `analyze_reference_pose_before_masking`: Automatically extracts 3-axis head orientation (`auto_yaw_deg`, `auto_pitch_deg` downward/upward bow, `auto_roll_deg` candid head tilt) and `auto_gaze_mode` (`downward_task_focus`, `off_camera_natural`, `camera_eye_contact`, `side_window_contemplative`) from any reference photo BEFORE privacy masking.
- Never pre-bake dozens of gaze angles beyond the 8 horizontal frontal-arc anchors, and NEVER generate a face separately to stitch/composite onto a generated body afterward (`사후 얼굴 합성 금지`). All 3-axis neck/chin/eyelid kinematics + scene relighting happen in ONE atomic `1원본들 → 1생성` pass.

### Step 4: Camera Body × Film Stock × Lens DOF & Anti-AI-Yellow/Crunch Optical Engine (`scene_radiometry.py` v3.5)
- **Explicit 3-Axis Photographic Controls (`카메라 바디 × 필름 컬러 사이언스 × 렌즈 심도`)**:
  - `CAMERA_PROFILES`: `hasselblad_x2d` (HNCS 16-bit neutral color), `leica_sl3` (APO/Summilux clarity), `sony_a7rv` (5500K commercial precision), `contax_645` (medium-format planar roll-off), `canon_r5` (translucent skin tone).
  - `FILM_PROFILES`: `digital_neutral_raw` (default, 5500K Capture One neutral, zero grain, zero yellow cast), `kodak_portra_400`, `fujifilm_pro_400h`, `cinestill_50d`, `kodak_gold_200`, `ilford_hp5_bw`.
  - `APERTURE_MODES`: `f1_4` (ultra-shallow planar bokeh), `f1_8`, `f2_0` (editorial standard 3-plane depth), `f2_8` (outfit-balanced DOF), `f5_6` (architectural depth).
- **Anti-AI-Yellow & Anti-AI-Crunch Laws**:
  - **Pre-Gen**: Enforces grey-card calibrated `5400K–5600K` daylight white balance (banning sepia/amber triggers), anchors pixel smoothness to `Image 2` (`0°` real camera portrait), and enforces **Distance-Appropriate Lens Resolution** (preventing 30cm macro product pebble-leather/suede grit from stamping repetitive patterns onto 2.5–3.5m shots).
  - **Post-Gen (`PostGenCohesionMasterer`)**: Quantitatively neutralizes GPT-Image's excess `CIE b*` yellow/orange cast across highlights/midtones/shadows, applies subtle Optical MTF De-Crunch outside the face oval, and supports `<0.3s` zero-regeneration Film Stock switching in Studio UI.

### Step 5: Pure Optical 4K/5K Composition & 9 Automated Quality Gates (`quality_gate.py`)
- Outputs `3840x2160` (`16:9` default, pure Lanczos-4 cover-scale with zero side-strip smear), `5040x2160` (`21:9`), `2160x3840` (`9:16`), `2400x3200` (`3:4`), or `2048x2048` (`1:1`), verified against all 9 Quality Gates (`Resolution`, `YuNet Face Detection`, `Global Sharpness`, `Seam L1 <= 6.0`, `Natural Skin Texture`, `Zero Specular Glare <= 0.04`, `Per-Organ Anti-Smudge`, `Min Face Scale`, and `SFace >= 0.65 + Apple Vision 3D Proportion Lock`).
