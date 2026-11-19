# Red Team Architecture & Zero-Hardcoding Audit Report (`모델생성` v2.0)

- **Project**: `모델생성` (`model-generation-engine` v2.0.0)
- **Repository Path**: `/Volumes/DOMA/Project/모델생성` (symlinked at `/Users/doma/Projects/모델생성`)
- **Audit Scope**: Full line-by-line static & dynamic Red Team verification across all 11 engine modules (`config.py`, `detector.py`, `skin_processor.py`, `prompt_builder.py`, `workflow.py`, `compositor.py`, `upscaler.py`, `quality_gate.py`, `pipeline.py`, `cli.py`, `__init__.py`) and adversarial test suite (`tests/test_model_generation_engine.py`).

---

## 1. Executive Summary

All 11 modules of `model_generation_engine` were audited against the 5 Universality Axes required for **`모델생성`**:
1. **Axis 1 — Universal Model (`SubjectProfile`)**: Zero hardcoded gender pronouns (`"man"`, `"his"`), zero hardcoded `"Korean male"`, zero hardcoded `"browline glasses"`, and zero hardcoded `175cm / 80kg`.
2. **Axis 2 — Universal Wardrobe, Location, and Camera Angle (`WardrobeSpec`, `LocationSpec`, `CameraRigSpec`, `UniversalVariationPlanner`, `cli.py`)**: Full combinatorial support for both multi-domain presets and arbitrary custom outfits, accessories, locations, props, lighting, camera angles, shot scales, poses, and lens focal lengths via Python API and CLI.
3. **Axis 3 — Adaptive Skin Tone & 3D Head Pose (`DynamicFaceDetector`, `AdaptiveSkinToneProfile`, `MatteSkinProcessor`)**: Zero hardcoded CIE LAB skin thresholds (`L > 125, A > 131, B > 133`); dynamic inner-cheek/nose-bridge sampling across all human skin tones (deep ebony, olive/tan, fair/pale) and 3D head yaw adaptation.
4. **Axis 4 — Universal Canvas & Aspect Ratio (`CanvasSpec`, `Seamless5KCompositor`)**: Automatic face-position-aware horizontal padding for ultrawide (`21:9` 5K, `16:9` 4K) AND face-centered horizontal cropping for narrow/vertical (`9:16`, `1:1`) aspect ratios so off-center subjects are never clipped.
5. **Axis 5 — Reference Isolation & Single-Pass Lineage (`SinglePassGenerationGuard`)**: Dual-layer lineage enforcement combining filename pattern checks and neural `DynamicFaceDetector` inspection to reject any auxiliary reference image containing a human face.

---

## 2. Red Team Findings & Fixes Applied

| ID | Module | Finding / Edge Case Discovered | Fix Applied | Verification Test |
| :--- | :--- | :--- | :--- | :--- |
| **RT-01** | `detector.py` (`AdaptiveSkinToneProfile.build_chrominance_gate`) | `l_low = max(35.0, self.l_median - l_tol)` imposed a floor of `L >= 35.0` in OpenCV CIE LAB (`0..255`), which could clip shadow-side cheek skin pixels on very deep ebony skin tones (`L ~ 16..42`). | Lowered the floor to `l_low = max(12.0, self.l_median - l_tol)` so deep ebony skin tones are never clipped while still excluding pure black background/hair pixels. | `test_09_adversarial_skin_tone_spectrum_ebony_olive_pale` |
| **RT-02** | `compositor.py` (`Seamless5KCompositor.build_ultrawide_canvas`) | When rendering a vertical `9:16` (`2160x3840`) or square `1:1` (`2160x2160`) canvas from a wide `16:9` or `21:9` source (`new_w >= spec.target_width`), the compositor previously used a static center crop `x0 = (new_w - spec.target_width) // 2`, which could clip or miss a subject standing on the left (`0.22 * w`) or right (`0.78 * w`). | Added neural face-centered crop window calculation (`x0 = clamp(round(face.center.x - 0.5 * spec.target_width), 0, new_w - spec.target_width)`) when `new_w >= spec.target_width`. | `test_10_adversarial_universal_canvas_aspect_ratios_and_offcenter_subjects` |
| **RT-03** | `cli.py` (`plan-variations` / `plan`, `build-prompts`) | 1) Passing a single custom override (e.g. `--custom-outfit`) previously replaced the entire location and camera preset lists with a single dummy fallback instead of combining the custom outfit with `--location-preset` and `--camera-preset`.<br>2) Missing CLI flags for `--identity-label`, `--expression`, `--facial-trait`, `--custom-accessories`, `--custom-props`, `--custom-lighting`, `--custom-palette-hex`, `--custom-shot-scale`, `--custom-pose`, `--focal-length`, `--negative-space-side`, and `--aspect-ratio-label`. | Upgraded `cli.py` to support independent per-axis custom overrides (combining custom outfit with preset locations/cameras or full custom specs), added the `plan` command alias, and exposed all 18 subject/wardrobe/location/camera CLI flags. | `test_11_adversarial_cli_custom_model_outfit_location_angle` |
| **RT-04** | `workflow.py` (`SinglePassGenerationGuard.validate_reference_lineage`) | Lineage validation checked image count and filename patterns, but lacked an optional neural face-detection guard when inspecting auxiliary reference paths. | Added `auxiliary_reference_paths` and `face_detector` parameters so `SinglePassGenerationGuard` can run `DynamicFaceDetector.detect()` on any auxiliary image and raise `ReferenceIsolationError("... contains a human face")`. Also fixed a missing space typo in `LOCATION_CATALOG["architectural_interior"]`. | `test_12_static_ast_and_regex_zero_hardcoding_audit` |

---

## 3. Adversarial Test Suite Results (12 / 12 Passed)

Command executed:
```bash
PYTHONPATH="/Volumes/DOMA/Project/모델생성" uv run --no-project --with opencv-python-headless --with onnxruntime python3 -m unittest discover -s "/Volumes/DOMA/Project/모델생성/tests" -v
```

| Test Case | Axis Audited | Result |
| :--- | :--- | :--- |
| `test_01_universal_subject_profiles_zero_hardcoding` | Axis 1 (`SubjectProfile`, neutral defaults, female 168cm/50kg no-glasses, male 175cm/80kg glasses) | **PASS** |
| `test_02_universal_variation_planner_across_4_axes` | Axis 2 (`UniversalVariationPlanner` presets + custom `WardrobeSpec`/`LocationSpec`/`CameraRigSpec`) | **PASS** |
| `test_03_prompt_firewall_blocks_forbidden_patterns` | Axis 1 & 3 (`PromptFirewall` blocks face-bloating, airbrush blur, plastic gloss, forced wrinkles) | **PASS** |
| `test_04_reference_isolation_and_single_pass_guard` | Axis 5 (`SinglePassGenerationGuard` blocks `depth > 0`, generated outputs, and mood refs) | **PASS** |
| `test_05_adaptive_skin_tone_profile_and_matte_skin_processor` | Axis 3 (`AdaptiveSkinToneProfile` & `MatteSkinProcessor` on real and darkened skin tones) | **PASS** |
| `test_06_auto_balanced_5k_compositor_dynamic_padding` | Axis 4 (`Seamless5KCompositor` auto-detects subject position when `pad_left_ratio=None`) | **PASS** |
| `test_07_similarity_alignment_validation` | Super-Resolution (`SuperResolutionEngine.validate_similarity_transform` blocks skew/shear) | **PASS** |
| `test_08_all_11_deliverables_pass_6_quality_gates` | End-to-End 6 Quality Gates across all 11 5K (`5040x2160`) deliverables in Google Drive Vault | **PASS** |
| `test_09_adversarial_skin_tone_spectrum_ebony_olive_pale` | **Red Team Axis 3**: Deep ebony (`L ~ 18..42`), olive/tan, and pale Nordic skin tones | **PASS** |
| `test_10_adversarial_universal_canvas_aspect_ratios_and_offcenter_subjects` | **Red Team Axis 4**: `21:9` 5K, `16:9` 4K, `9:16` vertical, `1:1` square with left/center/right subjects | **PASS** |
| `test_11_adversarial_cli_custom_model_outfit_location_angle` | **Red Team Axis 2**: CLI full custom flags + partial custom outfit with preset locations/angles | **PASS** |
| `test_12_static_ast_and_regex_zero_hardcoding_audit` | **Red Team Axes 1 & 5**: Static regex scan across all 11 `.py` files + neural auxiliary face rejection | **PASS** |
