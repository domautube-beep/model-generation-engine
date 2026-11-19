---
description: Execute the 7-Stage Universal Zero-Hardcoding Model Generation Workflow (모델생성 v2.0) for any model, outfit, location, and camera angle
---

# Universal Model Generation Workflow (`모델생성` v2.0)

1. **Stage 1 — Validate Identity Reference & Enforce Reference Isolation**
   - Verify the identity reference portrait (`--reference-face`) passes `SinglePassGenerationGuard.validate_reference_lineage` (`generation_depth == 0`, valid 5-point `YuNet` landmarks).
   - If the user provides a mood/pose reference photo containing another person, extract only its `WardrobeSpec`, `LocationSpec`, and `CameraRigSpec` attributes into text — never pass the second person's image to `image-generator`.

2. **Stage 2 & 3 — Plan Universal Multi-Axis Variation Matrix & Firewall Prompts**
   // turbo
   ```bash
   PYTHONPATH="/Volumes/DOMA/Project/모델생성" /Users/doma/.local/bin/uv run --no-project --with opencv-python-headless --with onnxruntime \
     python3 -m model_generation_engine.cli plan \
     --reference-face "/Users/doma/Library/CloudStorage/GoogleDrive-domautube@gmail.com/내 드라이브/Antigravity_Assets/모델생성/resources/ref_person_face.jpg" \
     --wardrobe-category tailored_editorial \
     --location-category studio_editorial \
     --camera-category seated_editorial \
     --count 6
   ```

3. **Stage 4 — Invoke Single-Pass From-Scratch Image Generation (`generation_depth == 0`)**
   - Pass each validated `single_pass_prompt` and ONLY the original identity reference image (`ref_person_face.jpg`) to `image-generator`.
   - Never pass a previously generated image back into `image-generator`.

4. **Stage 5, 6 & 7 — Run 4x Super-Resolution, Adaptive Halo-Free Matte Skin Micro-Texture, 5K 21:9 Outpainting & 6 Quality Gates**
   // turbo
   ```bash
   PYTHONPATH="/Volumes/DOMA/Project/모델생성" /Users/doma/.local/bin/uv run --no-project --with opencv-python-headless --with onnxruntime \
     python3 -m model_generation_engine.cli run \
     --input "<raw_single_pass_image.jpg>" \
     --output "/Users/doma/Library/CloudStorage/GoogleDrive-domautube@gmail.com/내 드라이브/Antigravity_Assets/모델생성/outputs/<output_5k.jpg>" \
     --verify-crop "/Users/doma/Library/CloudStorage/GoogleDrive-domautube@gmail.com/내 드라이브/Antigravity_Assets/모델생성/outputs/<closeup_verify.jpg>" \
     --reference-face "/Users/doma/Library/CloudStorage/GoogleDrive-domautube@gmail.com/내 드라이브/Antigravity_Assets/모델생성/resources/ref_person_face.jpg"
   ```

5. **Visual Verification**
   - Inspect both `<output_5k.jpg>` and `<closeup_verify.jpg>` with `read_file` to confirm identity accuracy, natural matte skin texture without plastic gloss, and seamless 5K 21:9 composition.
