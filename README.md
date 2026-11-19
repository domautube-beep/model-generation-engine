# 모델생성 (`model-generation-engine` v3.0 & `모델생성 스튜디오`)

> **100% Universal, Zero-Hardcoding Real-Person 8-Angle Identity, Multi-Styling, Multi-Location & 3D Previz Studio Engine**  
> Built for **all projects** without a single hardcoded gender, eyewear, height/weight, outfit, location, pixel coordinate, or LAB skin threshold.

---

## 1. Quick Launch — Standalone Web Studio (`모델생성 스튜디오 v3.0`)

- **macOS One-Click**: Double-click `모델생성 스튜디오 실행.command`
- **Terminal Launch (`http://127.0.0.1:18830`)**:
  ```bash
  PYTHONPATH="." /Users/doma/.local/bin/uv run --no-project \
    --with fastapi --with uvicorn --with python-multipart --with httpx \
    --with opencv-python-headless --with onnxruntime --with pillow \
    python3 launch_studio.py
  ```

### What You Can Do in `모델생성 스튜디오 v3.1`
1. **STEP 01 — Real-Person Frontal-Arc Detailed 8-Direction Registration (`완전좌 -90° ~ 완전우 +90°` 사이 `22.5°` 간격 8방향 + 중앙 `0°` 원본)**:
   - Upload a real person's face photo -> extracts an untouched pure optical `85mm` canonical anchor (`0°` `Gen-0`), measures 3D craniofacial ratios (`Apple Vision` + `YuNet`), and generates/manages **8 detailed frontal-hemisphere directions** (`-90°, -67.5°, -45°, -22.5°` | `[0° 원본]` | `+22.5°, +45°, +67.5°, +90°`) via 4-Hop `22.5°` progressive bridging (`<= 11.25°` max angular deviation across any portrait angle!).
2. **STEP 02 — Clothes, Accessories, Bags & Shoes Multi-Slot Attachments**:
   - Attach multiple photos across `옷 (garment)`, `악세사리 (accessory)`, `가방 (bag)`, and `신발 (shoe)` as needed.
   - Automatic `YuNet` **Foreign-Face Neutral-Grey Masking (`#8C8C8C`)** ensures models inside product/lookbook photos never contaminate the subject's identity.
3. **STEP 03 — Interactive 3D Previz Pose Director**:
   - Drag 3D mannequin joints (`head, neck, shoulders, elbows, wrists, hips, knees, ankles`) directly on the canvas, choose from 8 editorial presets, or adjust `0°..360°` head/torso azimuth, camera elevation, focal length (`35/50/85/105mm`), and frame placement—automatically routing the matching angle from the 8-Angle Turnaround Bank!
4. **STEP 04 — Multi-Location & Reference Style Parallel Batching (`N장 첨부 → N장 동시 생성`)**:
   - Upload **N** location/style reference images (or enter **M** text mood lines) to launch **N + M** parallel scene jobs with 16-Point Scene Tone & Manner Harmony.
5. **STEP 05 — Pure Optical `Lanczos-4` 5K Output & 9 Automated Quality Gates**:
   - `Real-ESRGAN`, synthetic skin noise, highlight chrominance shifts, and 2D oval `facelock` pixel pasting are **disabled by default** to guarantee crisp glasses rims, true iris color, and natural matte skin at 400% zoom.
   - Every 5K (`5040x2160`) / 4K (`3840x2160`) output is audited by 9 Quality Gates (`YuNet`, `SFace >= 0.65`, `Apple Vision 3D Proportion Lock`, `Per-Organ Anti-Smudge`, `Seam L1 <= 6.0`, `Zero Specular Glare`).

---

## 2. CLI Subcommands

```bash
# 1. Plan 3-Hop 8-Angle Turnaround from any real face photo
PYTHONPATH="." /Users/doma/.local/bin/uv run --no-project --with opencv-python-headless --with onnxruntime --with pillow \
  python3 -m model_generation_engine.cli plan-turnaround \
  --ref-face /path/to/person.jpg \
  --output-canonical /path/to/canonical_front.jpg \
  --angles 8

# 2. Compose N parallel jobs from 8-angle anchor + clothes/accessories/bag/shoes + multiple locations + 3D Previz pose
PYTHONPATH="." /Users/doma/.local/bin/uv run --no-project --with opencv-python-headless --with onnxruntime --with pillow \
  python3 -m model_generation_engine.cli compose-batch \
  --canonical-front /path/to/canonical_front.jpg \
  --garment /path/to/jacket.jpg \
  --accessory /path/to/watch.jpg \
  --bag /path/to/bag.jpg \
  --shoe /path/to/shoes.jpg \
  --location-image /path/to/place1.jpg \
  --location-image /path/to/place2.jpg \
  --pose-preset seated_armchair --head-azimuth 45 --torso-azimuth 30
```
