# `v2.1` Core Architecture Rebase & `v2.2` Next-Gen Development Roadmap (`2026-11-23`)

## 1. Executive Summary & Architectural Decision (`생-019 원패스 통인물 3D 생성` 채택)

Through controlled A/B benchmarking on the Classical Palazzo Charcoal-Suit Lounge Chair scene (`media_1795407280330_6bfa4cd0.jpg` + `Subject 03` `Gen-0`), we compared two fundamentally different architectures:

| Metric / Dimension | **`v2.1` Core (`1-Pass Native 3D + Pure Optical Lanczos-4`)** | `v15.1` (`Headless 1:1 Mesh-Lock on Foreign Body`) |
| :--- | :--- | :--- |
| **Whole-Body Anthropometric Unity** | **PASS (Superior)** — Head, neck thickness/length, shoulder span, suit collar drape, hands, and torso are synthesized together as **one single `7.45-head` Korean person** (`reattach_mode="none"`). | **Mismatch on Foreign Body** — Preserves the reference model's wide Western dropped shoulders, bony neck, and hands underneath a Korean face. |
| **Upscaler Fidelity (`Lanczos-4` vs `Upscayl 4x`)** | **Pure Optical `Lanczos-4` (`2679×4800`)**: `SFace = 0.6741` (`PASS`), `scleraDiffPct = -14.3%` (`Ocular PASS`). (`Upscayl 4x` degraded `SFace -> 0.6369` and darkened sclera to `-21.8%`.) | Native 4K registration (`2700×4800`). |
| **Facial Identity (`SFace` / `contourIoU`)** | `SFace = 0.6741`, `contourIoU = 0.9099` (slight 2D yaw difference `+8°` vs `+16.1°`, `107px` raw pre-upscale face width). | `SFace = 0.9174`, `contourIoU = 0.9887` (`Gen-0` pixel-locked). |

**Decision (`2026-11-23`)**: Re-anchor the primary engine onto **`v2.1` (`1-Pass Native Full-Person 3D Synthesis + Pure Optical Lanczos-4 + P88 MatteSkin + PostGenCohesionMasterer`)** and evolve `v2.1` directly into **`v2.2`** by solving `v2.1`'s remaining 4 precision bottlenecks—without ever pasting a Korean face onto a foreign Western mannequin body.

---

## 2. `v2.1 -> v2.2` Next-Generation Engineering Roadmap (4 Core Pillars)

### Pillar 1: Razor-Sharp Clean-Plate Person Erasure (`Zero Fog-Smudge Location Plate`)
- **Current `v2.1` Bottleneck**: `cv2.inpaint(..., INPAINT_TELEA)` leaves a large blurry grey fog blob where the foreign person sat, softening chair edges and background frescoes.
- **`v2.2` Solution**:
  - Segment ONLY the human silhouette (`Apple Vision VNGeneratePersonSegmentationRequest`) while protecting furniture/props (`chair/table edge mask`), and reconstruct background texture cleanly (patch-based exemplar synthesis / clean plate generation) so `Image 3 (loc_clean)` retains 100% crisp furniture geometry and background bokeh.

### Pillar 2: Native High-Density 2K/4K 1-Pass Generation (`Raw Face 107px -> 320px+`)
- **Current `v2.1` Bottleneck**: Standard 1K full-body generation (`768×1376`) yields only `~107×136px` (`IOD ~51px`) on the raw face before optical scaling, capping `400%` iris/eyelash micro-acuity and `SFace` around `~0.67–0.70`.
- **`v2.2` Solution**:
  - Drive **Native 2K/4K 1-Pass Full-Person Generation** (`gemini-3-pro-image` 2K/4K / Google Flow Ultra `Nano Banana Pro` `4K` `3584×4800` with clean-plate spatial guidance that stays within the 2-face-ref identity budget) so raw pre-upscale face width starts at **`>= 300px` (`IOD >= 140px`)** natively in 3D!

### Pillar 3: Pre-Mask 3-Axis Pose Extraction & Exact 8-Dir Anchor Pairing (`contourIoU 0.91 -> 0.96+`)
- **Current `v2.1` Bottleneck**: When target head yaw drifts even `7–8°` from `Gen-0` (`+8°` vs `+16.1°`), 2D cheek/jaw projection shifts (`contourIoU = 0.910`).
- **`v2.2` Solution**:
  - Run `analyze_reference_pose_before_masking` on the target scene photo **before** erasing the foreign model to measure exact `(yaw, pitch, roll)` + seated/standing `SizeKorea 8th` head-to-body ratio (`7.35–7.55 heads`).
  - Automatically pair `Image 1 = Untouched Gen-0` with `Image 2 = Nearest Verified 1-Hop 8-Dir Portrait` (`Δyaw <= 11.25°`) AND evaluate `contourIoU` against the **yaw-matched 8-dir reference** as well as `Gen-0` during `Best-of-K` (`K=4`) selection.

### Pillar 4: Zero-GAN Optical Micro-Acuity & Ocular Catchlight Mastering (`Pure Optical v2.2`)
- **Current `v2.1` Strength + Upgrade**: Keep `upscale_mode="optical_lanczos"` as the ironclad default (zero GAN `Upscayl` eye-darkening), and enhance `upscale_bgr_in_memory` + `MatteSkinProcessor` (`P88`) + `PostGenCohesionMasterer` with luminance-only Wiener/Richardson-Lucy micro-deconvolution restricted to ocular lash/iris & lip vermilion edges (`0px` geometric warp, `0%` face pasting).

---

## 3. Verified `v2.2` Live Benchmark Results (`2026-11-23` Palazzo Charcoal-Suit Scene)

Implemented across `identity_turnaround.py`, `upscaler.py`, and `fresh_origin_workflow.py` (`workflow_version = "2.2"`), executed 100% from scratch (`Stage 1 prepare -> Stage 3 Best-of-3 1-Pass -> Stage 4 rank -> Stage 5 optical_lanczos4_acuity_v22`):

| Metric (`0px` Warp · `0%` Pasting · `0%` Re-Feeding) | `v2.1` Baseline | **`v2.2` Unified Engine** |
| :--- | :---: | :---: |
| **`contourIoU` (1:1 Pupil-Aligned Silhouette)** | `0.9099` | **`0.9571`** |
| **`meanContourPointErrPx`** | `13.72 px` | **`6.30 px`** |
| **`cheek` / `temple` Width Diff** | `-2.2% / -3.6%` | **`-0.2% / -1.2%`** |
| **`eyelidEdgeNcc` (Ocular Fold Match)** | `0.2755` | **`0.4506`** |
| **`ocularDriftScore` / `scleraDiffPct`** | `10.82 / -14.3%` | **`6.11 / -7.5%` (PASS)** |
| **`exaggerationIndex`** | `3.07` | **`2.73` (PASS)** |

