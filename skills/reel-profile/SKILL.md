---
name: reel-profile
description: Use this skill immediately whenever the user mentions '릴프로필', '프로필', '릴 프로필', '모델생성', '릴프로필 스튜디오', or asks to generate, edit, or upscale real-person profile / editorial / portrait photos across any location, pose, outfit, or lens. Loads the canonical '릴프로필 (프로필)' project (정본 v1.0.0+ Default Baseline), enforces mandatory automated version checking (`scripts/version_ctl.py check` / `bump` above v1.0.0 whenever modified, with < v1.0 archived as history), Mandatory Step 1 1-Hop 8-Direction (-90°..+90°) Bank, Context-Adaptive 5-Lens (50/65/85/105/135mm) x 5-Scene-Matched Lighting Matrix, De-Airbrushed Raw Pore Restoration, and 4K Zero-Grain Optical Lanczos-4 Mastering.
---

# `릴프로필` (`프로필` — 정본 `v1.0.0+` Canonical Skill)

## 1. 프로젝트 공식 명칭 · 별칭 라우팅 · 경로 해석
- **정식 명칭**: **`릴프로필`** (약칭: **`프로필`**, 하위호환: `모델생성`, `reel-profile`, `profile`)
- **호출 철칙**: 사용자가 **`릴프로필`**, **`프로필`**, **`모델생성`** 중 무엇으로 부르든 항상 아래 단일 정본 저장소를 불러온다:
  1. `__REELPROFILE_REPO_DIR__` (설치 경로)
  2. `/Volumes/DOMA/Project/릴프로필` (또는 `/Volumes/DOMA/Project/모델생성`)
  3. `~/Projects/릴프로필` · `~/Projects/프로필` · `~/Projects/모델생성`
- **자산 볼트 자동 탐색 (`resolve_gdrive_vault` & 로컬 `models/` 폴백)**:
  - Google Drive `Antigravity_Assets/릴프로필` (또는 `모델생성`) 우선 탐색 후, 타 PC/독립 패키지 실행 시 `<repo>/models/` 내장 모델을 자동 사용한다.
- **로컬 웹 스튜디오 (`http://127.0.0.1:18830`)**:
  - macOS: `"<repo>/릴프로필 스튜디오 실행.command"` / Windows: `릴프로필_스튜디오_실행_Windows.bat`

---

## 2. 의무 버전체크 및 업그레이드 헌법 (`정본 v1.0.0` 기준)
1. **디폴트 정본 고정**: `v1.0.0` (`정본 v1.0.0`)이 이 프로젝트의 디폴트 기준 상태다.
2. **과거 이력 분리 (`< v1.0.0`)**: `정본 v1.0.0` 이전의 모든 개발본은 `VERSION.json` 및 `docs/VERSION_HISTORY.md`에 `v0.1.0 ~ v0.9.5` 히스토리로 분리 보관한다.
3. **수정·개발 시 자동 버전체크 필수 (`>= v1.0.1`)**:
   - 작업 시작 및 코드 수정 직후 반드시 실행:
     ```bash
     python3 <repo>/scripts/version_ctl.py check
     ```
   - 코드·프롬프트 엔진·스크립트가 수정개발되어 바뀌면 **반드시 `v1.0.0` 이후 상위 버전으로 업그레이드**하고 원장에 기록한다 (`Git pre-commit` 훅이 미업그레이드 커밋을 차단):
     ```bash
     python3 <repo>/scripts/version_ctl.py bump --part patch --summary "수정/개발 내역 요약"
     ```

---

## 3. 정본 `v1.0.0` 디폴트 생성 파이프라인 (무하드코딩 6단계)
1. **Step 1 — 순수 원본(`Gen-0`) 1-Hop 8방향(`-90° ~ +90°`) 마스터 뱅크 필수 선행**:
   - 새 인물은 반드시 순수 `Gen-0` 단독 입력(`hop == 1`)으로 `-90°, -67.5°, -45°, -22.5°, +22.5°, +45°, +67.5°, +90°` 8방향을 먼저 생성·조립(`*_master_8angle_turnaround_sheet.jpg`)한다.
2. **Stage 1~2 — `prepare --best-of 5 --no-film-grain` (장소·상황 적응형 5렌즈×5조명)**:
   ```bash
   python3 <repo>/scripts/run_fresh_origin_workflow.py prepare \
     --gen0 <Gen0.jpg> --turnaround-sheet <master_8angle_sheet.jpg> \
     --location <clean_plate.jpg> --clean-location-override <clean_plate.jpg> \
     --work-dir <stage_dir> --design-mode pure_v21 --no-film-grain --aspect 2:3 --best-of 5 \
     --scene "..." --outfit "..."
   ```
   - `SceneAdaptive5Lens5LightingPlanner`가 공간 조도와 상황을 자동 분류해 **`50 / 65 / 85 / 105 / 135mm` 5렌즈 × 5현장맞춤 독립조명** + 초점거리별 자동 클린플레이트(`loc01..loc05`)를 생성한다.
   - 초소형·증명사진 보정 원본(`skin_norm_laplacian < 24.0`)은 자동으로 에어브러싱/플래시 백탁을 분리하고 실제 모공 질감을 복원한다.
3. **Stage 3 — 원패스 통인물 3D 생성**: 이전 생성물 재투입 `0%`, 얼굴 과장·미화 `0%`, 외부 유료 API `0%`.
4. **Stage 4~6 — `rank` -> `finalize --upscale-mode optical_lanczos --no-film-grain` -> `read_file` 400% 육안 검수**:
   - 최종 4K 마스터, `*_4k_macro.jpg`, `*_eye_compare_400pct.jpg`, `*_silhouette_overlay.jpg`를 육안 확인 후 보고한다.
