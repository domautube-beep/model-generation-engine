---
name: reel-profile
description: Use this skill immediately whenever the user mentions '릴프로필', '프로필', '릴 프로필', '모델생성', '릴프로필 스튜디오', or asks to generate, edit, or upscale real-person profile / editorial / portrait photos across any location, pose, outfit, or lens. Loads the canonical '릴프로필 (프로필)' project (정본 v1.0.0+ baseline, currently v1.0.5+), enforces mandatory automated version checking (`scripts/version_ctl.py check` / `bump` above v1.0.0 whenever modified, with < v1.0 archived as history), Mandatory Step 1 1-Hop 8-Direction (-90°..+90°) Bank, Default 50:50 Subject-to-Space Balance (switching to close-up ONLY on explicit request), Default f/1.8–f/2.0 Wide-Aperture Optical Bokeh Z-Depth, and Zero-Touch 4K Passthrough Mastering (`0.0px` lighting/texture/likeness mutation).
---

# `릴프로필` (`프로필` — 정본 `v1.0.5+` Canonical Skill)

## 1. 프로젝트 공식 명칭 · 별칭 라우팅 · 2-Tier 하이브리드 클라우드 경로 해석
- **정식 명칭**: **`릴프로필`** (약칭: **`프로필`**, 하위호환: `모델생성`, `reel-profile`, `profile`)
- **호출 철칙**: 사용자가 **`릴프로필`**, **`프로필`**, **`모델생성`** 중 무엇으로 부르든 항상 아래 단일 정본 저장소를 불러온다 (GitHub: `domautube-beep/model-generation-engine`):
  1. `__REELPROFILE_REPO_DIR__` (설치 경로)
  2. `/Volumes/DOMA/Project/릴프로필` (또는 `/Volumes/DOMA/Project/모델생성`, `/Volumes/DOMA/Project/프로필`)
  3. `~/Projects/릴프로필` · `~/Projects/프로필` · `~/Projects/모델생성`
- **Tier 1 (iCloud Command & Skill Tower `< 50KB`)**:
  - `~/Library/Mobile Documents/com~apple~CloudDocs/Antigravity_Control_Tower/skills/reel-profile/SKILL.md` (이미지/영상/압축파일 저장 절대 금지)
- **Tier 2 (Google Drive Heavy Asset Vault — 멀티컴퓨터 자동 탐색 & 로컬 `models/` 폴백)**:
  ```python
  import glob, os
  from pathlib import Path

  def find_reelprofile_vault() -> Path | None:
      for name in ("릴프로필", "프로필", "모델생성"):
          for pat in (
              f"~/Library/CloudStorage/GoogleDrive-*/내 드라이브/Antigravity_Assets/{name}",
              f"~/Library/CloudStorage/GoogleDrive-*/My Drive/Antigravity_Assets/{name}",
              f"G:/내 드라이브/Antigravity_Assets/{name}",
              f"G:/My Drive/Antigravity_Assets/{name}",
          ):
              for m in glob.glob(os.path.expanduser(pat)):
                  if os.path.isdir(m) and os.path.isdir(os.path.join(m, "models")):
                      return Path(m)
      return None
  ```
  - 타 PC 오프라인/독립 실행 시 `<repo>/models/` 내장 모델을 자동 사용한다 (`릴프로필_정본_v1.0.0_전체패키지.zip` → `releases/` 미러 보관).
- **로컬 웹 스튜디오 (`http://127.0.0.1:18830`)**:
  - macOS: `"<repo>/릴프로필 스튜디오 실행.command"` / Windows: `릴프로필_스튜디오_실행_Windows.bat`

---

## 2. 의무 버전체크 및 업그레이드 헌법 (`정본 v1.0.0` 기준)
1. **디폴트 정본 기준**: `v1.0.0` (`정본 v1.0.0`) 이전 개발본은 `VERSION.json` 및 `docs/VERSION_HISTORY.md`에 `v0.1.0 ~ v0.9.5` 히스토리로 분리 보관한다.
2. **수정·개발 시 자동 버전체크 필수 (`>= v1.0.1`, 현재 `v1.0.5+`)**:
   - 작업 시작 및 코드 수정 직후 반드시 실행:
     ```bash
     python3 <repo>/scripts/version_ctl.py check
     ```
   - 코드·프롬프트 엔진·스크립트가 수정개발되어 바뀌면 **반드시 상위 버전으로 업그레이드**하고 원장에 기록한다 (`Git pre-commit` 훅이 미업그레이드 커밋을 자동 차단):
     ```bash
     python3 <repo>/scripts/version_ctl.py bump --part patch --summary "수정/개발 내역 요약"
     ```

---

## 3. 정본 `v1.0.5+` 디폴트 생성 파이프라인 (무하드코딩 6단계)
1. **Step 1 — 순수 원본(`Gen-0`) 1-Hop 8방향(`-90° ~ +90°`) 마스터 뱅크 필수 선행**:
   - 새 인물은 반드시 순수 `Gen-0` 단독 입력(`hop == 1`)으로 `-90°, -67.5°, -45°, -22.5°, +22.5°, +45°, +67.5°, +90°` 8방향을 먼저 생성·조립(`*_master_8angle_turnaround_sheet.jpg`)한다 (`원본1생성게이트.py` 물리 강제).
2. **Stage 1~2 — `prepare` (`50:50 인물·배경 여백` 디폴트 · 요청 시에만 `closeup` · `f/1.8~f/2.0` 아웃포커싱 깊이감)**:
   ```bash
   python3 <repo>/scripts/run_fresh_origin_workflow.py prepare \
     --gen0 <Gen0.jpg> --turnaround-sheet <master_8angle_sheet.jpg> \
     --work-dir <stage_dir> --design-mode zero_touch_4k --framing-mode auto \
     --no-film-grain --aspect 3:4 --best-of 5 \
     --scene "..." --outfit "..."
   ```
   - **기본 구도 (`balanced_50_50`, 말사전 `생-035`)**: 프레임을 인물로 꽉 채우지 않고 **인물 약 `45~50%` : 주변 배경 여백·헤드룸 약 `50~55%`**를 기본으로 잡으며, 사용자가 클로즈업을 명시 요청할 때만 `closeup`으로 자동 전환한다.
   - **기본 심도 (`f/1.8 ~ f/2.0` 개방 아웃포커싱 깊이감)**: 배경 디테일을 과잉 묘사하지 않고 눈·얼굴 초점면만 선명하게 맞춘 뒤 전경·후경을 부드러운 광학 보케로 날려 **3단 공간 깊이감(`Z-Depth`)**을 살린다.
3. **Stage 3 — 원패스 통인물 3D 생성**: 1인 1 Google Flow 프로젝트 고정, 이전 생성물 재투입 `0%`, 얼굴 과장·미화 `0%`, 외부 유료 API `0%`.
4. **Stage 4~6 — `rank` -> `finalize --upscale-mode optical_fullframe --cohesion-mode off` (`Zero-Touch 4K`, 말사전 `생-034`) -> `read_file` 400% 육안 검수**:
   - **Zero-Touch 4K 무변형 직통 기본**: 4K 변환 시 증명사진 피부색 역주입(`0%`), 하이라이트 압축(`0%`), 눈·코·입 국소 샤프닝(`0%`), 부위별 분리 GAN 합성(`0%`)을 전부 차단해 **처음 생성된 조명·색감·질감·생김새를 `100.0%` 무변형(`maxDiff = 0.0px`)으로 보존**하며, `*_4k_macro.jpg`, `*_eye_compare_400pct.jpg`, `*_silhouette_overlay.jpg`를 육안 검수한 뒤 보고한다.
