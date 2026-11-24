# Project Rule: `릴프로필` (`프로필` / 하위호환 `모델생성` — `정본 v1.0.0+`)

## 0. 프로젝트 공식 명칭 · 별칭 라우팅 · 버전 관리 헌법 (`2026-11-24` 확정)
- **정식 명칭**: **`릴프로필`** (약칭: **`프로필`**, 영문: `ReelProfile` / `Profile`, 구명칭 호환: `모델생성`)
- **호출 규칙**: 사용자가 **`릴프로필`**, **`프로필`**, **`모델생성`** 중 어떤 이름으로 호출하더라도 항상 본 단일 정본 프로젝트를 로드한다.
  - 로컬 경로 별칭: `/Volumes/DOMA/Project/릴프로필` · `/Volumes/DOMA/Project/프로필` · `/Users/doma/Projects/릴프로필` · `/Users/doma/Projects/프로필` (정본 디렉터리 `/Volumes/DOMA/Project/모델생성` 연결)
- **정본 `v1.0.0` 디폴트 고정 & 히스토리/업그레이드 규칙**:
  1. 현재 완성된 아키텍처(무하드코딩 원패스 3D 통인물 + 순수 원본 1-Hop 8방향 마스터 뱅크 + 장소·상황 적응형 5렌즈×5독립조명 + 소형/보정 원본 디에어브러싱 4K 포어 복원 + 무그레인 광학 `Lanczos-4`)를 디폴트 **`정본 v1.0.0`**으로 고정한다 (`VERSION.json`).
  2. `정본 v1.0.0` 이전의 모든 개발 이력은 **`v0.1.0 ~ v0.9.5` (`< v1.0.0`) 히스토리**로 `VERSION.json` 및 `docs/VERSION_HISTORY.md`에 보관한다.
  3. **`정본 v1.0.0` 이후 코드·기능 수정 개발 시 의무 버전체크**:
     - 수정 전/후 항상 `python3 scripts/version_ctl.py check`로 코드 트리 해시(`code_tree_sha256`) 무결성을 확인한다.
     - 코드 변경이 발생하면 반드시 `python3 scripts/version_ctl.py bump --part <patch|minor|major> --summary "..."`로 `v1.0.0` 초과 상위 버전(`1.0.1`, `1.1.0`, `2.0.0` 등)으로 업그레이드한다 (`Git pre-commit` 훅이 미업그레이드 커밋을 자동 차단).

---

## 1. 2-Tier Hybrid Cloud & Portable Standalone Architecture
- **Tier 1 (Code, Tests, Rules, Skills, Version Ledger — GitHub & iCloud `< 50KB`)**:
  - GitHub Repository: `https://github.com/domautube-beep/model-generation-engine`
  - Contains modular Python source (`model_generation_engine/`), CLI (`scripts/`), hooks (`hooks/`), version ledger (`VERSION.json`, `docs/VERSION_HISTORY.md`), and tests (`tests/`).
- **Tier 2 (Heavy ONNX/Upscayl Models, Reference Media, 4K Deliverables — Google Drive or Local `<repo>/models/`)**:
  - Auto-resolves Google Drive Vault (`Antigravity_Assets/릴프로필` · `프로필` · `모델생성`) OR local `<repo>/models/` directory so standalone packaged builds run 100% offline on any computer without path hardcoding.

---

## 2. Default `정본 v1.0.4+` Non-Skippable 6-Stage Workflow
1. **Step 1 (`8방향 마스터 뱅크`)**: 새 인물 원본(`Gen-0`)은 반드시 순수 `Gen-0` 단독 1-Hop으로 `-90°, -67.5°, -45°, -22.5°, +22.5°, +45°, +67.5°, +90°` 8방향을 생성·조립(`*_master_8angle_turnaround_sheet.jpg`)한 뒤 진행한다.
2. **Stage 1~2 (`prepare --design-mode zero_touch_4k --framing-mode auto`)**:
   - **기본 `50:50` 인물·배경 여백 밸런스 (`balanced_50_50`)**: 인물로 화면을 가득 채우지 않고 인물 `~45~50%` : 공간 여백 `~50~55%`를 기본으로 잡으며, 사용자가 클로즈업을 명시 요청할 때만 `closeup`으로 자동 전환한다.
   - **기본 `f/1.8~f/2.0` 광학 아웃포커싱 깊이감**: 배경 디테일을 과잉 묘사하지 않고 얕은 심도 보케로 날려 3단 `Z-Depth` 공간 깊이감을 살린다.
   - `SceneAdaptive5Lens5LightingPlanner`가 장소·조도를 분석해 **5렌즈(`50·65·85·105·135mm`) × 5현장맞춤 독립조명**을 생성한다.
3. **Stage 3 (`1-Pass Native Full-Person 3D Generation`)**: 이전 생성물 재투입 `0%`, 얼굴 과장·미화 `0%`, 외부 유료 API `0%`.
4. **Stage 4~6 (`rank` -> `finalize --upscale-mode optical_fullframe --cohesion-mode off` -> `read_file` 400% 육안 검수)**:
   - **`Zero-Touch 4K` 직통 기본**: 4K 변환 시 얼굴 피부색 역주입·하이라이트 압축·국소 샤프닝·부위별 합성을 `0.0%`로 차단하여 처음 생성된 조명·질감·생김새를 100% 보존한다.
