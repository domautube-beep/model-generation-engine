---
name: model-generation-engine
description: Official alias skill for '릴프로필' ('프로필' / legacy '모델생성' — 정본 v1.0.0+). Activate immediately whenever the user mentions '릴프로필', '프로필', '모델생성', '모델생성 스튜디오', or asks to generate, edit, or upscale real-person profile/editorial portraits. Loads `/Volumes/DOMA/Project/릴프로필` (`/Volumes/DOMA/Project/모델생성`), enforces `scripts/version_ctl.py check` & post-v1.0.0 version upgrades on any code change, Mandatory 1-Hop 8-Direction (-90°..+90°) Bank, Zero-Hardcoding 5-Lens (50/65/85/105/135mm) x 5-Scene-Matched Lighting Matrix, De-Airbrushed Raw Pore Restoration, and 4K Zero-Grain Optical Lanczos-4 Mastering.
---

# `릴프로필` (`프로필` / 구 `모델생성` — 정본 `v1.0.0+` Unified Engine)

> **프로젝트 명칭 및 버전 기준 (`2026-11-24` 확정)**:
> - 이 프로젝트의 공식 이름은 **`릴프로필`** 또는 **`프로필`**이며, 어떤 이름(`릴프로필` · `프로필` · `모델생성`)으로 호출하더라도 본 단일 정본 프로젝트(`/Volumes/DOMA/Project/릴프로필` -> `/Volumes/DOMA/Project/모델생성`, GitHub `domautube-beep/model-generation-engine`)를 로드한다.
> - 현재 완성 상태를 디폴트 **`정본 v1.0.0`**으로 고정하며, 이전 개발 이력은 `v0.1.0 ~ v0.9.5` (`< v1.0.0`) 히스토리로 관리한다 (`VERSION.json`, `docs/VERSION_HISTORY.md`).
> - **코드·기능이 수정개발되어 바뀌면 반드시 `python3 scripts/version_ctl.py bump --part <patch|minor|major> --summary "..."`로 버전체크 및 `v1.0.1+` 업그레이드를 수행한다.**

## 1. 하이브리드 클라우드 및 별칭 경로
- **Active Workspace**: `/Volumes/DOMA/Project/릴프로필` · `/Volumes/DOMA/Project/프로필` · `/Volumes/DOMA/Project/모델생성` (`~/Projects/릴프로필`, `~/Projects/프로필`)
- **Tier 1 (iCloud `< 50KB`)**: `~/Library/Mobile Documents/com~apple~CloudDocs/Antigravity_Control_Tower/skills/reel-profile/SKILL.md` (및 `model-generation-engine/SKILL.md`)
- **Tier 2 (Google Drive Vault / Local `models/` Fallback)**:
  `~/Library/CloudStorage/GoogleDrive-*/내 드라이브/Antigravity_Assets/릴프로필` (`모델생성` 심볼릭 연동, 타 PC 오프라인 실행 시 `<repo>/models/` 자동 폴백)
- **로컬 웹 스튜디오 (`http://127.0.0.1:18830`)**: `"/Volumes/DOMA/Project/릴프로필/릴프로필 스튜디오 실행.command"`

## 2. 정본 `v1.0.0` 디폴트 생성 파이프라인
1. **Step 1 (`8방향 마스터 뱅크` 필수 선행)**: 순수 `Gen-0` 단독 1-Hop으로 `-90°..+90°` 8방향 생성 및 `*_master_8angle_turnaround_sheet.jpg` 조립 (`원본1생성게이트.py` 강제).
2. **Stage 1~2 (`prepare --best-of 5 --no-film-grain`)**:
   - `SceneAdaptive5Lens5LightingPlanner`가 현장 조도·상황을 자동 판별해 **`50·65·85·105·135mm` 5렌즈 × 5현장맞춤 독립조명** + 초점거리별 자동 클린플레이트(`loc01..loc05`)를 생성한다.
   - `OrganAndDepthProfile.is_airbrushed_or_lowres_origin` (`skin_norm_laplacian < 24.0`)이 초소형·증명사진 보정뭉개짐을 자동 감지해 실제 피부 모공과 현장 톤을 복원한다.
3. **Stage 3~6 (`1-Pass 3D 생성` -> `rank` -> `finalize --upscale-mode optical_lanczos --no-film-grain` -> `read_file` 400% 육안 검수)**:
   - 외부 유료 API `0%`, 이전 생성물 재투입 `0%`, 얼굴 과장 `0%` 원칙으로 4K 마스터 및 매크로·눈매·윤곽 시트를 검수 후 전달한다.
