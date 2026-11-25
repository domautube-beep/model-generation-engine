---
name: model-generation-engine
description: Official alias skill for '릴프로필' ('프로필' / legacy '모델생성' — 정본 v1.0.5+). Activate immediately whenever the user mentions '릴프로필', '프로필', '모델생성', '모델생성 스튜디오', or asks to generate, edit, or upscale real-person profile/editorial portraits. Loads `/Volumes/DOMA/Project/릴프로필` (`/Volumes/DOMA/Project/모델생성`), enforces `scripts/version_ctl.py check` & post-v1.0.0 version upgrades on any code change, Mandatory 1-Hop 8-Direction (-90°..+90°) Bank, Default 50:50 Subject-to-Space Framing (close-up ONLY on request), Default f/1.8 Wide-Aperture Bokeh Z-Depth, and Zero-Touch 4K Passthrough Mastering (`0.0px` lighting/texture/likeness mutation).
---

# `릴프로필` (`프로필` / 구 `모델생성` — 정본 `v1.0.6+` Unified Alias Skill)

> **프로젝트 명칭 및 버전 헌법 (`2026-11-24` 확정 · `2026-11-25` v1.0.6 보안·초해상 업그레이드)**:
> - 공식 프로젝트명은 **`릴프로필`** (약칭 **`프로필`**, 하위호환 `모델생성`)이며, 어떤 이름으로 호출하더라도 단일 정본 저장소(`__REELPROFILE_REPO_DIR__` / `/Volumes/DOMA/Project/릴프로필` / `~/Projects/릴프로필`, GitHub `domautube-beep/model-generation-engine`)와 정본 스킬 `reel-profile`을 로드한다.
> - `v1.0.0` 이전 개발 이력은 `v0.1.0 ~ v0.9.5` (`< v1.0.0`) 히스토리로 관리하며, 코드·스크립트 수정 시 반드시 `python3 <repo>/scripts/version_ctl.py bump --part patch --summary "..."`로 상위 버전(`>= v1.0.1`, 현재 `v1.0.6+`) 업그레이드 및 버전체크(`check`)를 통과해야 한다.

## 1. 2-Tier 하이브리드 클라우드 경로 해석
- **Tier 1 (iCloud `< 50KB`)**: `~/Library/Mobile Documents/com~apple~CloudDocs/Antigravity_Control_Tower/skills/reel-profile/` 및 `model-generation-engine/`
- **Tier 2 (Google Drive Heavy Asset Vault)**:
  `glob.glob(os.path.expanduser("~/Library/CloudStorage/GoogleDrive-*/내 드라이브/Antigravity_Assets/모델생성"))` (`resources/subject_03..07`, `outputs/`, `releases/릴프로필_정본_v1.0.0_전체패키지.zip` 보관; 오프라인 실행 시 `<repo>/models/` 자동 폴백).

## 2. 정본 `v1.0.6+` 핵심 디폴트 4원칙
1. **Step 1 필수 선행 & 매 수정 시 Stage 1 순수 원본 재시작**: 순수 `Gen-0` 단독 1-Hop 8방향(`-90°..+90°`) 마스터 시트 구축 후 진행하며, 어떤 수정·피드백 요청이라도 기존 생성물을 다시 넣거나 기존 `image-generator`에 `send_message`로 덧그리는 **다중 재생성(Img2Img)을 `100%` 금지**하고 Stage 1 `prepare`부터 순수 원본으로 새로 생성한다 (`Default-Deny Whitelist` + `16x16` 시각 지문 물리 강제).
2. **기본 구도 & 심도 (`생-035`)**: **인물 `~45~50%` : 주변 배경 여백 `~50~55%` (`balanced_50_50`)** + **`f/1.8~f/2.0` 개방 아웃포커싱 3단 공간 깊이감(`Z-Depth`)** 기본 적용 (배경 과잉 디테일 묘사 금지, 클로즈업은 명시적 요청 시에만 전환).
3. **Tone-Locked True 4K 초해상 (`v1.0.6+` 기본)**: 네이티브 4K는 `0.0px` 무변형 직통 보존하고, `1K`/`2K` 원시 생성물은 `upscale_mode="tonelocked_upscayl_4k"`로 `Upscayl 4x` 초해상 + `CIE LAB` 저주파 조도·색온도 `100%` 고정(`ΔL ≈ 0, Δa* = 0, Δb* = 0`)을 적용해 **조명·색감·인상 변형 `0%` + 진짜 4K 선명도(`LapVar >= 17.5`)**를 동시에 달성한다.
4. **우회 0% 물리 잠금**: `--allow-audit-fail`, `strict_10gate=False`, `1K` 단순 `Lanczos` 가짜 4K, 영수증/상태 위변조 및 미통과 종료를 코드와 `원본1생성게이트.py` 양쪽에서 영구 봉쇄한다.
