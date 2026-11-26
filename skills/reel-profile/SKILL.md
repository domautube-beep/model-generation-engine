---
name: reel-profile
description: Use this skill immediately whenever the user mentions '릴프로필', '프로필', '릴 프로필', '모델생성', '릴프로필 스튜디오', or asks to generate, edit, or upscale real-person profile / editorial / portrait photos across any location, pose, outfit, or lens. Loads the canonical '릴프로필 (프로필)' project (정본 v1.0.0+ baseline, currently v1.2.0+), enforces mandatory automated version checking (`scripts/version_ctl.py check` / `bump` above v1.0.0 whenever modified, with < v1.0 archived as history), Mandatory Step 1 1-Hop 8-Direction (-90°..+90°) Bank, Native-4K Google Flow ULTRA Headed `x4` Pre-Ranked Generation (1K JPEG fallback permanently banned), Hard-Locked <=50% Subject Occupancy (`person_mask <= 38.8%`, close-up ONLY on explicit request), Default f/1.8–f/2.0 Wide-Aperture Optical Bokeh Z-Depth, Zero-Touch Native 4K Mastering (`0.0px` mutation) + 13-Gate Quality & Anti-Mottle Hardlock, and 3-Command High-Speed Batch CLI (`prepare-batch --emit-flow-batch --emit-finalize-batch` -> `google_flow_ultra_cli.py --batch-json` -> `finalize-batch`).
---

# `릴프로필` (`프로필` — 정본 `v1.2.0+` High-Speed Native-4K & 13-Gate Hardlock Skill)

## 1. 프로젝트 공식 명칭 · 별칭 라우팅 · 2-Tier 하이브리드 클라우드 + 로컬 NVMe 캐시
- **정식 명칭**: **`릴프로필`** (약칭: **`프로필`**, 하위호환: `모델생성`, `reel-profile`, `profile`)
- **호출 철칙**: 사용자가 **`릴프로필`**, **`프로필`**, **`모델생성`** 중 무엇으로 부르든 단일 정본 저장소(GitHub: `domautube-beep/model-generation-engine`)를 로드한다:
  1. `/Users/doma/.gemini/antigravity/scratch/릴프로필` (로컬 고속 작업 정본)
  2. `/Volumes/DOMA/Project/릴프로필` (`~/Projects/릴프로필` · `~/Projects/프로필` · `~/Projects/모델생성`)
- **Tier 1 (iCloud Command & Skill Tower `< 50KB`)**:
  - `~/Library/Mobile Documents/com~apple~CloudDocs/Antigravity_Control_Tower/skills/reel-profile/SKILL.md` (이미지/영상/모델 저장 절대 금지)
- **Tier 2 (Google Drive Heavy Asset Vault + 로컬 NVMe SSD 자동 미러 캐시)**:
  - 원본 리소스·최종 4K 산출물 영구 보관: `~/Library/CloudStorage/GoogleDrive-*/내 드라이브/Antigravity_Assets/{릴프로필,프로필,모델생성}`
  - **로컬 NVMe SSD 초고속 실행 캐시**:
    - `~/.cache/model_generation_engine/models/` (`yunet.onnx`, `sface.onnx`, `upscayl-bin`, `ultramix-balanced-4x.{param,bin}` 자동 스테이징)
    - `~/.cache/model_generation_engine/vision_cache/` (`AppleVisionAnalyzer` 랜드마크·세그멘테이션·관절 2계층 영구 캐시)
    - `~/.cache/model_generation_engine/blueprint_cache/` (`Gen-0` SHA-256 기준 25항목 두개안면 블루프린트 영구 캐시)

---

## 2. 의무 버전체크 및 업그레이드 헌법 (`정본 v1.0.0` 기준 · 현재 `v1.2.0+`)
1. `v1.0.0` 이전 개발본은 `VERSION.json` 및 `docs/VERSION_HISTORY.md`에 `v0.1.0 ~ v0.9.5` 히스토리로 분리 보관한다.
2. 작업 시작 및 코드 수정 직후 반드시 검증·업그레이드한다 (`Git pre-commit` 훅 물리 강제):
   ```bash
   python3 <repo>/scripts/version_ctl.py check
   python3 <repo>/scripts/version_ctl.py bump --part patch --summary "수정/개발 내역 요약"
   ```

---

## 3. 정본 `v1.2.0+` 초고속 · 무우회 · 네이티브 4K 3-커맨드 표준 프로세스 (13대 하드 게이트)

1. **Step 1 — 순수 원본(`Gen-0`) 1-Hop 8방향(`-90° ~ +90°`) 마스터 뱅크 필수 선행**:
   - 새 인물은 반드시 순수 `Gen-0` 단독 입력(`hop == 1`)으로 8방향을 먼저 생성·조립(`*_master_8angle_turnaround_sheet.jpg`)한다 (`원본1생성게이트.py` 물리 강제).
2. **Command 1 (Stage 1~2) — 단일 프로세스 고속 배치 매니페스트 & Flow/Finalize 배치 JSON 동시 생성 (`< 0.15s`)**:
   - 수정 요청 시에도 매 턴 반드시 Stage 1부터 순수 `Gen-0`으로 재시작한다 (`1원본 → 1생성`).
   ```bash
   python3 <repo>/scripts/run_fresh_origin_workflow.py prepare-batch \
     --spec-json <prep_batch.json> \
     --emit-flow-batch <flow_batch.json> \
     --emit-finalize-batch <fin_batch.json>
   ```
   - **원패스 합격 물리 고정**: `person_mask <= 34%`(`balanced_50_50` `<= 50%` 시각 비중), 원본 동공·흰자위 비율(`34%~42%`), 8.2등신 자연 어깨폭, 턱밑 투명 반사광이 프롬프트에 자동 주입되며, 콜라주 시트 대신 `순수 Gen-0 + 단일 매칭 8방향 + 무손실 눈매 앵커`만 `flow_batch.json`에 바인딩된다.
3. **Command 2 (Stage 3) — Google Flow ULTRA 유창 모드(`headed`) `x4` 원샷 생성 + 브라우저 내 `<0.4s` 13게이트 프리랭킹 + 1등 네이티브 4K 단독 다운로드**:
   ```bash
   /Users/doma/Projects/shorts-automation/.venv/bin/python <repo>/scripts/google_flow_ultra_cli.py \
     --batch-json <flow_batch.json>
   ```
   - **다중 인물 단일 Chrome 세션 처리**: 인물별 고유 프로젝트 URL(`1인 1프로젝트`)을 작업 단위로 전환하고 프롬프트 칩을 인플레이스 초기화(`<0.4s`)한다.
   - **`x4` 1회 제출 + 프리랭킹**: 인물당 1번만 제출해 4장 타일을 동시 생성(스피너 종료 즉시 대기 탈출, 불필요 요청 75% 절감 → `FLOW_UNUSUAL_ACTIVITY_TOAST` 예방)하고, 캔버스 프리뷰를 13게이트 기준(`<0.4s`)으로 즉시 채점해 **합격 1순위 타일만 `3584×4800` 무손실 4K PNG로 다운로드**한다.
   - **절대 금지**: `1K` 생성기(`image-generator`) 도피, `--headless` 실행, `ZERO catchlight` / `porcelain-white sclera` / 몰래 줌인 프롬프트 조작은 훅에서 `100%` 차단된다. 로그인/봇 확인창 발생 시 절대 우회하지 않고 즉시 멈춰 보고한다.
4. **Command 3 (Stage 4~5) — `finalize-batch` (`<0.3s/컷`, `0.0px` 무변형 네이티브 4K 직통 + 13대 하드 게이트 + Vault 동기화)**:
   ```bash
   python3 <repo>/scripts/run_fresh_origin_workflow.py finalize-batch --spec-json <fin_batch.json>
   ```
   - **13대 물리 게이트 전원 통과 필수 (`0.0%` 기준 완화 없음)**:
     - 기본 10대 게이트 (`3584×4800+` 네이티브 4K, `LapVar >= 17.5`, `SFace >= 0.65`, `Contour IoU >= 0.92`, `MaxErr <= 11.5px`, `FaceExaggerationGuard`)
     - **Gate 11 (`50% 이하 인물 면적`)**: `balanced_50_50` 기준 `person_mask_pct <= 38.8%` 초과 시 즉시 탈락
     - **Gate 12 (`눈동자 암부 뭉개짐 & 흰자 점박이 얼룩 차단`)**: `sclera_speckle_p85 <= 10.6` 및 `iris_core_luma_p25 >= 8.5` 미달 시 즉시 탈락
     - **Gate 13 (`맨살 볼/이마 명암-색차 레이어 얼룩 차단`)**: `L_hf / ab_hf <= 15.5` 초과 시 즉시 탈락
5. **Stage 6 — `read_file` 4종 400% 육안 검수 의무**:
   - 보고 전 반드시 `*_4k.jpg`, `*_4k_macro.jpg`, `*_eye_compare_400pct.jpg`, `*_silhouette_overlay.jpg`를 `read_file`로 직접 확인한다.
