---
trigger: always_on
description: Mandatory non-skippable 6-stage Fresh-Origin Editorial Workflow (v9.0) for real-person model/portrait generation.
---

# 필수 6단계 순수 원본(`Gen-0`) 재-생성 워크플로우 (`v9.0` — 절대 건너뛰기 금지)

실제 인물 사진(`모델생성` / 인물 화보·포트레이트) 생성·수정 요청을 받으면 **예외 없이 아래 6단계를 순서대로 모두 실행**해야 한다. 어느 한 단계라도 건너뛰거나 구 결과물을 후처리('띡 바꾸기')하는 행위는 금지되며 OS 훅(`원본1생성게이트.py` / `enforce_1origin_1gen_hook.py`)이 차단한다.

1. **Stage 1 (`8방향 마스터 뱅크` 필수 선행 + `prepare` — 절대 건너뛰기 금지, OS 훅 물리 차단)**:
   - **Step 1-A (8방향 개별 1-Hop 생성 및 마스터 시트 조립 — 1인 1프로젝트 고정)**:
     - 새 원본(`Gen-0`) 인물은 반드시 먼저 `google_flow_ultra_cli.py --batch-json ... --tier 2K`로 **8방향(`-90°, -67.5°, -45°, -22.5°, +22.5°, +45°, +67.5°, +90°`)**을 순수 `Gen-0` 단독 1-Hop으로 생성하고 `assemble_master_turnaround_sheet`로 마스터 시트(`*_master_8angle_turnaround_sheet.jpg`)를 완성·검증한다.
     - **1인 1프로젝트 원칙**: 동일 인물(`Gen-0` SHA-256 기준)은 반드시 `~/.model_generation_studio/flow_subject_projects.json`에 저장된 **단일 Google Flow 프로젝트 URL** 안에서 8방향 생성과 화보 생성을 계속 진행한다(프로젝트 난립 금지).
   - **Step 1-B (`prepare --turnaround-sheet` — 25항목 Blueprint + 광학 눈매·턱선 앵커 + 매니페스트)**:
     ```bash
     PYTHONPATH="/Volumes/DOMA/Project/모델생성" /Users/doma/.local/bin/uv run --no-project --with opencv-python-headless --with onnxruntime --with pillow python3 /Volumes/DOMA/Project/모델생성/scripts/run_fresh_origin_workflow.py prepare --gen0 <순수원본> --turnaround-sheet <검증된_8방향_마스터_시트> --work-dir <작업폴더> --scene "<장면>" --outfit "<의상>" --aspect 3:4 --best-of 3
     ```
   - `--turnaround-sheet` 없는 `prepare` 또는 8방향 뱅크 미첨부 장면 생성은 `v11.0` OS 훅(`원본1생성게이트.py`)이 즉시 `deny` 차단한다.
2. **Stage 3 (`Best-of-3` 순수 처음 원본 병렬 생성 — Google Flow ULTRA 4K CLI 우선)**:
   - **외부 유료 API 절대 금지.**
   - **우선 엔진 — Google Flow ULTRA 4K 자동 CLI (`scripts/google_flow_ultra_cli.py`, Google AI Ultra 구독 직결, API 과금 0원)**:
     ```bash
     /Users/doma/Projects/shorts-automation/.venv/bin/python /Volumes/DOMA/Project/모델생성/scripts/google_flow_ultra_cli.py --manifest <work_dir>/workflow_manifest.json --output <out.png> --model "Nano Banana Pro" --aspect 3:4 --count 1 --headed
     ```
     - 사용자 Chrome 프로필(`~/.shorts-factory/browser`)로 자동 실행(업로드 권한 팝업 0회), **`Nano Banana Pro` + `4K 업스케일링됨` (`3584×4800`)** 네이티브 4K 다운로드로 전신/의자 샷의 눈매 열화(`896×1200` + `Upscayl` 뭉개짐)를 근본 차단.
     - 배경은 사용자가 "재생성해도 된다"고 할 때 배경 레퍼런스 사진을 빼고 **얼굴 2장(`Gen-0` + 무손실 광학 눈매 크롭)**만 첨부해 얼굴 토큰 집중도를 극대화한다.
   - 보조 엔진 — Antigravity 내장 `image-generator` 병렬 `K=3` (반신·바스트샷 등).
   - 이전 턴 생성물 재투입, `8dir -> 8dir` 체이닝, 사전 Upscayl 입력 절대 금지.
3. **Stage 4 (`rank` — 25항목 네이티브 채점 & 우승 후보 선정)**:
   - `run_fresh_origin_workflow.py rank --manifest ... --candidates c1 c2 c3` 실행.
   - **워프 `0px`, 얼굴 덮어쓰기 `0%`** 상태에서 `SFace` + 18항목(턱각도·볼면적·하관면적·어깨대비얼굴 28.5%~37.5%·두상) + 7항목(눈매·쌍꺼풀·흰자노출·눈매NCC)을 네이티브 채점해 최고 후보를 선택한다. (3장 모두 미달이면 보정하지 말고 원본으로 다시 생성!)
4. **Stage 5 (`finalize` — 단일 `Upscayl 4x` + 10대 품질 게이트 + 4종 산출물 + Vault 동기화)**:
   - `run_fresh_origin_workflow.py finalize --manifest ... --winner <우승후보> --output-dir <출력폴더> --stem <이름>` 실행.
   - **단 1회 `Upscayl 4x`(`high-fidelity-4x` 사진 / `digital-art-4x` 일러스트)** + 무워프(`calibrate_micro_proportions=False`) + 무합성(`reattach_mode="none"`) + **10대 품질 게이트 통과 필수**.
   - `*_4k.jpg`, `*_4k_closeup.jpg`, `*_4k_macro.jpg`, **`*_eye_compare_400pct.jpg`** 4종과 `*_workflow_receipt.json`을 생성하고 Google Drive Vault `outputs/`에 동기화한다.
5. **Stage 6 (육안 검수 — `read_file` 필수)**:
   - 사용자에게 보고하기 전 반드시 `read_file`로 `*_4k.jpg`, `*_4k_macro.jpg`, **`*_eye_compare_400pct.jpg`**를 직접 눈으로 확인해 눈매·쌍꺼풀·흰자·턱각도·볼면적·비율 이상 유무를 검증한다.
6. **얼굴 과장·미화·캐리커처 절대 금지 (`FaceExaggerationGuard` 7축 전면 차단)**:
   - `PromptFirewall` + `원본1생성게이트.py` 훅 + `Gate 10c`가 **눈 과장(인형눈·과대홍채·두꺼운 겉쌍)**, **입술 과장(필러화·과채도 립스틱)**, **코 과장(성형 깎기·길이 변형)**, **눈썹 과장(진하게 그린 눈썹)**, **턱·볼 과장(V라인 깎기·광대 셰이딩)**, **화보 과다 메이크업**, **두상/대두·과소두 과장**을 전면 차단한다.
7. **사용자 피드백 시 절대 원칙**:
   - 사용자가 "눈이 아쉽다 / 턱이 다르다 / 얼굴이 과장됐다" 등 피드백을 주면 **절대 이전 생성물을 후처리·워프·재업스케일하지 않고 Stage 1부터 순수 처음 원본(`Gen-0`)으로 매번 새로 재생성**한다.
