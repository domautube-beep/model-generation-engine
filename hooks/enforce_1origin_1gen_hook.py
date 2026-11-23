#!/usr/bin/env python3
"""전역 `매번 프로세스 Stage 1(순수 원본 Gen-0)부터 재-생성 강제 게이트 훅` (`원본1생성게이트.py` v10.0).

닫은 허점까지 포함한 5중 물리 차단:
1. `PreInvocation`: 사용자 입력(`lastUserInput`)이 바뀔 때마다 `turn_started_epoch` 갱신.
   - 인물 생성 세션에서 수정·피드백·재생성 지시(`다시`, `재생성`, `고쳐`, `바꿔`, `과장`, `잡아`, `만들어`, `생성해`, `수정해`, `뽑아`, `1.3` 등)가
     들어오면 `requires_fresh_restart_from_stage1 = True`를 걸어, **프로세스 제일 처음(Stage 1 `prepare`)부터 끝(Stage 6 육안 검수)까지
     완료하지 않으면 멈출 수 없게 강제**한다!
2. `PreToolUse` (`run_command`):
   - 이전 턴에 만든 구 생성물(`mtime < turn_started_epoch - 2.0`)을 후처리·재보정·재업스케일·재채점·재감사
     (`calibrate`, `upscale`, `finalize`, `rank`, `audit`, `compare`, `cv2.remap`)하려 하면 즉시 `deny` 차단!
3. `PreToolUse` (`.*generate_image.*|invoke_subagent|send_message|gti`):
   - 이전 생성물/파생물 재투입 즉시 `deny` 차단 (`1원본들 → 1생성`).
   - `requires_fresh_restart_from_stage1 == True`인데 Stage 1(`run_fresh_origin_workflow.py prepare`)을 먼저 실행하지 않고
     바로 생성기를 부르려 하면 즉시 `deny` 차단! ("프로세스 제일 처음으로 가서 바꿔라")
   - 얼굴 과장/미화/캐리커처 유발 프롬프트 즉시 `deny` 차단.
   - 다단계 턴어라운드 연쇄(`8dir -> 8dir`) · 합성 보드 · 3D 프리비즈 와이어프레임 첨부 즉시 `deny` 차단.
4. `PostToolUse` (`run_command|read_file`):
   - `run_fresh_origin_workflow.py prepare` 완료 -> `stage1_prepare_ran = True`
   - `run_fresh_origin_workflow.py finalize` 완료 -> `finalized_4k_receipt = True`
   - `read_file`로 `*_4k_macro.jpg` / `*_eye_compare_400pct.jpg` / `*_silhouette_overlay.jpg` 육안 확인 -> `macro_visually_inspected = True`
5. `Stop` (`--event stop`):
   - 이번 턴에 `requires_fresh_restart_from_stage1 == True` 또는 `fresh_gen_created == True`인데
     (`stage1_prepare_ran`, `fresh_gen_created`, `finalized_4k_receipt`, `macro_visually_inspected`) 중 하나라도 미완료면
     `{"decision": "continue", "reason": "..."}`로 정지 불가 처리 (최대 3회 강제 보완).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

STATE_PATH = Path.home() / ".model_generation_studio" / "fresh_origin_turn_state.json"
LEDGER_PATH = Path.home() / ".model_generation_studio" / "origin_1gen_ledger.json"

FORBIDDEN_GEN_INPUT_PATTERNS = [
    (re.compile(r"(?:^|[/\\])live_test_", re.IGNORECASE), "생성된 테스트 결과물 재투입 금지"),
    (re.compile(r"(?:^|[/\\])(?:hero|5k_21x9|21x9|closeup|macro_verify)_", re.IGNORECASE), "생성된 5K/클로즈업 결과물 재투입 금지"),
    (re.compile(r"(?:^|[/\\])(?:raw_batch|raw_scene|nb_raw|nb_dop|nb_subj|s03_crimson|exp_m|exp_final)_", re.IGNORECASE), "1차 장면 생성물 재투입 금지"),
    (re.compile(r"(?:^|[/\\])s\d+_v\d+_", re.IGNORECASE), "이전 버전 생성 후보/결과물 재투입 금지 (반드시 순수 처음 원본 Gen-0만 입력할 것)"),
    (re.compile(r"(?:_verified_|_closeup|_macro|_eye_compare_|_silhouette_overlay)\b", re.IGNORECASE), "검증/클로즈업/매크로 파생 결과물 재투입 금지"),
    (re.compile(r"upscayl4k_|_upscayl", re.IGNORECASE), "Upscayl 변형본을 다시 생성기 입력으로 넣는 다단계 변형 금지 (순수 원본만 입력할 것)"),
    (re.compile(r"composite_styling_board", re.IGNORECASE), "합성 스타일링 보드 금지 (개별 원본 에셋을 직접 전달할 것)"),
    (re.compile(r"master_8angle_turnaround_sheet", re.IGNORECASE), "9분할 콜라주 시트를 생성기에 직접 넣으면 다중 얼굴 오인으로 정체성이 깨집니다 — 8방향 뱅크의 개별 단독 포트레이트(예: s03_8dir_06_right_45.png)를 첨부하세요"),
    (re.compile(r"(?:^|[/\\])previz_.*\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE), "3D 프리비즈 이미지 첨부 금지 (포즈 텍스트 + 원본만 전달할 것)"),
    (re.compile(r"[/\\]batches[/\\][^/\\]+[/\\](?:raw_|hero_|closeup_)", re.IGNORECASE), "배치 출력물 재투입 금지"),
]

SAFE_ORIGIN_PATTERNS = [
    re.compile(r"[/\\]\.user_uploaded[/\\]", re.IGNORECASE),
    re.compile(r"[/\\]originals[/\\]", re.IGNORECASE),
    re.compile(r"[/\\]loc\d+.*_clean\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE),
    re.compile(r"[/\\]gen0_optical_ocular_detail_anchor\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE),
    re.compile(r"[/\\]stage2a_mesh_precomposite.*\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE),
]

NON_FACE_CLEAN_PLATE_PATTERN = re.compile(
    r"[/\\](?:loc\d+.*_clean|stage2a_mesh_precomposite.*)\.(?:jpg|jpeg|png|webp)$",
    re.IGNORECASE,
)

HEADLESS_TARGET_PATTERN = re.compile(r"(?:headless_body|body_plate|stage2b_flow_harmonized|stage1_body)", re.IGNORECASE)
TURNAROUND_TARGET_PATTERN = re.compile(r"(?:8dir|turn_|angle_[mp]\d|turnaround)", re.IGNORECASE)
TURNAROUND_GENERATED_INPUT_PATTERN = re.compile(
    r"(?:^|[/\\])(?:t2_turn_|crisp_gen1_turn_|subj\d+_turn_|s\d+_8dir_|angle_[mp]\d|.*master_8angle)", re.IGNORECASE
)
VERIFIED_8DIR_BANK_REF_PATTERN = re.compile(
    r"(?:master_8angle|8dir_bank|s\d+_8dir_|turnaround_sheet|8angle_turnaround|angle_[mp]\d|t2_turn_|subj\d+_turn_)",
    re.IGNORECASE,
)
FLOW_CLI_CMD_PATTERN = re.compile(r"google_flow_ultra_cli\.py\b", re.IGNORECASE)

# 구 결과물 후처리·재보정·재감사('거기서 띡 바꾸기' 및 '구 결과물로 재채점해서 넘어가지 않기') 전면 차단 패턴
STALE_TOUCH_CMD_PATTERN = re.compile(
    r"(?:calibrate_to_gen0|MicroProportionCalibrator|proportion_calibrator|upscale_with_upscayl|"
    r"run_single_pass_5k|run_tonematched_5k_synthesis|integrate_relit_face|apply_facelock|"
    r"blend_redrawn_head|cv2\.remap|finalize_native_winner_4k|score_and_select_native_winner|"
    r"run_fresh_origin_workflow\.py\s+(?:rank|finalize)|FaceExaggerationGuard|OcularAnatomyAnalyzer|"
    r"auditor\.audit|guard\.audit)",
    re.IGNORECASE,
)

STAGE1_PREPARE_CMD_PATTERN = re.compile(
    r"(?:run_fresh_origin_workflow\.py\s+prepare|prepare_origin_bundle|run_headless_mesh_workflow\.py)",
    re.IGNORECASE,
)

FINALIZE_SUCCESS_CMD_PATTERN = re.compile(
    r"(?:run_fresh_origin_workflow\.py\s+finalize|finalize_native_winner_4k|run_headless_mesh_workflow\.py\s+finalize)",
    re.IGNORECASE,
)

MACRO_INSPECT_FILE_PATTERN = re.compile(
    r"(?:_4k_macro|_eye_compare_400pct|_silhouette_overlay)\.(?:jpg|jpeg|png|webp)$",
    re.IGNORECASE,
)

IMAGE_PATH_REGEX = re.compile(
    r"([A-Za-z0-9_./가-힣\-\\]+?\.(?:jpg|jpeg|png|webp))",
    re.IGNORECASE,
)

# 수정·피드백·재생성 지시 감지 (단순 질문만 있는 턴은 오탐 방지)
REGENERATION_ACTION_PATTERN = re.compile(
    r"(?:다시\s*(?:해|생성|만들|뽑|돌려)|재생성|처음부터|제일\s*처음|과장.*잡아|싹다\s*잡아|"
    r"고쳐\s*(?:줘|라|서)|바꿔\s*(?:줘|라|서)|수정해|새로\s*(?:생성|만들|뽑)|원본으로\s*(?:무조건\s*)?재생성)",
    re.IGNORECASE,
)

FORBIDDEN_FACIAL_EXAGGERATION_PATTERNS = [
    (re.compile(r"(?<!no\s)(?<!zero\s)(?<!without\s)(?<!never\s)(?:plump(?:ed)?|pouty|luscious|voluminous|juicy|bee[-\s]?stung|glossy\s+red)\s+(?:(?:\w+\s+){0,1})?lips?\b", re.IGNORECASE), "입술 과장/필러화 표현 금지"),
    (re.compile(r"(?<!no\s)(?<!zero\s)(?<!without\s)(?<!never\s)(?:big|large|huge|doe|doll|anime|cat[-\s]?eye|foxy|siren)\s+(?:(?:\w+\s+){0,1})?eyes?\b", re.IGNORECASE), "눈 과장/인형눈 표현 금지"),
    (re.compile(r"(?<!no\s)(?<!zero\s)(?<!without\s)(?<!never\s)(?:sculpted|chiseled|razor|hollow(?:ed)?|contoured)\s+(?:(?:\w+\s+){0,1})?(?:cheekbones?|cheeks?|jaw(?:line)?)\b", re.IGNORECASE), "광대·턱 과장 깎기 표현 금지"),
    (re.compile(r"(?<!no\s)(?<!zero\s)(?<!without\s)(?<!never\s)(?:button|barbie|upturned|pinched|razor[-\s]?sharp)\s+(?:(?:\w+\s+){0,1})?nose\b", re.IGNORECASE), "코 과장/성형 표현 금지"),
    (re.compile(r"(?<!no\s)(?<!zero\s)(?<!without\s)(?<!never\s)(?:heavy|dramatic|full[-\s]?glam|red[-\s]?carpet)\s+(?:(?:\w+\s+){0,1})?(?:makeup|eyeliner|eyeshadow|lipstick|contour(?:ing)?)\b", re.IGNORECASE), "과도한 화보 메이크업/윤곽 과장 표현 금지"),
    (re.compile(r"(?<!no\s)(?<!zero\s)(?<!without\s)(?<!never\s)(?:soft,\s+full,\s+gently\s+rounded|soft\s+rounded\s+chin|full\s+rounded\s+cheeks?)\b", re.IGNORECASE), "턱선·볼 둥글림/부풀림 유도 표현 금지 (원본 턱 각도 1:1 중립 유지)"),
]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(p: Path) -> dict:
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_json(p: Path, obj: dict) -> None:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _is_portrait_session(payload: dict, prev_entry: dict) -> bool:
    if prev_entry.get("ever_generated_portrait"):
        return True
    art_dir = str(payload.get("artifactDirectoryPath") or "").strip()
    if art_dir:
        up_dir = Path(art_dir) / ".user_uploaded"
        if up_dir.is_dir() and any(up_dir.glob("*.jpg")):
            return True
    for wp in payload.get("workspacePaths") or []:
        if "모델생성" in str(wp) or "인물사진" in str(wp):
            return True
    return False


def _get_or_update_turn_entry(payload: dict) -> tuple[str, dict, dict]:
    conv_id = str(payload.get("conversationId") or "default")
    last_input = str(payload.get("lastUserInput") or "").strip()
    state = _load_json(STATE_PATH)
    entry = state.get(conv_id) or {}
    if last_input:
        input_hash = hashlib.sha256(last_input.encode("utf-8")).hexdigest()[:16]
        if entry.get("last_input_hash") != input_hash:
            is_portrait = _is_portrait_session(payload, entry)
            needs_restart = bool(is_portrait and REGENERATION_ACTION_PATTERN.search(last_input))
            entry = {
                "last_input_hash": input_hash,
                "turn_started_epoch": time.time(),
                "last_input_preview": last_input[:120],
                "ever_generated_portrait": bool(entry.get("ever_generated_portrait", is_portrait)),
                "requires_fresh_restart_from_stage1": needs_restart,
                "stage1_prepare_ran": False,
                "fresh_gen_created": False,
                "finalized_4k_receipt": False,
                "macro_visually_inspected": False,
                "stop_nudge_count": 0,
            }
            state[conv_id] = entry
            _save_json(STATE_PATH, state)
    return conv_id, entry, state


def _is_safe_origin(path_str: str) -> bool:
    return any(pat.search(path_str) for pat in SAFE_ORIGIN_PATTERNS)


def _check_gen_input_path(img_str: str, ledger: dict, is_turnaround_target: bool = False) -> str | None:
    if _is_safe_origin(img_str):
        return None

    for pat, reason in FORBIDDEN_GEN_INPUT_PATTERNS:
        if pat.search(img_str):
            return (
                f"[원본재생성·1원본1생성 훅 강제 차단] '{Path(img_str).name}': {reason}. "
                "매번 처음 원본(Gen-0) 사진만 입력해 새롭게 생성해야 합니다."
            )

    if is_turnaround_target and TURNAROUND_GENERATED_INPUT_PATTERN.search(img_str):
        return (
            f"[다단계 체이닝 훅 강제 차단] '{Path(img_str).name}': 이미 생성된 8방향/턴어라운드 이미지를 다시 다른 각도 생성에 "
            "투입하는 다단계 연쇄 생성(Hop-Chaining)은 절대 금지됩니다! 모든 8방향은 오직 순수 Gen-0 원본에서만 1-Hop 생성하세요."
        )

    p = Path(img_str)
    if p.is_file():
        digest = _sha256(p)
        forb = ledger.get("forbidden_generated_outputs_sha256") or {}
        app_8 = ledger.get("approved_8dir_faces_sha256") or {}
        app_o = ledger.get("approved_origins_sha256") or {}
        if digest in forb and digest not in app_8 and digest not in app_o:
            return (
                f"[원본재생성·1원본1생성 훅 강제 차단] '{p.name}' 파일은 이전에 생성된 결과물(SHA-256 일치)입니다. "
                "생성물에 다시 생성하는 재먹이기는 절대 금지되며, 매번 처음 원본(Gen-0)으로 새로 생성해야 합니다."
            )
        if is_turnaround_target and digest in app_8 and int((app_8.get(digest) or {}).get("depth", 0)) >= 1:
            return (
                f"[다단계 체이닝 훅 강제 차단] '{p.name}' (depth>=1 8방향 생성물)을 다시 다른 각도 생성에 재투입할 수 없습니다."
            )
    return None


def _check_stale_postprocess_command(cmd: str, payload: dict, turn_started_epoch: float) -> str | None:
    if not STALE_TOUCH_CMD_PATTERN.search(cmd):
        return None
    if re.search(r"\b(?:unittest|pytest|test_model_generation_engine)\b", cmd):
        return None
    if turn_started_epoch <= 0.0:
        return None

    artifact_dir = str(payload.get("artifactDirectoryPath") or "").strip()
    found_paths = IMAGE_PATH_REGEX.findall(cmd)

    for raw_p in found_paths:
        if _is_safe_origin(raw_p):
            continue
        candidates_to_stat = [Path(raw_p)]
        if artifact_dir and not raw_p.startswith("/"):
            candidates_to_stat.append(Path(artifact_dir) / raw_p)
        for p in candidates_to_stat:
            if not p.is_file():
                continue
            name = p.name
            looks_generated_input = bool(
                re.search(
                    r"(?:^s\d+_v\d+_|^s03_|^raw_|^exp_|_c\d+_|_3x4_|_16x9_|_pure1pass_|_chintuck_|_verified_|_closeup|_macro|_final_)",
                    name,
                    re.IGNORECASE,
                )
            )
            if not looks_generated_input:
                continue
            is_only_output = bool(
                re.search(rf"(?:out_\w*|output_\w*|save\w*|--stem)\s*[= ]+[^\n]*{re.escape(name)}", cmd)
            ) and not bool(
                re.search(rf"(?:cand|raw|src|input|scene|winner|imread|open|v\d+\s*=)\w*[^\n]*{re.escape(name)}", cmd, re.IGNORECASE)
            )
            if is_only_output:
                continue
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if mtime < (turn_started_epoch - 2.0):
                return (
                    f"[원본재생성 강제 훅 차단] 이전 턴에 생성된 구 결과물 '{name}'을(를) 후처리·재보정·재감사할 수 없습니다! "
                    "반드시 프로세스 제일 처음(Stage 1 `run_fresh_origin_workflow.py prepare`)부터 순수 처음 원본(Gen-0)으로 "
                    "새 이미지를 먼저 생성한 뒤, 그 새 생성물에 대해서만 채점·업스케일·검증을 실행하세요."
                )
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", default="auto", choices=("auto", "pre_invocation", "pre_tool", "post_tool", "stop"))
    cli_args, _ = parser.parse_known_args()

    try:
        raw = sys.stdin.read().strip()
        if not raw:
            print(json.dumps({}))
            return 0
        payload = json.loads(raw)
        conv_id, entry, state = _get_or_update_turn_entry(payload)
        turn_started_epoch = float(entry.get("turn_started_epoch", 0.0))

        # ── STOP EVENT: block stopping if restart/fresh-gen required and ANY stage (1, 3, 5, 6) was skipped!
        if cli_args.event == "stop" or "terminationReason" in payload:
            if not payload.get("fullyIdle", True):
                print(json.dumps({}))
                return 0
            if str(payload.get("terminationReason") or "") in ("USER_CANCELED", "ERROR"):
                print(json.dumps({}))
                return 0

            needs_full_cycle = bool(entry.get("requires_fresh_restart_from_stage1") or entry.get("fresh_gen_created"))
            if needs_full_cycle and int(entry.get("stop_nudge_count", 0)) < 3:
                missing = []
                if entry.get("requires_fresh_restart_from_stage1") and not entry.get("stage1_prepare_ran"):
                    missing.append("Stage 1~2(`run_fresh_origin_workflow.py prepare` — 프로세스 제일 처음 순수 원본 계측·매니페스트 생성)")
                if not entry.get("fresh_gen_created"):
                    missing.append("Stage 3(순수 원본 Gen-0만으로 새 후보 병렬 재-생성 — 이전 결과물 재사용 금지)")
                if not entry.get("finalized_4k_receipt"):
                    missing.append("Stage 4~5(`run_fresh_origin_workflow.py rank -> finalize` — 단일 Upscayl 4x + 10대 게이트 + 매크로/눈매/외곽 비교 생성)")
                if not entry.get("macro_visually_inspected"):
                    missing.append("Stage 6(`read_file`로 최종 4K 매크로·400% 눈매·1:1 외곽 오버레이 육안 검수)")
                if missing:
                    entry["stop_nudge_count"] = int(entry.get("stop_nudge_count", 0)) + 1
                    state[conv_id] = entry
                    _save_json(STATE_PATH, state)
                    print(
                        json.dumps(
                            {
                                "decision": "continue",
                                "reason": (
                                    "[프로세스 처음부터 순수 원본 재-생성 강제 훅] 필수 워크플로우 단계가 누락되어 종료할 수 없습니다: "
                                    f"{' / '.join(missing)}. 반드시 프로세스 제일 처음(Stage 1)부터 끝까지 순수 원본으로 완료하세요."
                                ),
                            },
                            ensure_ascii=False,
                        )
                    )
                    return 0
            print(json.dumps({}))
            return 0

        # ── PRE_INVOCATION EVENT
        tool_call = payload.get("toolCall")
        if cli_args.event == "pre_invocation" or not tool_call:
            print(json.dumps({}))
            return 0

        t_name = str(tool_call.get("name") or "").lower()
        args = tool_call.get("args") or {}

        # ── POST_TOOL EVENT: track Stage 1 prepare, Stage 5 finalize, Stage 6 visual inspection!
        if cli_args.event == "post_tool":
            if not payload.get("error"):
                if t_name == "run_command":
                    cmd = str(args.get("CommandLine") or args.get("commandLine") or "")
                    if not re.search(r"\b(?:unittest|pytest)\b", cmd):
                        if STAGE1_PREPARE_CMD_PATTERN.search(cmd):
                            entry["stage1_prepare_ran"] = True
                            if "--turnaround-sheet" in cmd:
                                entry["stage1_8dir_verified"] = True
                            state[conv_id] = entry
                            _save_json(STATE_PATH, state)
                        if FINALIZE_SUCCESS_CMD_PATTERN.search(cmd):
                            entry["finalized_4k_receipt"] = True
                            state[conv_id] = entry
                            _save_json(STATE_PATH, state)
                elif t_name == "read_file":
                    fpath = str(args.get("Path") or args.get("path") or "")
                    if MACRO_INSPECT_FILE_PATTERN.search(fpath):
                        entry["macro_visually_inspected"] = True
                        state[conv_id] = entry
                        _save_json(STATE_PATH, state)
            print(json.dumps({}))
            return 0

        # ── PRE_TOOL EVENT: hard guardrails
        candidates: list[str] = []
        target_name = ""
        is_image_gen_call = False
        prompt_to_audit = ""

        if "generate_image" in t_name:
            is_image_gen_call = True
            target_name = str(args.get("ImageName") or args.get("imageName") or "")
            for k in ("ImagePaths", "imagePaths", "image_paths", "referenceImages"):
                val = args.get(k)
                if isinstance(val, list):
                    candidates.extend(str(x) for x in val)
            prompt_txt = str(args.get("Prompt") or args.get("prompt") or "")
            prompt_to_audit = prompt_txt
            if prompt_txt:
                candidates.extend(IMAGE_PATH_REGEX.findall(prompt_txt))

        elif t_name == "invoke_subagent":
            sub_type = str(args.get("TypeName") or args.get("typeName") or "").lower()
            if "image" in sub_type:
                is_image_gen_call = True
                target_name = str(args.get("Role") or args.get("role") or "")
                prompt_txt = str(args.get("Prompt") or args.get("prompt") or "")
                prompt_to_audit = prompt_txt
                candidates.extend(IMAGE_PATH_REGEX.findall(prompt_txt))

        elif t_name == "send_message":
            msg_txt = str(args.get("Message") or args.get("message") or "")
            if re.search(r"(?:generate|image|생성|이미지)", msg_txt, re.IGNORECASE):
                prompt_to_audit = msg_txt
                candidates.extend(IMAGE_PATH_REGEX.findall(msg_txt))

        elif t_name == "run_command":
            cmd = str(args.get("CommandLine") or args.get("commandLine") or "")
            if re.search(r"(?:^|\s|/)gti\s+", cmd):
                is_image_gen_call = True
                prompt_to_audit = cmd
                out_m = re.search(r"--output\s+(['\"]?)([^\s'\"]+)\1", cmd)
                if out_m:
                    target_name = out_m.group(2)
                for m in re.finditer(r"--image\s+(['\"]?)([^\s'\"]+)\1", cmd):
                    candidates.append(m.group(2))

            elif FLOW_CLI_CMD_PATTERN.search(cmd) and "--login" not in cmd and "--download-edit-url" not in cmd:
                is_image_gen_call = True
                prompt_to_audit = cmd
                out_m = re.search(r"--output\s+(['\"]?)([^\s'\"]+)\1", cmd)
                if out_m:
                    target_name = out_m.group(2)
                b_m = re.search(r"--batch-json\s+(['\"]?)([^\s'\"]+)\1", cmd)
                if b_m:
                    bp = Path(b_m.group(2))
                    target_name = target_name or bp.name
                    if bp.is_file():
                        try:
                            bjobs = json.loads(bp.read_text(encoding="utf-8"))
                            if isinstance(bjobs, list) and bjobs:
                                if all(TURNAROUND_TARGET_PATTERN.search(str(jb.get("tag") or jb.get("output") or "")) for jb in bjobs):
                                    target_name = "8dir_turnaround_batch"
                                for jb in bjobs:
                                    candidates.extend(str(x) for x in (jb.get("images") or []))
                                    prompt_to_audit += "\n" + str(jb.get("prompt") or "")
                        except Exception:
                            pass
                m_m = re.search(r"--manifest\s+(['\"]?)([^\s'\"]+)\1", cmd)
                if m_m:
                    mp = Path(m_m.group(2))
                    if mp.is_file():
                        try:
                            mf = json.loads(mp.read_text(encoding="utf-8"))
                            for ref in mf.get("ordered_reference_images") or []:
                                if ref.get("path"):
                                    candidates.append(str(ref["path"]))
                            for pr in mf.get("prompts") or []:
                                prompt_to_audit += "\n" + str(pr.get("prompt") or "")
                        except Exception:
                            pass
                for raw_p in IMAGE_PATH_REGEX.findall(cmd):
                    if target_name and (raw_p == target_name or Path(raw_p).name == Path(target_name).name):
                        continue
                    candidates.append(raw_p)

            # Enforce `--turnaround-sheet` on Stage 1 `run_fresh_origin_workflow.py prepare` (ZERO skipping 8-dir bank!)
            if STAGE1_PREPARE_CMD_PATTERN.search(cmd) and not re.search(r"\b(?:unittest|pytest)\b", cmd):
                if "run_fresh_origin_workflow.py" in cmd and "--turnaround-sheet" not in cmd:
                    print(
                        json.dumps(
                            {
                                "decision": "deny",
                                "reason": (
                                    "[8방향 필수 선행 훅 차단] `run_fresh_origin_workflow.py prepare` 실행 시 반드시 "
                                    "`--turnaround-sheet <검증된_8방향_마스터_시트>`를 지정해야 합니다! "
                                    "Step 1(원본 Gen-0 → -90°~+90° 8방향 개별 1-Hop 생성·조립·검증)을 먼저 완료한 뒤 진행하세요."
                                ),
                            },
                            ensure_ascii=False,
                        )
                    )
                    return 0

            stale_err = _check_stale_postprocess_command(cmd, payload, turn_started_epoch)
            if stale_err:
                print(json.dumps({"decision": "deny", "reason": stale_err}, ensure_ascii=False))
                return 0

        if is_image_gen_call and prompt_to_audit:
            low_p = prompt_to_audit.lower()
            for ex_pat, ex_reason in FORBIDDEN_FACIAL_EXAGGERATION_PATTERNS:
                for m in ex_pat.finditer(low_p):
                    prefix = low_p[max(0, m.start() - 80) : m.start()]
                    clause_prefix = re.split(r"[.;\n]", prefix)[-1]
                    if re.search(r"\b(?:never|do\s+not|don't|without|zero|no|avoid)\b", clause_prefix):
                        continue
                    print(
                        json.dumps(
                            {
                                "decision": "deny",
                                "reason": f"[얼굴 과장 전면 차단 훅] {ex_reason}. 절대 원본 얼굴의 눈·코·입·눈썹·턱·볼을 과장하거나 미화하지 마세요.",
                            },
                            ensure_ascii=False,
                        )
                    )
                    return 0

        is_turn_target = bool(TURNAROUND_TARGET_PATTERN.search(target_name) or HEADLESS_TARGET_PATTERN.search(target_name))
        if candidates:
            ledger = _load_json(LEDGER_PATH)
            for cp in candidates:
                err = _check_gen_input_path(cp, ledger, is_turnaround_target=is_turn_target)
                if err:
                    print(json.dumps({"decision": "deny", "reason": err}, ensure_ascii=False))
                    return 0

        # Enforce Stage 1 `prepare`, Mandatory Verified 8-Direction Bank, AND Max-2 Face Refs on Google Flow!
        if is_image_gen_call and not is_turn_target:
            has_gen0_portrait = any(_is_safe_origin(c) for c in candidates)
            has_8dir_bank_ref = any(VERIFIED_8DIR_BANK_REF_PATTERN.search(c) for c in candidates) or bool(
                entry.get("stage1_8dir_verified")
            )
            if has_gen0_portrait and not has_8dir_bank_ref:
                print(
                    json.dumps(
                        {
                            "decision": "deny",
                            "reason": (
                                "[8방향 필수 선행 훅 차단] Step 1(원본 Gen-0 → -90°~+90° 8방향 마스터 뱅크 생성·조립·검증)을 "
                                "건너뛰고 2D 원본 사진만으로 바로 화보/장면 생성을 호출할 수 없습니다! "
                                "반드시 먼저 8방향 1-Hop 뱅크를 완성하고 `prepare --turnaround-sheet`를 실행하세요."
                            ),
                        },
                        ensure_ascii=False,
                    )
                )
                return 0
            if t_name == "run_command" and FLOW_CLI_CMD_PATTERN.search(str(args.get("CommandLine") or args.get("commandLine") or "")):
                unique_face_refs = {
                    Path(c).name
                    for c in candidates
                    if (_is_safe_origin(c) or VERIFIED_8DIR_BANK_REF_PATTERN.search(c))
                    and not NON_FACE_CLEAN_PLATE_PATTERN.search(c)
                }
                if len(unique_face_refs) > 2:
                    print(
                        json.dumps(
                            {
                                "decision": "deny",
                                "reason": (
                                    f"[Google Flow 3장 이상 얼굴 참조 정체성 붕괴 차단] 얼굴 참조 이미지가 {len(unique_face_refs)}장 첨부되었습니다 "
                                    f"({', '.join(sorted(unique_face_refs))}). Google Flow Nano Banana Pro는 얼굴 참조 3장 이상 첨부 시 "
                                    "정체성 전이가 풀려 엉뚱한 인물이 나옵니다! 반드시 최대 2장(Gen-0 + 검증된 단독 8방향 1장)만 첨부하세요."
                                ),
                            },
                            ensure_ascii=False,
                        )
                    )
                    return 0
            if entry.get("requires_fresh_restart_from_stage1") and not entry.get("stage1_prepare_ran"):
                print(
                    json.dumps(
                        {
                            "decision": "deny",
                            "reason": (
                                "[Stage 1 처음부터 재시작 강제 훅 차단] 수정·재생성 요청 시 반드시 프로세스 제일 처음"
                                "(Stage 1: `scripts/run_fresh_origin_workflow.py prepare --turnaround-sheet ...`)을 먼저 실행해 "
                                "순수 원본(Gen-0) + 8방향 뱅크 25항목 계측·광학 앵커·매니페스트를 새로 만든 뒤에만 이미지 생성을 호출할 수 있습니다!"
                            ),
                        },
                        ensure_ascii=False,
                    )
                )
                return 0
            if has_gen0_portrait:
                entry["ever_generated_portrait"] = True
                entry["fresh_gen_created"] = True
                entry["eight_dir_bank_attached"] = has_8dir_bank_ref
                entry["finalized_4k_receipt"] = False
                entry["macro_visually_inspected"] = False
                state[conv_id] = entry
                _save_json(STATE_PATH, state)
    except Exception:
        pass

    print(json.dumps({"decision": "allow"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
