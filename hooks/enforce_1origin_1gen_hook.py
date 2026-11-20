#!/usr/bin/env python3
"""전역 `매번 순수 원본(Gen-0) 6단계 필수 워크플로우 강제 게이트 훅` (`원본1생성게이트.py` v9.0).

지원 라이프사이클 이벤트:
1. `PreInvocation`: 사용자 입력(`lastUserInput`)이 바뀔 때마다 해당 대화(`conversationId`)의
   `turn_started_epoch`를 갱신하고 이번 턴 체크리스트(`fresh_gen_created`, `finalized_4k_receipt`,
   `macro_visually_inspected`, `stop_nudge_count`)를 초기화한다.
2. `PreToolUse` (`.*generate_image.*|invoke_subagent|send_message|run_command`):
   - **이전 턴 구 결과물 후처리('거기서 띡 바꾸기') 절대 금지**: `mtime < turn_started_epoch`인 구 생성물에
     후처리/재보정/재업스케일을 시도하면 즉시 `deny`.
   - **생성 결과물 재투입 금지 (`1원본들 → 1생성`)**: `generate_image`, `invoke_subagent(image-generator)`,
     `send_message`, `gti --image`에 이전 생성물 파일명이나 원장 SHA-256이 들어가면 즉시 `deny`.
   - **다단계 턴어라운드 연쇄 생성(`8dir -> 8dir`) · 합성 보드 · 3D 프리비즈 와이어프레임 첨부 금지**.
   - **순수 광학 `Gen-0` 눈매/얼굴 디테일 크롭(`gen0_optical_ocular_detail_anchor.jpg`) 및 사람제거 클린 배경(`loc*_clean.jpg`)은 허용**.
   - 실제 인물 이미지 생성 시 `fresh_gen_created = True` 기록.
3. `PostToolUse` (`run_command|read_file`):
   - `run_command`가 `run_fresh_origin_workflow.py finalize` 또는 `Upscayl + 10-Gate`를 통과(`workflow_receipt.json` 생성)하면
     `finalized_4k_receipt = True` 기록.
   - `read_file`이 `*_4k_macro.jpg` 또는 `*_eye_compare_400pct.jpg`를 육안 검수하면 `macro_visually_inspected = True` 기록.
4. `Stop` (`--event stop`):
   - 이번 턴에 인물 이미지 생성(`fresh_gen_created == True`)이 일어났는데,
     (a) `finalized_4k_receipt` (최종 `Upscayl 4x` + 10대 품질 게이트 + 매크로/400% 눈매 비교 스트립 생성) 또는
     (b) `macro_visually_inspected` (`read_file` 육안 검수) 중 하나라도 빠졌으면
     `{"decision": "continue", "reason": "..."}`를 반환하여 건너뛰기/빼먹기를 물리적으로 차단한다! (최대 2회 강제 보완)
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
    (re.compile(r"(?:_verified_|_closeup|_macro|_eye_compare_)\b", re.IGNORECASE), "검증/클로즈업/매크로 파생 결과물 재투입 금지"),
    (re.compile(r"upscayl4k_|_upscayl", re.IGNORECASE), "Upscayl 변형본을 다시 생성기 입력으로 넣는 다단계 변형 금지 (순수 원본만 입력할 것)"),
    (re.compile(r"composite_styling_board", re.IGNORECASE), "합성 스타일링 보드 금지 (개별 원본 에셋을 직접 전달할 것)"),
    (re.compile(r"(?:^|[/\\])previz_.*\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE), "3D 프리비즈 이미지 첨부 금지 (포즈 텍스트 + 원본만 전달할 것)"),
    (re.compile(r"[/\\]batches[/\\][^/\\]+[/\\](?:raw_|hero_|closeup_)", re.IGNORECASE), "배치 출력물 재투입 금지"),
]

SAFE_ORIGIN_PATTERNS = [
    re.compile(r"[/\\]\.user_uploaded[/\\]", re.IGNORECASE),
    re.compile(r"[/\\]originals[/\\]", re.IGNORECASE),
    re.compile(r"[/\\]loc\d+.*_clean\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE),
    re.compile(r"[/\\]gen0_optical_ocular_detail_anchor\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE),
]

TURNAROUND_TARGET_PATTERN = re.compile(r"(?:8dir|turn_|angle_[mp]\d)", re.IGNORECASE)
TURNAROUND_GENERATED_INPUT_PATTERN = re.compile(
    r"(?:^|[/\\])(?:t2_turn_|crisp_gen1_turn_|subj\d+_turn_|s\d+_8dir_|angle_[mp]\d)", re.IGNORECASE
)

POST_PROCESS_CMD_PATTERN = re.compile(
    r"(?:calibrate_to_gen0|MicroProportionCalibrator|proportion_calibrator|upscale_with_upscayl|"
    r"run_single_pass_5k|run_tonematched_5k_synthesis|integrate_relit_face|apply_facelock|"
    r"blend_redrawn_head|cv2\.remap|finalize_native_winner_4k|run_fresh_origin_workflow\.py\s+finalize)",
    re.IGNORECASE,
)

FINALIZE_SUCCESS_CMD_PATTERN = re.compile(
    r"(?:run_fresh_origin_workflow\.py\s+finalize|finalize_native_winner_4k|upscale_with_upscayl)",
    re.IGNORECASE,
)

MACRO_INSPECT_FILE_PATTERN = re.compile(
    r"(?:_4k_macro|_eye_compare_400pct|_4k)\.(?:jpg|jpeg|png|webp)$",
    re.IGNORECASE,
)

IMAGE_PATH_REGEX = re.compile(
    r"([A-Za-z0-9_./가-힣\-\\]+?\.(?:jpg|jpeg|png|webp))",
    re.IGNORECASE,
)

# 4. 얼굴 과장/미화/캐리커처 유발 프롬프트 즉시 차단 패턴 (절대 얼굴 보호)
FORBIDDEN_FACIAL_EXAGGERATION_PATTERNS = [
    (re.compile(r"(?<!no\s)(?<!zero\s)(?<!without\s)(?<!never\s)(?:plump(?:ed)?|pouty|luscious|voluminous|juicy|bee[-\s]?stung|glossy\s+red)\s+(?:(?:\w+\s+){0,1})?lips?\b", re.IGNORECASE), "입술 과장/필러화 표현 금지"),
    (re.compile(r"(?<!no\s)(?<!zero\s)(?<!without\s)(?<!never\s)(?:big|large|huge|doe|doll|anime|cat[-\s]?eye|foxy|siren)\s+(?:(?:\w+\s+){0,1})?eyes?\b", re.IGNORECASE), "눈 과장/인형눈 표현 금지"),
    (re.compile(r"(?<!no\s)(?<!zero\s)(?<!without\s)(?<!never\s)(?:sculpted|chiseled|razor|hollow(?:ed)?|contoured)\s+(?:(?:\w+\s+){0,1})?(?:cheekbones?|cheeks?|jaw(?:line)?)\b", re.IGNORECASE), "광대·턱 과장 깎기 표현 금지"),
    (re.compile(r"(?<!no\s)(?<!zero\s)(?<!without\s)(?<!never\s)(?:button|barbie|upturned|pinched|razor[-\s]?sharp)\s+(?:(?:\w+\s+){0,1})?nose\b", re.IGNORECASE), "코 과장/성형 표현 금지"),
    (re.compile(r"(?<!no\s)(?<!zero\s)(?<!without\s)(?<!never\s)(?:heavy|dramatic|full[-\s]?glam|red[-\s]?carpet)\s+(?:(?:\w+\s+){0,1})?(?:makeup|eyeliner|eyeshadow|lipstick|contour(?:ing)?)\b", re.IGNORECASE), "과도한 화보 메이크업/윤곽 과장 표현 금지"),
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


def _get_or_update_turn_entry(payload: dict) -> tuple[str, dict, dict]:
    conv_id = str(payload.get("conversationId") or "default")
    last_input = str(payload.get("lastUserInput") or "").strip()
    state = _load_json(STATE_PATH)
    entry = state.get(conv_id) or {}
    if last_input:
        input_hash = hashlib.sha256(last_input.encode("utf-8")).hexdigest()[:16]
        if entry.get("last_input_hash") != input_hash:
            entry = {
                "last_input_hash": input_hash,
                "turn_started_epoch": time.time(),
                "last_input_preview": last_input[:120],
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
    if not POST_PROCESS_CMD_PATTERN.search(cmd):
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
                    r"(?:^s\d+_v\d+_|^s03_|^raw_|^exp_|_c\d+_|_3x4_|_16x9_|_pure1pass_|_chintuck_|_verified_|_closeup|_macro)",
                    name,
                    re.IGNORECASE,
                )
            )
            if not looks_generated_input:
                continue
            is_only_output = bool(
                re.search(rf"(?:out_\w*|output_\w*|save\w*|--stem)\s*[= ]+[^\n]*{re.escape(name)}", cmd)
            ) and not bool(
                re.search(rf"(?:cand|raw|src|input|scene|winner|imread|open)\w*[^\n]*{re.escape(name)}", cmd, re.IGNORECASE)
            )
            if is_only_output:
                continue
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if mtime < (turn_started_epoch - 2.0):
                return (
                    f"[원본재생성 강제 훅 차단] 이전 턴에 생성된 구 결과물 '{name}'을(를) 후처리/재보정('거기서 띡 바꾸기')할 수 없습니다! "
                    "사용자 피드백이 들어오면 반드시 순수 처음 원본(Gen-0) 사진으로 새 이미지를 먼저 1회 생성한 뒤, "
                    "그 새 생성물에 대해서만 최종 업스케일·10대 게이트 검증을 실행하세요."
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

        # ── STOP EVENT: prevent stopping if fresh portrait was generated without 10-Gate 4K finalization + macro visual check!
        if cli_args.event == "stop" or "terminationReason" in payload:
            # Allow user-cancel / error / non-idle background waits cleanly
            if not payload.get("fullyIdle", True):
                print(json.dumps({}))
                return 0
            if str(payload.get("terminationReason") or "") in ("USER_CANCELED", "ERROR"):
                print(json.dumps({}))
                return 0
            if entry.get("fresh_gen_created") and int(entry.get("stop_nudge_count", 0)) < 2:
                missing = []
                if not entry.get("finalized_4k_receipt"):
                    missing.append(
                        "Stage 4~5(FreshOriginEditorialWorkflow rank -> finalize: 단일 Upscayl 4x + 10대 품질 게이트 + 4K/매크로/400% 눈매 비교 스트립)"
                    )
                if not entry.get("macro_visually_inspected"):
                    missing.append("Stage 6(`read_file`로 최종 4K 매크로 및 400% 눈매 비교 스트립 육안 검수)")
                if missing:
                    entry["stop_nudge_count"] = int(entry.get("stop_nudge_count", 0)) + 1
                    state[conv_id] = entry
                    _save_json(STATE_PATH, state)
                    print(
                        json.dumps(
                            {
                                "decision": "continue",
                                "reason": (
                                    "[필수 6단계 워크플로우 누락 방지 훅] 이번 턴에 인물 이미지를 새로 생성했지만 아래 필수 단계가 "
                                    f"완료되지 않았습니다: {' / '.join(missing)}. 건너뛰지 말고 반드시 완료한 뒤 보고하세요."
                                ),
                            },
                            ensure_ascii=False,
                        )
                    )
                    return 0
            print(json.dumps({}))
            return 0

        # ── PRE_INVOCATION EVENT: turn state already stamped above
        tool_call = payload.get("toolCall")
        if cli_args.event == "pre_invocation" or not tool_call:
            print(json.dumps({}))
            return 0

        t_name = str(tool_call.get("name") or "").lower()
        args = tool_call.get("args") or {}

        # ── POST_TOOL EVENT: track Stage 5 finalization & Stage 6 visual inspection!
        if cli_args.event == "post_tool":
            if not payload.get("error"):
                if t_name == "run_command":
                    cmd = str(args.get("CommandLine") or args.get("commandLine") or "")
                    if FINALIZE_SUCCESS_CMD_PATTERN.search(cmd) and not re.search(r"\b(?:unittest|pytest)\b", cmd):
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

        is_turn_target = bool(TURNAROUND_TARGET_PATTERN.search(target_name))
        if candidates:
            ledger = _load_json(LEDGER_PATH)
            for cp in candidates:
                err = _check_gen_input_path(cp, ledger, is_turnaround_target=is_turn_target)
                if err:
                    print(json.dumps({"decision": "deny", "reason": err}, ensure_ascii=False))
                    return 0

        # Mark that a fresh non-turnaround portrait generation has started in this turn
        if is_image_gen_call and not is_turn_target and any(_is_safe_origin(c) for c in candidates):
            entry["fresh_gen_created"] = True
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
