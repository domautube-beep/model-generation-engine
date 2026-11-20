#!/usr/bin/env python3
"""Workspace Antigravity Lifecycle Hook (`PreInvocation` + `PreToolUse`) enforcing:
1. `매번 순수 처음 원본(Gen-0) 재-생성 강제` (Zero Stale-Output Post-Hoc Tweaking / '거기서 띡 바꾸기' HARD BAN).
2. `1원본들 → 1생성` (Zero Scene Output Re-Feeding across `generate_image`, `invoke_subagent(image-generator)`, `send_message`, and `gti`).
3. `다단계 턴어라운드 연쇄 생성 금지` (Zero Hop-Chaining — `8dir -> 8dir` HARD BAN; all 8 directions MUST be 1-Hop from `Gen-0` ONLY).
4. `합성 보드 · 3D 프리비즈 와이어프레임 첨부 금지`.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

STATE_PATH = Path.home() / ".model_generation_studio" / "fresh_origin_turn_state.json"
LEDGER_PATH = Path.home() / ".model_generation_studio" / "origin_1gen_ledger.json"

FORBIDDEN_INPUT_PATTERNS = [
    (re.compile(r"(?:^|[/\\])live_test_", re.IGNORECASE), "생성된 테스트 결과물 재투입 금지"),
    (re.compile(r"(?:^|[/\\])(?:hero|5k_21x9|21x9|closeup|macro_verify)_", re.IGNORECASE), "생성된 5K/클로즈업 결과물 재투입 금지"),
    (re.compile(r"(?:^|[/\\])(?:raw_batch|raw_scene|nb_raw|nb_dop|nb_subj|s03_crimson|exp_m|exp_final)_", re.IGNORECASE), "1차 장면 생성물 재투입 금지"),
    (re.compile(r"(?:^|[/\\])s\d+_v\d+_", re.IGNORECASE), "이전 버전 생성 후보/결과물 재투입 금지 (반드시 순수 처음 원본 Gen-0만 입력할 것)"),
    (re.compile(r"(?:_verified_|_closeup|_macro)\b", re.IGNORECASE), "검증/클로즈업/매크로 파생 결과물 재투입 금지"),
    (re.compile(r"upscayl4k_|_upscayl", re.IGNORECASE), "Upscayl 변형본을 다시 생성기 입력으로 넣는 다단계 변형 금지 (순수 원본만 입력할 것)"),
    (re.compile(r"composite_styling_board", re.IGNORECASE), "합성 스타일링 보드 금지 (개별 원본 에셋을 직접 전달할 것)"),
    (re.compile(r"(?:^|[/\\])previz_.*\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE), "3D 프리비즈 이미지 첨부 금지 (포즈 텍스트 + 원본만 전달할 것)"),
    (re.compile(r"[/\\]batches[/\\][^/\\]+[/\\](?:raw_|hero_|closeup_)", re.IGNORECASE), "배치 출력물 재투입 금지"),
]

SAFE_ORIGIN_PATTERNS = [
    re.compile(r"[/\\]\.user_uploaded[/\\]", re.IGNORECASE),
    re.compile(r"[/\\]originals[/\\]", re.IGNORECASE),
    re.compile(r"[/\\]loc\d+.*_clean\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE),
]

TURNAROUND_TARGET_PATTERN = re.compile(r"(?:8dir|turn_|angle_[mp]\d)", re.IGNORECASE)
TURNAROUND_GENERATED_INPUT_PATTERN = re.compile(
    r"(?:^|[/\\])(?:t2_turn_|crisp_gen1_turn_|subj\d+_turn_|s\d+_8dir_|angle_[mp]\d)", re.IGNORECASE
)

POST_PROCESS_CMD_PATTERN = re.compile(
    r"(?:calibrate_to_gen0|MicroProportionCalibrator|proportion_calibrator|upscale_with_upscayl|"
    r"run_single_pass_5k|run_tonematched_5k_synthesis|integrate_relit_face|apply_facelock|"
    r"blend_redrawn_head|cv2\.remap)",
    re.IGNORECASE,
)

IMAGE_PATH_REGEX = re.compile(
    r"([A-Za-z0-9_./가-힣\-\\]+?\.(?:jpg|jpeg|png|webp))",
    re.IGNORECASE,
)


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


def _update_turn_state(payload: dict) -> float:
    conv_id = str(payload.get("conversationId") or "default")
    last_input = str(payload.get("lastUserInput") or "").strip()
    if not last_input:
        state = _load_json(STATE_PATH)
        return float((state.get(conv_id) or {}).get("turn_started_epoch", 0.0))

    input_hash = hashlib.sha256(last_input.encode("utf-8")).hexdigest()[:16]
    state = _load_json(STATE_PATH)
    entry = state.get(conv_id) or {}
    if entry.get("last_input_hash") != input_hash:
        now = time.time()
        state[conv_id] = {
            "last_input_hash": input_hash,
            "turn_started_epoch": now,
            "last_input_preview": last_input[:120],
        }
        _save_json(STATE_PATH, state)
        return now
    return float(entry.get("turn_started_epoch", 0.0))


def _is_safe_origin(path_str: str) -> bool:
    return any(pat.search(path_str) for pat in SAFE_ORIGIN_PATTERNS)


def _check_image_path(img_path_str: str, ledger: dict, is_turnaround_target: bool = False) -> str | None:
    if _is_safe_origin(img_path_str):
        return None

    for pat, reason in FORBIDDEN_INPUT_PATTERNS:
        if pat.search(img_path_str):
            return (
                f"[원본재생성·1원본1생성 훅 차단] '{Path(img_path_str).name}': {reason}. "
                "매번 처음 원본(Gen-0) 사진만 입력해 새롭게 생성해야 합니다."
            )

    if is_turnaround_target and TURNAROUND_GENERATED_INPUT_PATTERN.search(img_path_str):
        return (
            f"[다단계 체이닝 훅 차단] '{Path(img_path_str).name}': 이미 생성된 8방향/턴어라운드 이미지를 다시 다른 각도 생성에 "
            "투입하는 다단계 연쇄 생성(Hop-Chaining)은 절대 금지됩니다! 모든 8방향은 오직 순수 Gen-0 원본에서만 1-Hop 생성하세요."
        )

    p = Path(img_path_str)
    if p.is_file():
        digest = _sha256(p)
        forb = ledger.get("forbidden_generated_outputs_sha256") or {}
        app_8 = ledger.get("approved_8dir_faces_sha256") or {}
        app_o = ledger.get("approved_origins_sha256") or {}
        if digest in forb and digest not in app_8 and digest not in app_o:
            return (
                f"[원본재생성·1원본1생성 훅 차단] '{p.name}' 파일은 이전에 생성된 출력물(SHA-256 일치)입니다. "
                "생성한 이미지에 다시 생성하는 재먹이기는 엄격히 금지되며, 매번 처음 원본(Gen-0)으로 새로 생성해야 합니다."
            )
        if is_turnaround_target and digest in app_8 and int((app_8.get(digest) or {}).get("depth", 0)) >= 1:
            return (
                f"[다단계 체이닝 훅 차단] '{p.name}' (depth>=1 8방향 생성물)을 다시 다른 각도 생성에 재투입할 수 없습니다."
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
                re.search(rf"(?:out_\w*|output_\w*|save\w*)\s*=\s*[^\n]*{re.escape(name)}", cmd)
            ) and not bool(
                re.search(rf"(?:cand|raw|src|input|scene|imread|open)\w*[^\n]*{re.escape(name)}", cmd, re.IGNORECASE)
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
                    "그 새 생성물에 대해서만 최종 업스케일·검증을 실행하세요."
                )
    return None


def main() -> int:
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({}))
        return 0
    try:
        payload = json.loads(raw)
    except Exception:
        print(json.dumps({}))
        return 0

    turn_started_epoch = _update_turn_state(payload)
    tool_call = payload.get("toolCall")
    if not tool_call:
        print(json.dumps({}))
        return 0

    t_name = str(tool_call.get("name") or "").lower()
    args = tool_call.get("args") or {}
    ledger = _load_json(LEDGER_PATH)

    candidate_paths: list[str] = []
    target_name = ""

    if "generate_image" in t_name:
        target_name = str(args.get("ImageName") or args.get("imageName") or "")
        for k in ("ImagePaths", "imagePaths", "image_paths", "referenceImages"):
            val = args.get(k)
            if isinstance(val, list):
                candidate_paths.extend(str(x) for x in val)
        prompt_txt = str(args.get("Prompt") or args.get("prompt") or "")
        if prompt_txt:
            candidate_paths.extend(IMAGE_PATH_REGEX.findall(prompt_txt))
    elif t_name == "invoke_subagent":
        sub_type = str(args.get("TypeName") or args.get("typeName") or "").lower()
        if "image" in sub_type:
            target_name = str(args.get("Role") or args.get("role") or "")
            prompt_txt = str(args.get("Prompt") or args.get("prompt") or "")
            candidate_paths.extend(IMAGE_PATH_REGEX.findall(prompt_txt))
    elif t_name == "send_message":
        msg_txt = str(args.get("Message") or args.get("message") or "")
        if re.search(r"(?:generate|image|생성|이미지)", msg_txt, re.IGNORECASE):
            candidate_paths.extend(IMAGE_PATH_REGEX.findall(msg_txt))
    elif t_name == "run_command":
        cmd = str(args.get("CommandLine") or args.get("commandLine") or "")
        if re.search(r"(?:^|\s|/)gti\s+", cmd):
            out_m = re.search(r"--output\s+(['\"]?)([^\s'\"]+)\1", cmd)
            if out_m:
                target_name = out_m.group(2)
            for m in re.finditer(r"--image\s+(['\"]?)([^\s'\"]+)\1", cmd):
                candidate_paths.append(m.group(2))
        stale_err = _check_stale_postprocess_command(cmd, payload, turn_started_epoch)
        if stale_err:
            print(json.dumps({"decision": "deny", "reason": stale_err}, ensure_ascii=False))
            return 0

    is_turn_target = bool(TURNAROUND_TARGET_PATTERN.search(target_name))
    for cp in candidate_paths:
        err = _check_image_path(cp, ledger, is_turnaround_target=is_turn_target)
        if err:
            print(json.dumps({"decision": "deny", "reason": err}, ensure_ascii=False))
            return 0

    print(json.dumps({"decision": "allow"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
