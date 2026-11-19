#!/usr/bin/env python3
"""Antigravity Lifecycle Hook (`PreToolUse`) enforcing:
1. `1원본들 → 1생성` (Zero Scene Output Re-Feeding — `Depth == terminal` HARD BAN).
2. `다단계 턴어라운드 연쇄 생성 금지` (Zero Hop-Chaining — `8dir -> 8dir` HARD BAN; all 8 directions MUST be 1-Hop from `Gen-0` ONLY).
3. `합성 보드 · 3D 프리비즈 와이어프레임 첨부 금지`.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

FORBIDDEN_INPUT_PATTERNS = [
    (re.compile(r"(?:^|[/\\])live_test_", re.IGNORECASE), "생성된 테스트 결과물 재투입 금지"),
    (re.compile(r"(?:^|[/\\])(?:hero|5k_21x9|21x9|closeup|macro_verify)_", re.IGNORECASE), "생성된 5K/클로즈업 결과물 재투입 금지"),
    (re.compile(r"(?:^|[/\\])(?:raw_batch|raw_scene|nb_raw|nb_dop|nb_subj|s03_crimson)_", re.IGNORECASE), "1차 장면 생성물 재투입 금지"),
    (re.compile(r"composite_styling_board", re.IGNORECASE), "합성 스타일링 보드 금지 (개별 원본 에셋을 직접 전달할 것)"),
    (re.compile(r"(?:^|[/\\])previz_.*\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE), "3D 프리비즈 이미지 첨부 금지 (포즈 텍스트 + 8방향 얼굴 원본만 전달할 것)"),
    (re.compile(r"[/\\]batches[/\\][^/\\]+[/\\](?:raw_|hero_|closeup_)", re.IGNORECASE), "배치 출력물 재투입 금지"),
]

TURNAROUND_TARGET_PATTERN = re.compile(r"(?:8dir|turn_|angle_[mp]\d)", re.IGNORECASE)
TURNAROUND_GENERATED_INPUT_PATTERN = re.compile(
    r"(?:^|[/\\])(?:t2_turn_|crisp_gen1_turn_|subj\d+_turn_|s\d+_8dir_|angle_[mp]\d)", re.IGNORECASE
)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_ledger() -> dict:
    lp = Path.home() / ".model_generation_studio" / "origin_1gen_ledger.json"
    if lp.exists():
        try:
            return json.loads(lp.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _check_image_path(img_path_str: str, ledger: dict, is_turnaround_target: bool = False) -> str | None:
    for pat, reason in FORBIDDEN_INPUT_PATTERNS:
        if pat.search(img_path_str):
            return f"[레드팀 1원본들→1생성 훅 차단] '{Path(img_path_str).name}': {reason}. 생성물을 다시 생성기에 넣는 행위는 절대 금지됩니다."

    if is_turnaround_target and TURNAROUND_GENERATED_INPUT_PATTERN.search(img_path_str):
        return (
            f"[레드팀 다단계 체이닝 차단] '{Path(img_path_str).name}': 이미 생성된 8방향/턴어라운드 이미지를 다시 다른 각도 생성에 "
            "투입하는 다단계 연쇄 생성(Hop-Chaining)은 절대 금지됩니다! 모든 8방향은 오직 순수 Gen-0 원본에서만 1-Hop 생성해야 합니다."
        )

    p = Path(img_path_str)
    if p.is_file():
        digest = _sha256(p)
        forb = ledger.get("forbidden_generated_outputs_sha256") or {}
        app_8 = ledger.get("approved_8dir_faces_sha256") or {}
        app_o = ledger.get("approved_origins_sha256") or {}
        if digest in forb and digest not in app_8 and digest not in app_o:
            return (
                f"[레드팀 1원본들→1생성 훅 차단] '{p.name}' 파일은 이전에 생성된 출력물(SHA-256 일치)입니다. "
                "생성한 이미지에 다시 생성하는 재먹이기는 엄격히 금지됩니다."
            )
        if is_turnaround_target and digest in app_8 and int((app_8.get(digest) or {}).get("depth", 0)) >= 1:
            return (
                f"[레드팀 다단계 체이닝 차단] '{p.name}' (depth>=1 8방향 생성물)을 다시 다른 각도 생성에 재투입할 수 없습니다. "
                "오직 순수 Gen-0 원본만 입력하세요."
            )
    return None


def main() -> int:
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"decision": "allow"}))
        return 0
    try:
        payload = json.loads(raw)
    except Exception:
        print(json.dumps({"decision": "allow"}))
        return 0

    tool_call = payload.get("toolCall") or {}
    t_name = str(tool_call.get("name") or "").lower()
    args = tool_call.get("args") or {}
    ledger = _load_ledger()

    candidate_paths: list[str] = []
    target_name = ""

    if "generate_image" in t_name:
        target_name = str(args.get("ImageName") or args.get("imageName") or "")
        for k in ("ImagePaths", "imagePaths", "image_paths", "referenceImages"):
            val = args.get(k)
            if isinstance(val, list):
                candidate_paths.extend(str(x) for x in val)
    elif t_name == "run_command":
        cmd = str(args.get("CommandLine") or args.get("commandLine") or "")
        if re.search(r"(?:^|\s|/)gti\s+", cmd):
            out_m = re.search(r"--output\s+(['\"]?)([^\s'\"]+)\1", cmd)
            if out_m:
                target_name = out_m.group(2)
            for m in re.finditer(r"--image\s+(['\"]?)([^\s'\"]+)\1", cmd):
                candidate_paths.append(m.group(2))

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
