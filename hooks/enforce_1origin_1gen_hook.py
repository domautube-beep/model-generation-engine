#!/usr/bin/env python3
"""Antigravity Lifecycle Hook (`PreToolUse` / `PostToolUse`) enforcing `1원본들 → 1생성`.

Blocks (`"decision": "deny"`) any tool call (`generate_image`, `run_command` calling `gti`, etc.) that:
1. Passes a previously generated scene image, 5K output, closeup crop, composite styling board, or 3D previz wireframe
   back into an image generator (`재먹이기 / Depth >= 2 차단`).
2. Attempts a 2nd `generate_image` pass inside an `image-generator` subagent after already generating an image (`1생성 잠금`).
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
    (re.compile(r"(?:^|[/\\])raw_batch_|(?:^|[/\\])raw_scene_", re.IGNORECASE), "배치 1차 생성물 재투입 금지"),
    (re.compile(r"composite_styling_board", re.IGNORECASE), "합성 스타일링 보드 금지 (개별 원본 에셋을 직접 전달할 것)"),
    (re.compile(r"(?:^|[/\\])previz_.*\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE), "3D 프리비즈 이미지 첨부 금지 (포즈 텍스트 + 8방향 얼굴 원본만 전달할 것)"),
    (re.compile(r"[/\\]batches[/\\][^/\\]+[/\\](?:raw_|hero_|closeup_)", re.IGNORECASE), "배치 출력물 재투입 금지"),
]


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


def _check_image_path(img_path_str: str, ledger: dict) -> str | None:
    for pat, reason in FORBIDDEN_INPUT_PATTERNS:
        if pat.search(img_path_str):
            return f"[1원본들→1생성 훅 차단] '{Path(img_path_str).name}': {reason}. 오직 8방향 얼굴 + 원본 에셋만 입력하세요."
    p = Path(img_path_str)
    if p.is_file():
        digest = _sha256(p)
        forb = ledger.get("forbidden_generated_outputs_sha256") or {}
        app_8 = ledger.get("approved_8dir_faces_sha256") or {}
        app_o = ledger.get("approved_origins_sha256") or {}
        if digest in forb and digest not in app_8 and digest not in app_o:
            return (
                f"[1원본들→1생성 훅 차단] '{p.name}' 파일은 이전에 생성된 출력물(SHA-256 일치)입니다. "
                "생성한 이미지에 다시 생성하는 행위는 엄격히 금지됩니다."
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

    # 1. Inspect built-in `generate_image` tool calls
    if "generate_image" in t_name:
        for k in ("ImagePaths", "imagePaths", "image_paths", "referenceImages"):
            val = args.get(k)
            if isinstance(val, list):
                candidate_paths.extend(str(x) for x in val)

    # 2. Inspect `run_command` calls invoking `gti` with `--image`
    elif t_name == "run_command":
        cmd = str(args.get("CommandLine") or args.get("commandLine") or "")
        if re.search(r"(?:^|\s|/)gti\s+", cmd):
            for m in re.finditer(r"--image\s+(['\"]?)([^\s'\"]+)\1", cmd):
                candidate_paths.append(m.group(2))

    for cp in candidate_paths:
        err = _check_image_path(cp, ledger)
        if err:
            print(json.dumps({"decision": "deny", "reason": err}, ensure_ascii=False))
            return 0

    print(json.dumps({"decision": "allow"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
