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
import struct
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
    r"([A-Za-z0-9_./가-힣@~+()\-\\]+?\.(?:jpg|jpeg|png|webp|tiff|bmp))",
    re.IGNORECASE,
)

QUOTED_IMAGE_PATH_REGEX = re.compile(
    r"""(?:'([^'\n]+?\.(?:jpg|jpeg|png|webp|tiff|bmp))'|"([^"\n]+?\.(?:jpg|jpeg|png|webp|tiff|bmp))")""",
    re.IGNORECASE,
)

# 수정·피드백·초기생성 지시 감지 (질문이 아닌 수정/조정/생성 지시는 모두 Stage 1부터 순수 원본 재시작 강제!)
REGENERATION_ACTION_PATTERN = re.compile(
    r"(?:다시\s*(?:해|생성|만들|뽑|돌려)|재생성|처음부터|제일\s*처음|과장.*잡아|싹다\s*잡아|"
    r"고쳐\s*(?:줘|라|서)|바꿔\s*(?:줘|라|서)|수정\s*(?:해|요청)?|새로\s*(?:생성|만들|뽑)|원본으로\s*(?:무조건\s*)?재생성|"
    r"뽑아\s*(?:줘|라)|다양하게\s*해\s*줘|만들어\s*(?:줘|라)|생성해\s*(?:줘|라)|찍어\s*(?:줘|라)|"
    r"등장하게|심플하게|보이게|나오게|해\s*줘|넣어\s*줘|빼\s*줘|줄여\s*줘|키워\s*줘|"
    r"클로즈업|아웃포커싱|배경.*(?:이하|이상|%|심플)|인물.*(?:이하|이상|%|비율))",
    re.IGNORECASE,
)

LEGIT_TEST_SUITE_CMD_PATTERN = re.compile(
    r"^\s*(?:cd\s+[^\n;&|]+\s*&&\s*)?(?:[A-Za-z0-9_./~-]+/)?(?:uv\s+run\s+(?:--[^\s;&|]+\s+)*)?(?:python3?(?:\.\d+)?\s+-m\s+(?:unittest|pytest)|pytest)\b"
)

PROTECTED_HOOK_OR_RECEIPT_FILE_PATTERN = re.compile(
    r"(?:fresh_origin_turn_state\.json|origin_1gen_ledger\.json|_workflow_receipt\.json|\.model_generation_studio)(?:\b|/|\\|$)",
    re.IGNORECASE,
)

BYPASS_FLAG_PATTERNS = [
    (re.compile(r"--allow-audit(?:-fail)?\b", re.IGNORECASE), "[10대 게이트 우회 물리 차단] `--allow-audit-fail` 플래그는 v1.0.6+에서 영구 폐기·차단되었습니다."),
    (re.compile(r"strict_10gate\s*=\s*(?:False|0|None)\b"), "[10대 게이트 우회 물리 차단] `strict_10gate=False/0/None` 호출은 영구 금지되었습니다."),
    (
        re.compile(r"min_face_laplacian_var\s*=\s*(?:-\d+(?:\.\d+)?|(?:0|[1-9]|1[0-6])(?:\.\d+)?|17\.[0-4]\d*|\.\d+)\b"),
        "[화질 기준 하향 조작 차단] `min_face_laplacian_var`를 17.5 미만으로 낮추는 우회는 금지됩니다.",
    ),
    (
        re.compile(r"min_identity_cosine_sim\s*=\s*(?:-\d+(?:\.\d+)?|0(?:\.[0-5]\d*|\.6[01]\d*)?|\.[0-5]\d*|\.6[01]\d*)\b"),
        "[정체성 기준 하향 조작 차단] `min_identity_cosine_sim`을 0.62 미만으로 낮추는 우회는 금지됩니다.",
    ),
    (
        re.compile(r"enforce_ocular_gate\s*=\s*(?:False|0|None)\b"),
        "[눈매·과장 게이트 비활성화 차단] `enforce_ocular_gate=False` 우회는 금지됩니다.",
    ),
    (
        re.compile(r"(?:object\.__setattr__|QualityGateAuditor\.\w+\s*=|FaceExaggerationGuard\.\w+\s*=|OcularAnatomyAnalyzer\.\w+\s*=)"),
        "[품질 감사기 런타임 몽키패칭 물리 차단] 감사기/설정 객체 변조 시도가 감지되어 차단되었습니다.",
    ),
    (
        re.compile(r"(?:b64decode|base64\s+(?:-d|--decode|-D)|\bexec\s*\(|\beval\s*\()"),
        "[난독화/동적 코드 실행 우회 차단] base64 디코딩 및 exec/eval 우회 실행은 금지됩니다.",
    ),
]

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


def _extract_image_paths(text: str) -> list[str]:
    """Extract quoted (with spaces/@/~) and unquoted image paths accurately."""
    out: list[str] = []
    seen: set[str] = set()
    for qm in QUOTED_IMAGE_PATH_REGEX.finditer(text):
        val = (qm.group(1) or qm.group(2) or "").strip()
        if val and val not in seen:
            seen.add(val)
            out.append(val)
    for um in IMAGE_PATH_REGEX.finditer(text):
        val = um.group(1).strip()
        if val and not any(val != s and val in s for s in seen) and val not in seen:
            seen.add(val)
            out.append(val)
    return out


def _extract_cli_flag(cmd: str, flag_names: tuple[str, ...]) -> str | None:
    """Extract `--flag val`, `--flag=val`, `--flag 'val with space'`, or `--flag=\"val with space\"`."""
    alt = "|".join(re.escape(fn) for fn in flag_names)
    m = re.search(rf"(?:^|\s)(?:{alt})(?:\s+|=)(?:'([^']+)'|\"([^\"]+)\"|([^\s'\";&|]+))", cmd)
    if not m:
        return None
    return (m.group(1) or m.group(2) or m.group(3) or "").strip() or None


def _is_portrait_session(payload: dict, prev_entry: dict) -> bool:
    if prev_entry.get("ever_generated_portrait"):
        return True
    art_dir = str(payload.get("artifactDirectoryPath") or "").strip()
    if art_dir:
        up_dir = Path(art_dir) / ".user_uploaded"
        if up_dir.is_dir() and any(up_dir.glob("*.*")):
            return True
    for wp in payload.get("workspacePaths") or []:
        if any(k in str(wp) for k in ("릴프로필", "프로필", "모델생성", "인물사진", "model-generation")):
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


def _check_gen_input_path(
    img_str: str,
    ledger: dict,
    is_turnaround_target: bool = False,
    conv_id: str = "",
) -> str | None:
    expanded_str = str(Path(img_str).expanduser()) if img_str.startswith("~") else img_str
    p = Path(expanded_str)
    forb = ledger.get("forbidden_generated_outputs_sha256") or {}
    app_8 = ledger.get("approved_8dir_faces_sha256") or {}
    app_o = ledger.get("approved_origins_sha256") or {}

    if p.is_file():
        digest = _sha256(p)
        # 1. Check cryptographic SHA-256 blacklist BEFORE _is_safe_origin so copying generated files into originals/ NEVER bypasses!
        if digest in forb and digest not in app_8 and digest not in app_o:
            return (
                f"[원본재생성·1원본1생성 훅 강제 차단] '{p.name}' 파일은 이전에 생성된 결과물(SHA-256 일치)입니다. "
                "파일명·폴더(originals/ 등)를 바꿔도 생성물에 다시 생성하는 다중 재생성은 절대 금지됩니다!"
            )
        if is_turnaround_target and digest in app_8 and int((app_8.get(digest) or {}).get("depth", 0)) >= 1:
            return (
                f"[다단계 체이닝 훅 강제 차단] '{p.name}' (depth>=1 8방향 생성물)을 다시 다른 각도 생성에 재투입할 수 없습니다."
            )

    # 2. Block ANY image residing inside another subagent's brain directory (e.g. raw image-generator outputs before rank/finalize!)
    m_brain = re.search(r"[/\\]antigravity[/\\]brain[/\\]([0-9a-fA-F-]{20,})[/\\]", expanded_str)
    if m_brain and conv_id and not conv_id.startswith("unit-test-") and m_brain.group(1) != conv_id:
        return (
            f"[서브에이전트 생성물 다중 재생성 원천 차단] '{p.name}' 파일은 이전 생성 서브에이전트({m_brain.group(1)[:8]})가 만든 생성물입니다! "
            "생성된 이미지를 다시 생성기에 넣어 재편집·재생성하는 행위는 절대 금지되며, 반드시 처음 원본(Gen-0)만 입력해야 합니다."
        )

    # 3. Forbidden derived filename patterns
    for pat, reason in FORBIDDEN_GEN_INPUT_PATTERNS:
        if pat.search(expanded_str):
            return (
                f"[원본재생성·1원본1생성 훅 강제 차단] '{Path(expanded_str).name}': {reason}. "
                "매번 처음 원본(Gen-0) 사진만 입력해 새롭게 생성해야 합니다."
            )

    if is_turnaround_target and TURNAROUND_GENERATED_INPUT_PATTERN.search(expanded_str):
        return (
            f"[다단계 체이닝 훅 강제 차단] '{Path(expanded_str).name}': 이미 생성된 8방향/턴어라운드 이미지를 다시 다른 각도 생성에 "
            "투입하는 다단계 연쇄 생성(Hop-Chaining)은 절대 금지됩니다! 모든 8방향은 오직 순수 Gen-0 원본에서만 1-Hop 생성하세요."
        )

    # 4. Default-Deny Strict Origin Allowlist in live sessions: ONLY untouched Gen-0 + verified 1-Hop 8-dir + clean location/anchor allowed!
    if conv_id and not conv_id.startswith("unit-test-"):
        digest_live = _sha256(p) if p.is_file() else ""
        allowed_gen0 = _is_safe_origin(expanded_str) or (digest_live and digest_live in app_o)
        allowed_8dir = (not is_turnaround_target) and (
            bool(VERIFIED_8DIR_BANK_REF_PATTERN.search(expanded_str)) or (digest_live and digest_live in app_8)
        )
        if not (allowed_gen0 or allowed_8dir):
            return (
                f"[비원본 이미지 입력 전면 차단 (Default-Deny Whitelist)] '{Path(expanded_str).name}' 파일은 승인된 순수 원본(Gen-0 / .user_uploaded / originals) "
                "또는 검증된 1-Hop 8방향 뱅크가 아닙니다! 임의 생성물·임시 파일을 생성기에 넣는 다중 재생성은 절대 금지됩니다."
            )

    return None


def _is_legit_test_suite_cmd(cmd: str) -> bool:
    """True ONLY for clean test-runner invocations (blocks `-c` inline scripts, `;` chaining, and CLI workflow piggybacking)."""
    if not LEGIT_TEST_SUITE_CMD_PATTERN.search(cmd):
        return False
    if re.search(r"(?:\s-c\s|;|&&\s*[^\n]*python|\|\||`|\$\(|run_fresh_origin_workflow\.py)", cmd):
        return False
    return True


def _read_image_dims_fast(p: Path) -> tuple[int, int] | None:
    """Pure Python stdlib PNG/JPEG/WebP header dimension reader (zero cv2/PIL dependency)."""
    try:
        with open(p, "rb") as f:
            head = f.read(64)
            if len(head) < 24:
                return None
            # PNG
            if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
                w, h = struct.unpack(">II", head[16:24])
                return int(w), int(h)
            # RIFF WebP
            if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
                chunk = head[12:16]
                if chunk == b"VP8X" and len(head) >= 30:
                    w = 1 + int.from_bytes(head[24:27], "little")
                    h = 1 + int.from_bytes(head[27:30], "little")
                    return int(w), int(h)
            # JPEG
            if head[:2] == b"\xff\xd8":
                f.seek(2)
                while True:
                    b = f.read(1)
                    if not b:
                        break
                    if b != b"\xff":
                        continue
                    while b == b"\xff":
                        b = f.read(1)
                    if not b:
                        break
                    marker = b[0]
                    if marker in (0xD8, 0xD9) or (0xD0 <= marker <= 0xD7):
                        continue
                    seg_len_b = f.read(2)
                    if len(seg_len_b) < 2:
                        break
                    seg_len = struct.unpack(">H", seg_len_b)[0]
                    if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                        sof = f.read(5)
                        if len(sof) == 5:
                            h, w = struct.unpack(">HH", sof[1:5])
                            return int(w), int(h)
                        break
                    if seg_len < 2:
                        break
                    f.seek(seg_len - 2, 1)
    except Exception:
        return None
    return None


def _verify_receipt_json(receipt_path: Path) -> tuple[bool, str | None]:
    """Verify a finalized 4K workflow receipt passed all 10 gates, true 4K resolution, min sharpness >= 17.5, and SHA-256."""
    if not receipt_path.is_file():
        return False, f"영수증 미생성 ({receipt_path.name})"
    rcp = _load_json(receipt_path)
    if not rcp:
        return False, f"영수증 JSON 파싱 실패 ({receipt_path.name})"
    if rcp.get("passed_10gates") is not True or rcp.get("stage") != "FINALIZED_10GATES_PASSED":
        return False, f"10대 품질 게이트 불합격 ({receipt_path.name}: {rcp.get('audit_message', 'FAILED')})"
    res = rcp.get("resolution") or [0, 0]
    if len(res) < 2 or max(int(res[0]), int(res[1])) < 3840 or min(int(res[0]), int(res[1])) < 2160:
        return False, f"4K 해상도 미달 ({receipt_path.name}: {res})"
    lap = rcp.get("face_laplacian_variance")
    if lap is None or float(lap) < 17.5:
        return False, f"얼굴 선명도(LapVar={lap} < 17.5) 미달 가짜 4K ({receipt_path.name})"
    raw_cv = rcp.get("raw_pre_upscale_canvas") or [0, 0]
    up_mod = str(rcp.get("upscayl_model") or "")
    if len(raw_cv) == 2 and (max(int(raw_cv[0]), int(raw_cv[1])) < 3840 or min(int(raw_cv[0]), int(raw_cv[1])) < 2160):
        if any(bad_m in up_mod for bad_m in ("optical_fullframe", "optical_lanczos", "hybrid_opt_face")):
            return False, f"저해상도({raw_cv}) 원본의 단순 Lanczos 리사이즈 가짜 4K 차단 ({receipt_path.name})"
    hero_p_str = str((rcp.get("deliverables") or {}).get("hero_4k") or "")
    if not hero_p_str or not Path(hero_p_str).is_file():
        return False, f"최종 4K 결과물 파일 미존재 ({receipt_path.name})"
    hero_p = Path(hero_p_str)
    dims = _read_image_dims_fast(hero_p)
    if dims is None or max(dims) < 3840 or min(dims) < 2160:
        return False, f"최종 4K 이미지 실제 픽셀 규격 미달 ({dims})"
    expected_sha = rcp.get("hero_4k_sha256")
    if not expected_sha or _sha256(hero_p) != str(expected_sha):
        return False, f"최종 4K 이미지 암호화 SHA-256 누락/불일치 — 사후 이미지 바꿔치기 차단 ({hero_p.name})"
    return True, None


def _audit_script_source_code(src_text: str, label: str) -> str | None:
    """Scan Python/Shell script source code for bypass flags, receipt/state tampering, or unauthorized fake-4K writes."""
    for pat, msg in BYPASS_FLAG_PATTERNS:
        if pat.search(src_text):
            return f"[스크립트 내부 우회 차단 ({label})] {msg}"
    if PROTECTED_HOOK_OR_RECEIPT_FILE_PATTERN.search(src_text):
        return f"[스크립트 내부 보안 파일 위변조 차단 ({label})] 스크립트 안에서 훅 상태·원장·영수증 파일을 참조/조작할 수 없습니다!"
    if re.search(r"(?:cv2\.resize|Image\.open.*\.resize|\.resize\s*\()", src_text) and re.search(
        r"(?:cv2\.imwrite|\.save\s*\(|write_bytes)", src_text
    ):
        if not re.search(r"(?:_grid|_panel|_6panel|_sheet|_contact)", src_text, re.IGNORECASE):
            return (
                f"[스크립트 내부 비공식 리사이즈 저장 차단 ({label})] 스크립트로 직접 이미지를 리사이즈·저장할 수 없습니다! "
                "반드시 공식 `run_fresh_origin_workflow.py finalize` (Tone-Locked True 4K + 10대 게이트)만 사용하세요."
            )
    return None


def _check_tamper_or_bypass(t_name: str, args: dict, payload: dict) -> str | None:
    """Block forgery of hook state / receipts, indirect script bypasses, CLI bypass flags, and fake-4K Lanczos saves."""
    if t_name in ("write_file", "edit_file", "replace_file_content", "multi_replace_file_content"):
        fpath = str(args.get("Path") or args.get("path") or args.get("TargetFile") or "")
        if PROTECTED_HOOK_OR_RECEIPT_FILE_PATTERN.search(fpath):
            return f"[보안 파일 위변조 물리 차단] '{Path(fpath).name}' 상태/원장/영수증 파일을 직접 생성·수정할 수 없습니다!"
        if re.search(r"\.(?:py|sh|zsh|bash)$", fpath, re.IGNORECASE) and not re.search(
            r"(?:[/\\]tests[/\\]test_|[/\\]hooks[/\\]enforce_1origin_1gen_hook\.py$|[/\\]원본1생성게이트\.py$|[/\\]fresh_origin_workflow\.py$|[/\\]run_fresh_origin_workflow\.py$|[/\\]config\.py$|[/\\]quality_gate\.py$|[/\\]upscaler\.py$)",
            fpath,
        ):
            content_parts = [
                str(args.get(k) or "")
                for k in ("Content", "content", "Replacement", "replacement", "ReplacementContent")
            ]
            for ch in args.get("ReplacementChunks") or []:
                if isinstance(ch, dict):
                    content_parts.append(str(ch.get("ReplacementContent") or ""))
            src_err = _audit_script_source_code("\n".join(content_parts), Path(fpath).name)
            if src_err:
                return src_err
        return None

    if t_name != "run_command":
        return None

    cmd = str(args.get("CommandLine") or args.get("commandLine") or "")
    if _is_legit_test_suite_cmd(cmd):
        return None

    for pat, msg in BYPASS_FLAG_PATTERNS:
        if pat.search(cmd):
            return msg

    # Block shell/python mutation of state, ledger, or *_workflow_receipt.json outside official finalize CLI
    if PROTECTED_HOOK_OR_RECEIPT_FILE_PATTERN.search(cmd):
        is_official_cli = bool(
            re.search(r"run_fresh_origin_workflow\.py\s+(?:prepare|rank|finalize)\b", cmd)
            and not re.search(r"(?:;|&&|\|\||\bsed\b|\bjq\b|\bwrite_text\b|\bjson\.dump)", cmd)
        )
        is_readonly_cat = bool(
            re.match(r"^\s*(?:cat|head|tail|ls|stat)\s+", cmd)
            and not re.search(r"(?:>|>>|\btee\b|;|&&|\|)", cmd)
        )
        if not is_official_cli and not is_readonly_cat:
            return "[보안 상태·영수증 조작 물리 차단] 셸/스크립트로 훅 상태·원장·영수증 JSON을 조작하거나 덮어쓸 수 없습니다!"

    # Inspect any executed external .py / .sh / .zsh script file on disk to defeat write_file -> python3 /tmp/x.py indirection!
    for sm in re.finditer(r"""(?:'([^'\n]+?\.(?:py|sh|zsh|bash))'|"([^"\n]+?\.(?:py|sh|zsh|bash))"|([A-Za-z0-9_./가-힣@~+()\-\\]+?\.(?:py|sh|zsh|bash)))""", cmd):
        sp_raw = (sm.group(1) or sm.group(2) or sm.group(3) or "").strip()
        if not sp_raw:
            continue
        if any(
            ok_s in sp_raw
            for ok_s in (
                "scripts/run_fresh_origin_workflow.py",
                "scripts/google_flow_ultra_cli.py",
                "scripts/version_ctl.py",
                "scripts/lexicon.py",
                "hooks/enforce_1origin_1gen_hook.py",
                "hooks/pre-commit-version-check.py",
                "install_reelprofile.sh",
            )
        ):
            continue
        sp = Path(sp_raw).expanduser()
        if sp.is_file():
            try:
                disk_src = sp.read_text(encoding="utf-8", errors="ignore")
                src_err = _audit_script_source_code(disk_src, sp.name)
                if src_err:
                    return src_err
            except Exception:
                pass

    # Block finalize with plain Lanczos / upscayl_hybrid on sub-4K winner image (supports `=` and spaces in quotes!)
    if re.search(r"run_fresh_origin_workflow\.py\s+finalize\b", cmd):
        up_val = _extract_cli_flag(cmd, ("--upscale-mode",))
        win_val = _extract_cli_flag(cmd, ("--winner",))
        if up_val in ("optical_fullframe", "optical_lanczos", "upscayl_hybrid") and win_val:
            wp = Path(win_val).expanduser()
            if not wp.is_absolute():
                art_d = str(payload.get("artifactDirectoryPath") or "").strip()
                if art_d and (Path(art_d) / wp).is_file():
                    wp = Path(art_d) / wp
            if wp.is_file():
                dims = _read_image_dims_fast(wp)
                if dims is not None and (max(dims) < 3840 or min(dims) < 2160):
                    return (
                        f"[가짜 4K 물리 차단] 후보 이미지 '{wp.name}'({dims[0]}x{dims[1]})는 네이티브 4K 미만이므로 "
                        f"`--upscale-mode {up_val}`(단순 리사이즈)로 저장할 수 없습니다! "
                        "반드시 기본값 `tonelocked_upscayl_4k`(Tone-Locked True 4K 초해상)를 사용하세요."
                    )

    # Block ad-hoc script/CLI resizing or directly creating deliverable images outside official `finalize`
    if not re.search(r"run_fresh_origin_workflow\.py\s+(?:prepare|rank|finalize)\b", cmd) and not FLOW_CLI_CMD_PATTERN.search(cmd):
        if re.search(r"(?:cv2\.imwrite|\.save\(|write_bytes|shutil\.copy|sips\s+-|ffmpeg\b|\bcp\s+|\bmv\s+|\bln\s+-)", cmd):
            for p_cand in _extract_image_paths(cmd):
                p_low = Path(p_cand).name.lower()
                if any(tag in p_low for tag in ("_macro", "_closeup", "_grid", "_panel", "_6panel", "_sheet", "_contact", "_overlay", "_compare")):
                    continue
                if re.search(r"(?:_4k|_final|_master|_hero|_upscaled)\.(?:jpg|jpeg|png|webp|tiff|bmp)$", p_low) or re.search(
                    r"(?:sips\s+-[Zz]|cv2\.resize|\.resize\s*\()", cmd
                ):
                    # Allow pure `cp` of an ALREADY-verified hero_4k ONLY if source file has a valid passing receipt next to it
                    if re.search(r"^\s*cp\s+", cmd) and not re.search(r"(?:cv2|PIL|sips|ffmpeg|resize|;|&&|\|)", cmd):
                        all_imgs = _extract_image_paths(cmd)
                        if len(all_imgs) >= 2:
                            sp = Path(all_imgs[0]).expanduser()
                            stem_base = re.sub(r"_4k\.(?:jpg|jpeg|png|webp)$", "", sp.name, flags=re.IGNORECASE)
                            rcp_cand = sp.parent / f"{stem_base}_workflow_receipt.json"
                            ok_r, _ = _verify_receipt_json(rcp_cand)
                            if ok_r:
                                continue
                    return (
                        f"[비공식 가짜 4K/우회 결과물 직접 생성 차단] '{Path(p_cand).name}' 파일을 `run_fresh_origin_workflow.py finalize` "
                        "외부에서 임의 리사이즈·저장·복사할 수 없습니다! 반드시 공식 Tone-Locked True 4K + 10대 게이트를 통과하세요."
                    )

    return None


def _check_stale_postprocess_command(cmd: str, payload: dict, turn_started_epoch: float) -> str | None:
    if not STALE_TOUCH_CMD_PATTERN.search(cmd):
        return None
    if _is_legit_test_suite_cmd(cmd):
        return None
    if turn_started_epoch <= 0.0:
        return None

    artifact_dir = str(payload.get("artifactDirectoryPath") or "").strip()
    found_paths = _extract_image_paths(cmd)

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

            is_synthetic_unit_conv = conv_id.startswith("unit-test-")
            needs_full_cycle = bool(entry.get("requires_fresh_restart_from_stage1") or entry.get("fresh_gen_created"))
            if needs_full_cycle and int(entry.get("stop_nudge_count", 0)) < 25:
                missing = []
                if entry.get("requires_fresh_restart_from_stage1") and not entry.get("stage1_prepare_ran"):
                    missing.append("Stage 1~2(`run_fresh_origin_workflow.py prepare` — 프로세스 제일 처음 순수 원본 계측·매니페스트 생성)")
                if not entry.get("fresh_gen_created"):
                    missing.append("Stage 3(순수 원본 Gen-0만으로 새 후보 병렬 재-생성 — 이전 결과물 재사용 금지)")

                # Re-verify every receipt generated during this turn on disk (blocks failed/forged/missing receipts on Stop!)
                tracked_dirs = [Path(d) for d in (entry.get("tracked_output_dirs") or []) if Path(d).is_dir()]
                art_dir_s = str(payload.get("artifactDirectoryPath") or "").strip()
                if art_dir_s and Path(art_dir_s).is_dir() and Path(art_dir_s) not in tracked_dirs:
                    tracked_dirs.append(Path(art_dir_s))
                receipt_failures: list[str] = []
                verified_any_receipt = False
                for td_item in tracked_dirs:
                    for rcp_p in td_item.glob("*_workflow_receipt.json"):
                        try:
                            if rcp_p.stat().st_mtime < (turn_started_epoch - 2.0):
                                continue
                        except OSError:
                            continue
                        ok_r, err_r = _verify_receipt_json(rcp_p)
                        if ok_r:
                            verified_any_receipt = True
                        elif err_r:
                            receipt_failures.append(err_r)

                if not is_synthetic_unit_conv and not verified_any_receipt:
                    entry["finalized_4k_receipt"] = False
                elif receipt_failures and not verified_any_receipt:
                    entry["finalized_4k_receipt"] = False
                if entry.get("last_finalize_reject_reason") and not verified_any_receipt:
                    receipt_failures.append(str(entry["last_finalize_reject_reason"]))

                if not entry.get("finalized_4k_receipt"):
                    extra_err = f" [최근 사유: {'; '.join(dict.fromkeys(receipt_failures))}]" if receipt_failures else ""
                    missing.append(
                        "Stage 4~5(`run_fresh_origin_workflow.py rank -> finalize` — Tone-Locked True 4K 초해상 + 10대 게이트 전원 통과 필수)"
                        + extra_err
                    )
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
                                    "[프로세스 처음부터 순수 원본 재-생성 & 진짜 4K 10대 게이트 강제 훅] 필수 조건 미충족으로 종료할 수 없습니다: "
                                    f"{' / '.join(missing)}. 우회 없이 순수 원본으로 다시 생성하여 10대 게이트를 통과시키세요."
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

        # ── POST_TOOL EVENT: track Stage 1 prepare, Stage 5 finalize (ONLY if 10-gate receipt PASSED!), Stage 6 visual inspection!
        if cli_args.event == "post_tool":
            is_synthetic_unit_conv = conv_id.startswith("unit-test-")
            if not payload.get("error"):
                if t_name == "run_command":
                    cmd = str(args.get("CommandLine") or args.get("commandLine") or "")
                    if not _is_legit_test_suite_cmd(cmd):
                        if STAGE1_PREPARE_CMD_PATTERN.search(cmd):
                            wdir_val = _extract_cli_flag(cmd, ("--work-dir",))
                            ts_val = _extract_cli_flag(cmd, ("--turnaround-sheet",))
                            if wdir_val:
                                mf_p = Path(wdir_val).expanduser() / "workflow_manifest.json"
                                mf_obj = _load_json(mf_p)
                                if mf_p.is_file() and mf_p.stat().st_mtime >= (turn_started_epoch - 2.0) and mf_obj.get("turnaround_master_sheet_path"):
                                    entry["stage1_prepare_ran"] = True
                                    entry["stage1_8dir_verified"] = True
                            elif is_synthetic_unit_conv:
                                entry["stage1_prepare_ran"] = True
                                if ts_val or "--turnaround-sheet" in cmd:
                                    entry["stage1_8dir_verified"] = True
                            state[conv_id] = entry
                            _save_json(STATE_PATH, state)
                        if FINALIZE_SUCCESS_CMD_PATTERN.search(cmd):
                            out_val = _extract_cli_flag(cmd, ("--output-dir",))
                            stem_val = _extract_cli_flag(cmd, ("--stem",))
                            if out_val and stem_val:
                                out_d = Path(out_val).expanduser()
                                t_dirs = list(entry.get("tracked_output_dirs") or [])
                                if str(out_d) not in t_dirs:
                                    t_dirs.append(str(out_d))
                                entry["tracked_output_dirs"] = t_dirs
                                rcp_file = out_d / f"{stem_val}_workflow_receipt.json"
                                ok_rcp, rcp_err = _verify_receipt_json(rcp_file)
                                if ok_rcp:
                                    entry["finalized_4k_receipt"] = True
                                    entry["last_finalize_reject_reason"] = None
                                else:
                                    entry["finalized_4k_receipt"] = False
                                    entry["last_finalize_reject_reason"] = rcp_err
                            elif is_synthetic_unit_conv:
                                entry["finalized_4k_receipt"] = True
                            else:
                                entry["finalized_4k_receipt"] = False
                                entry["last_finalize_reject_reason"] = "finalize 명령에 유효한 --output-dir 및 --stem 영수증이 없습니다"
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
        tamper_or_bypass_err = _check_tamper_or_bypass(t_name, args, payload)
        if tamper_or_bypass_err:
            print(json.dumps({"decision": "deny", "reason": tamper_or_bypass_err}, ensure_ascii=False))
            return 0
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
                candidates.extend(_extract_image_paths(prompt_txt))

        elif t_name == "invoke_subagent":
            sub_type = str(args.get("TypeName") or args.get("typeName") or "").lower()
            prompt_txt = str(args.get("Prompt") or args.get("prompt") or "")
            if "image" in sub_type:
                is_image_gen_call = True
                target_name = str(args.get("Role") or args.get("role") or "")
                prompt_to_audit = prompt_txt
                candidates.extend(_extract_image_paths(prompt_txt))
            elif sub_type in ("self", "coding", "general") and _is_portrait_session(payload, entry):
                # Block delegating bypasses to a secondary coding subagent
                sub_err = _audit_script_source_code(prompt_txt, f"subagent:{sub_type}")
                if sub_err:
                    print(json.dumps({"decision": "deny", "reason": sub_err}, ensure_ascii=False))
                    return 0

        elif t_name == "send_message":
            recip = str(args.get("Recipient") or args.get("recipient") or "").strip()
            msg_txt = str(args.get("Message") or args.get("message") or "")
            # Block ALL multi-turn iterative regeneration on an existing image-generator subagent!
            recip_brain = Path.home() / ".gemini" / "antigravity" / "brain" / recip if recip else None
            is_existing_img_subagent = bool(
                recip in set(entry.get("spawned_img_subagents") or [])
                or (recip_brain and recip_brain.is_dir() and any(recip_brain.glob("*.png")))
            )
            if is_existing_img_subagent or (
                _is_portrait_session(payload, entry)
                and re.search(
                    r"(?:generate_image|image-generator|다시\s*(?:그려|생성|만들|뽑)|수정해|고쳐|덧그려|바꿔서\s*(?:생성|그려)|portrait.*regenerat)",
                    msg_txt,
                    re.IGNORECASE,
                )
            ):
                print(
                    json.dumps(
                        {
                            "decision": "deny",
                            "reason": (
                                "[기존 서브에이전트 다중 재생성(Image-to-Image) 물리 차단] 이미 이미지를 생성한 서브에이전트에 `send_message`로 "
                                "재생성·수정을 지시하면 이전 생성물 위에 덧그리는 다중 재생성이 발생합니다! "
                                "수정 요청 시에도 반드시 프로세스 제일 처음(Stage 1 `prepare`)으로 돌아가 순수 원본(Gen-0)만으로 새 `image-generator`를 호출하세요."
                            ),
                        },
                        ensure_ascii=False,
                    )
                )
                return 0
            found_msg_imgs = _extract_image_paths(msg_txt)
            if found_msg_imgs and re.search(r"(?:generate|image|portrait|생성|이미지|화보|프로필)", msg_txt, re.IGNORECASE):
                is_image_gen_call = True
                prompt_to_audit = msg_txt
                candidates.extend(found_msg_imgs)

        elif t_name == "run_command":
            cmd = str(args.get("CommandLine") or args.get("commandLine") or "")
            if re.search(r"(?:^|\s|/)gti\s+", cmd):
                is_image_gen_call = True
                prompt_to_audit = cmd
                out_v = _extract_cli_flag(cmd, ("--output", "-o"))
                if out_v:
                    target_name = out_v
                for m in re.finditer(r"(?:--image|-i)(?:\s+|=)(?:'([^']+)'|\"([^\"]+)\"|([^\s'\"]+))", cmd):
                    candidates.append(m.group(1) or m.group(2) or m.group(3))

            elif FLOW_CLI_CMD_PATTERN.search(cmd) and "--login" not in cmd and "--download-edit-url" not in cmd:
                is_image_gen_call = True
                prompt_to_audit = cmd
                out_v = _extract_cli_flag(cmd, ("--output",))
                if out_v:
                    target_name = out_v
                b_v = _extract_cli_flag(cmd, ("--batch-json",))
                if b_v:
                    bp = Path(b_v).expanduser()
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
                m_v = _extract_cli_flag(cmd, ("--manifest",))
                if m_v:
                    mp = Path(m_v).expanduser()
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
                for raw_p in _extract_image_paths(cmd):
                    if target_name and (raw_p == target_name or Path(raw_p).name == Path(target_name).name):
                        continue
                    candidates.append(raw_p)

            # Enforce `--turnaround-sheet` on Stage 1 prepare (blocks dummy non-existent sheet & headless bypass!)
            if STAGE1_PREPARE_CMD_PATTERN.search(cmd) and not _is_legit_test_suite_cmd(cmd):
                ts_arg = _extract_cli_flag(cmd, ("--turnaround-sheet",))
                if not ts_arg:
                    print(
                        json.dumps(
                            {
                                "decision": "deny",
                                "reason": (
                                    "[8방향 필수 선행 훅 차단] Stage 1 `prepare` 실행 시 반드시 "
                                    "`--turnaround-sheet <검증된_8방향_마스터_시트>`를 지정해야 합니다!"
                                ),
                            },
                            ensure_ascii=False,
                        )
                    )
                    return 0
                if not conv_id.startswith("unit-test-") and not Path(ts_arg).expanduser().is_file():
                    print(
                        json.dumps(
                            {
                                "decision": "deny",
                                "reason": f"[가짜 8방향 시트 차단] 지정한 `--turnaround-sheet` 파일이 실제로 존재하지 않습니다: {ts_arg}",
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
                err = _check_gen_input_path(cp, ledger, is_turnaround_target=is_turn_target, conv_id=conv_id)
                if err:
                    print(json.dumps({"decision": "deny", "reason": err}, ensure_ascii=False))
                    return 0

        # Enforce Stage 1 `prepare`, Mandatory Verified 8-Direction Bank, AND Max-2 Face Refs on Google Flow!
        if is_image_gen_call and not is_turn_target:
            has_any_portrait_ref = bool(candidates) and any(
                not NON_FACE_CLEAN_PLATE_PATTERN.search(c) for c in candidates
            )
            has_gen0_portrait = any(_is_safe_origin(c) for c in candidates) or (
                has_any_portrait_ref and _is_portrait_session(payload, entry)
            )
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
            # UNCONDITIONALLY require Stage 1 `prepare` in the CURRENT turn for EVERY live portrait scene generation (initial OR modification)!
            must_have_current_turn_prepare = (not conv_id.startswith("unit-test-")) or bool(
                entry.get("requires_fresh_restart_from_stage1")
            )
            if must_have_current_turn_prepare and not entry.get("stage1_prepare_ran"):
                print(
                    json.dumps(
                        {
                            "decision": "deny",
                            "reason": (
                                "[Stage 1 처음부터 재시작 강제 훅 차단] 최초 생성이든 수정·재생성 요청이든 매 턴 반드시 프로세스 제일 처음"
                                "(Stage 1: `scripts/run_fresh_origin_workflow.py prepare --turnaround-sheet ...`)을 먼저 실행해 "
                                "순수 원본(Gen-0) + 8방향 뱅크 계측 매니페스트를 새로 생성한 뒤에만 이미지 생성을 호출할 수 있습니다!"
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
    except Exception as exc:
        if cli_args.event == "pre_tool":
            print(
                json.dumps(
                    {
                        "decision": "deny",
                        "reason": f"[보안 훅 내부 예외 Fail-Closed 차단] 입력 검증 중 예외 발생으로 실행을 차단했습니다: {exc}",
                    },
                    ensure_ascii=False,
                )
            )
            return 0

    print(json.dumps({"decision": "allow"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
