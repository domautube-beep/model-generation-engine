#!/usr/bin/env python3
"""Git `pre-commit` Hook for `릴프로필` (`프로필` — 정본 v1.0.0+ Mandatory Version Check).

Blocks any git commit that modifies engine/CLI/hook Python files without updating
`VERSION.json` (`canonical_version >= 1.0.0` + matching `code_tree_sha256` via `scripts/version_ctl.py bump`).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_generation_engine.version_guard import check_version_integrity, parse_semver


def _git_lines(*args: str) -> list[str]:
    res = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True)
    if res.returncode != 0:
        return []
    return [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]


def _head_canonical_version() -> str | None:
    res = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", "HEAD:VERSION.json"],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0 or not res.stdout.strip():
        return None
    try:
        return str(json.loads(res.stdout).get("canonical_version") or "")
    except Exception:
        return None


def main() -> int:
    rep = check_version_integrity(REPO_ROOT)
    if not rep.passed:
        sys.stderr.write(
            "\n[릴프로필 버전체크 훅 차단] 코드가 수정되었으나 버전 원장(VERSION.json)이 동기화/업그레이드되지 않았습니다!\n"
        )
        for iss in rep.issues:
            sys.stderr.write(f"  - {iss}\n")
        if rep.changed_files:
            sys.stderr.write(f"  - 변경 감지 파일: {', '.join(rep.changed_files)}\n")
        sys.stderr.write(
            "\n해결 방법:\n"
            '  python3 scripts/version_ctl.py bump --part patch --summary "변경 요약"\n'
            "  git add VERSION.json model_generation_engine/__init__.py pyproject.toml docs/VERSION_HISTORY.md\n\n"
        )
        return 1

    staged = _git_lines("diff", "--cached", "--name-only")
    code_touched = [
        f
        for f in staged
        if (
            f.startswith("model_generation_engine/")
            or f.startswith("scripts/")
            or f.startswith("hooks/")
            or f == "launch_studio.py"
        )
        and f.endswith(".py")
    ]
    prev_ver = _head_canonical_version()
    if code_touched and prev_ver:
        if parse_semver(rep.canonical_version) <= parse_semver(prev_ver):
            sys.stderr.write(
                f"\n[릴프로필 버전 업그레이드 강제 훅 차단] 엔진 코드({', '.join(code_touched[:4])})가 수정되었습니다.\n"
                f"정본 v1.0.0 이후 수정분은 반드시 이전 버전(v{prev_ver})보다 상위 버전으로 업그레이드해야 합니다!\n"
                '실행: python3 scripts/version_ctl.py bump --part patch --summary "변경 요약"\n\n'
            )
            return 1
        if "VERSION.json" not in staged:
            sys.stderr.write(
                "\n[릴프로필 버전체크 훅 차단] 업데이트된 VERSION.json을 `git add VERSION.json`으로 스테이징하세요.\n\n"
            )
            return 1

    print(f"[릴프로필 버전체크 PASS] {rep.project_name} ({rep.short_alias}) -> v{rep.canonical_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
