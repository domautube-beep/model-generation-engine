#!/usr/bin/env python3
"""CLI for `릴프로필` (`프로필`) Canonical Version Check & Upgrade Governance (`정본 v1.0.0+`).

Usage:
  python3 scripts/version_ctl.py status
  python3 scripts/version_ctl.py check
  python3 scripts/version_ctl.py bump --part patch --summary "설명..."
  python3 scripts/version_ctl.py sync-baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_generation_engine.version_guard import (
    check_version_integrity,
    sync_or_bump_version,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="릴프로필 (프로필) 정본 v1.0.0+ 버전체크 및 업그레이드 관리 CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="현재 프로젝트 별칭, 정본 버전, 코드 해시 무결성 출력")
    p_chk = sub.add_parser("check", help="코드 변경 대비 버전 업데이트 누락 여부 검증 (실패 시 exit 1)")
    p_chk.add_argument("--json", action="store_true", help="JSON 포맷으로 출력")

    p_bump = sub.add_parser("bump", help="코드 수정/개발 후 버전 업그레이드 (v1.0.0 이후 상위 버전만 허용)")
    p_bump.add_argument("--part", choices=("patch", "minor", "major"), default="patch")
    p_bump.add_argument("--version", default=None, help="명시적 버전 지정 (예: 1.1.0)")
    p_bump.add_argument("--summary", required=True, help="변경/개발 요약 (필수)")

    p_sync = sub.add_parser("sync-baseline", help="정본 v1.0.0 초기 해시 동기화 전용")
    p_sync.add_argument("--summary", default="")

    args = parser.parse_args()

    if args.cmd == "status":
        rep = check_version_integrity(REPO_ROOT)
        print(json.dumps(rep.to_summary_dict(), ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "check":
        rep = check_version_integrity(REPO_ROOT)
        if args.json:
            print(json.dumps(rep.to_summary_dict(), ensure_ascii=False, indent=2))
        else:
            status_str = "PASS" if rep.passed else "FAIL (버전 업그레이드 필요)"
            print(f"[{rep.project_name} ({rep.short_alias}) — {rep.canonical_label}] Version Check: {status_str}")
            print(f"  canonical_version : {rep.canonical_version}")
            print(f"  live_code_sha256  : {rep.live_code_tree_sha256[:16]}...")
            if rep.changed_files:
                print(f"  changed_files     : {', '.join(rep.changed_files)}")
            for iss in rep.issues:
                print(f"  - {iss}")
        return 0 if rep.passed else 1

    if args.cmd == "bump":
        rep = sync_or_bump_version(
            REPO_ROOT,
            part=args.part,
            explicit_version=args.version,
            summary=args.summary,
            allow_baseline_sync=False,
        )
        print(f"[{rep.project_name}] 버전 업그레이드 완료 -> v{rep.canonical_version} (passed={rep.passed})")
        return 0 if rep.passed else 1

    if args.cmd == "sync-baseline":
        rep = sync_or_bump_version(
            REPO_ROOT,
            summary=args.summary,
            allow_baseline_sync=True,
        )
        print(f"[{rep.project_name}] 정본 베이스라인 해시 동기화 완료 -> v{rep.canonical_version} (passed={rep.passed})")
        return 0 if rep.passed else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
