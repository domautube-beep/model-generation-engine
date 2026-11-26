"""Canonical Project Identity & Mandatory Automated Version-Check Guard for `릴프로필` (`프로필` — 정본 v1.0.0+).

Enforces the user's permanent project & versioning governance (`2026-11-24`):
1. Canonical Project Name: `릴프로필` (short alias: `프로필`; legacy compatible: `모델생성`).
   Calling ANY registered alias resolves to this exact Single-Source-of-Truth (`정본`) project.
2. Canonical Default Baseline (`정본 v1.0.0`):
   - Current unified state (1-Pass Native Full-Person 3D Synthesis + Mandatory 1-Hop 8-Dir Bank
     + Zero-Hardcoding Context-Adaptive 5-Lens [50/65/85/105/135mm] x 5-Scene-Matched Lighting Matrix
     + Auto Focal Clean Plates + De-Airbrushed Raw Micro-Pore Synthesis + Pure Optical Lanczos-4 Zero-Grain Default)
     is locked as `1.0.0` (`정본 v1.0.0`).
   - All prior development iterations are archived as historical `< 1.0.0` (`v0.1.0` .. `v0.9.5`) releases.
   - All future modifications/features MUST strictly upgrade above `1.0.0` (`1.0.1`, `1.1.0`, `2.0.0`, ...).
3. Mandatory Automated Version-Check (`check_version_integrity` + Git `pre-commit` hook):
   - Hashes every tracked engine/CLI/hook Python file (`code_tree_sha256`).
   - Detects any code edit that has not bumped `VERSION.json` / `__version__` / `pyproject.toml`.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CANONICAL_PROJECT_NAME: str = "릴프로필"
SHORT_PROJECT_NAME: str = "프로필"
PROJECT_ALIASES: Tuple[str, ...] = (
    "릴프로필",
    "프로필",
    "릴 프로필",
    "프로필프로젝트",
    "릴프로필스튜디오",
    "모델생성",
    "모델생성스튜디오",
    "reelprofile",
    "reel-profile",
    "reel_profile",
    "profile",
    "model-generation-engine",
)
MIN_CANONICAL_SEMVER: Tuple[int, int, int] = (1, 0, 0)

TRACKED_CODE_GLOBS: Tuple[str, ...] = (
    "model_generation_engine/**/*.py",
    "scripts/*.py",
    "hooks/*.py",
    "launch_studio.py",
)


def parse_semver(version_str: str) -> Tuple[int, int, int]:
    cleaned = str(version_str).strip().lstrip("vV")
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)$", cleaned)
    if not m:
        raise ValueError(f"Invalid semantic version '{version_str}' (expected MAJOR.MINOR.PATCH, e.g. '1.0.0').")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def format_semver(parts: Tuple[int, int, int]) -> str:
    return f"{parts[0]}.{parts[1]}.{parts[2]}"


def resolve_project_alias(name_or_query: str) -> bool:
    """Return True if `name_or_query` refers to `릴프로필` / `프로필` / legacy `모델생성`."""
    norm = re.sub(r"[\s_\-]+", "", str(name_or_query or "").strip().lower())
    if not norm:
        return False
    normalized_aliases = {re.sub(r"[\s_\-]+", "", a.lower()) for a in PROJECT_ALIASES}
    if norm in normalized_aliases:
        return True
    return any(alias in norm for alias in ("릴프로필", "reelprofile", "모델생성", "modelgenerationengine"))


def get_repo_root(repo_root: Optional[Path] = None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    return Path(__file__).resolve().parent.parent


def compute_engine_code_tree_sha256(repo_root: Optional[Path] = None) -> Tuple[str, Dict[str, str]]:
    """Compute a deterministic SHA-256 over all tracked engine/CLI/hook Python files."""
    root = get_repo_root(repo_root)
    rel_files: List[Path] = []
    seen = set()
    for pattern in TRACKED_CODE_GLOBS:
        for p in sorted(root.glob(pattern)):
            if not p.is_file():
                continue
            if "__pycache__" in p.parts or p.name.startswith("."):
                continue
            rel = p.relative_to(root)
            rel_posix = rel.as_posix()
            if rel_posix not in seen:
                seen.add(rel_posix)
                rel_files.append(rel)

    rel_files.sort(key=lambda r: r.as_posix())
    file_hashes: Dict[str, str] = {}
    master = hashlib.sha256()
    for rel in rel_files:
        raw_bytes = (root / rel).read_bytes().replace(b"\r\n", b"\n")
        f_sha = hashlib.sha256(raw_bytes).hexdigest()
        rel_posix = rel.as_posix()
        file_hashes[rel_posix] = f_sha
        master.update(f"{rel_posix}:{f_sha}\n".encode("utf-8"))
    return master.hexdigest(), file_hashes


def _read_pyproject_version(pyproject_path: Path) -> Optional[str]:
    if not pyproject_path.is_file():
        return None
    txt = pyproject_path.read_text(encoding="utf-8")
    m = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', txt)
    return m.group(1).strip() if m else None


def _read_init_version(init_path: Path) -> Optional[str]:
    if not init_path.is_file():
        return None
    txt = init_path.read_text(encoding="utf-8")
    m = re.search(r'(?m)^__version__\s*=\s*"([^"]+)"\s*$', txt)
    return m.group(1).strip() if m else None


@dataclass
class VersionIntegrityReport:
    passed: bool
    project_name: str
    short_alias: str
    aliases: List[str]
    canonical_version: str
    canonical_label: str
    init_version: Optional[str]
    pyproject_version: Optional[str]
    recorded_code_tree_sha256: str
    live_code_tree_sha256: str
    unbumped_code_changes_detected: bool
    changed_files: List[str]
    pre_v1_history_valid: bool
    issues: List[str]

    def to_summary_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def load_version_ledger(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = get_repo_root(repo_root)
    ledger_path = root / "VERSION.json"
    if not ledger_path.is_file():
        raise FileNotFoundError(f"Missing canonical version ledger: {ledger_path}")
    return json.loads(ledger_path.read_text(encoding="utf-8"))


_INTEGRITY_MEMO: Dict[str, Tuple[Tuple[Tuple[str, int, int], ...], int, int, "VersionIntegrityReport"]] = {}


def _stat_signature(root: Path) -> Tuple[Tuple[Tuple[str, int, int], ...], int, int]:
    entries: List[Tuple[str, int, int]] = []
    seen = set()
    for pattern in TRACKED_CODE_GLOBS:
        for p in sorted(root.glob(pattern)):
            if not p.is_file() or "__pycache__" in p.parts or p.name.startswith("."):
                continue
            rel_posix = p.relative_to(root).as_posix()
            if rel_posix not in seen:
                seen.add(rel_posix)
                st = p.stat()
                entries.append((rel_posix, int(st.st_mtime_ns), int(st.st_size)))
    entries.sort(key=lambda t: t[0])
    vj = root / "VERSION.json"
    pp = root / "pyproject.toml"
    vj_ns = int(vj.stat().st_mtime_ns) if vj.is_file() else 0
    pp_ns = int(pp.stat().st_mtime_ns) if pp.is_file() else 0
    return tuple(entries), vj_ns, pp_ns


def check_version_integrity(repo_root: Optional[Path] = None) -> VersionIntegrityReport:
    """Verify canonical `>= 1.0.0` version sync, `< 1.0.0` historical archive bounds, and zero unbumped code edits."""
    root = get_repo_root(repo_root)
    root_key = str(root)
    try:
        sig = _stat_signature(root)
        memo = _INTEGRITY_MEMO.get(root_key)
        if memo is not None and memo[:3] == sig:
            return memo[3]
    except Exception:
        sig = None
    issues: List[str] = []
    ledger = load_version_ledger(root)

    proj_name = str(ledger.get("project_name") or "")
    short_alias = str(ledger.get("short_name") or SHORT_PROJECT_NAME)
    aliases = list(ledger.get("aliases") or list(PROJECT_ALIASES))
    canon_ver = str(ledger.get("canonical_version") or "")
    canon_label = str(ledger.get("canonical_label") or f"정본 v{canon_ver}")
    recorded_sha = str(ledger.get("code_tree_sha256") or "")
    recorded_files: Dict[str, str] = dict(ledger.get("file_hashes") or {})

    if proj_name != CANONICAL_PROJECT_NAME:
        issues.append(f"project_name must be '{CANONICAL_PROJECT_NAME}' (got '{proj_name}').")
    for req_alias in ("릴프로필", "프로필", "모델생성"):
        if req_alias not in aliases:
            issues.append(f"Required alias '{req_alias}' missing from VERSION.json aliases.")

    try:
        sem = parse_semver(canon_ver)
        if sem < MIN_CANONICAL_SEMVER:
            issues.append(f"canonical_version '{canon_ver}' must be >= 1.0.0 (정본 v1 baseline).")
    except ValueError as exc:
        issues.append(str(exc))

    init_ver = _read_init_version(root / "model_generation_engine" / "__init__.py")
    pyproj_ver = _read_pyproject_version(root / "pyproject.toml")
    if init_ver != canon_ver:
        issues.append(f"__init__.__version__ ('{init_ver}') != VERSION.json ('{canon_ver}').")
    if pyproj_ver != canon_ver:
        issues.append(f"pyproject.toml version ('{pyproj_ver}') != VERSION.json ('{canon_ver}').")

    # Verify all pre-v1 historical entries are strictly < 1.0.0
    pre_v1_valid = True
    hist = ledger.get("history_pre_v1") or []
    if not hist:
        pre_v1_valid = False
        issues.append("history_pre_v1 must archive pre-v1 development history (< 1.0.0).")
    for item in hist:
        hv = str(item.get("version") or "")
        try:
            hsem = parse_semver(hv)
            if hsem >= (1, 0, 0):
                pre_v1_valid = False
                issues.append(f"Historical pre-v1 entry '{hv}' must be strictly < 1.0.0.")
        except ValueError as exc:
            pre_v1_valid = False
            issues.append(f"Invalid pre-v1 version '{hv}': {exc}")

    live_sha, live_files = compute_engine_code_tree_sha256(root)
    changed: List[str] = []
    if recorded_sha != live_sha:
        all_keys = sorted(set(recorded_files.keys()) | set(live_files.keys()))
        for k in all_keys:
            if recorded_files.get(k) != live_files.get(k):
                changed.append(k)
        if not changed:
            changed.append("<code_tree_sha256_mismatch>")
        issues.append(
            "Engine code was modified without bumping version! Run "
            "`python3 scripts/version_ctl.py bump --part patch --summary \"...\"` to record the upgrade."
        )

    rep = VersionIntegrityReport(
        passed=(len(issues) == 0),
        project_name=proj_name or CANONICAL_PROJECT_NAME,
        short_alias=short_alias,
        aliases=aliases,
        canonical_version=canon_ver,
        canonical_label=canon_label,
        init_version=init_ver,
        pyproject_version=pyproj_ver,
        recorded_code_tree_sha256=recorded_sha,
        live_code_tree_sha256=live_sha,
        unbumped_code_changes_detected=(recorded_sha != live_sha),
        changed_files=changed,
        pre_v1_history_valid=pre_v1_valid,
        issues=issues,
    )
    if sig is not None:
        _INTEGRITY_MEMO[root_key] = (sig[0], sig[1], sig[2], rep)
    return rep


def _write_version_strings(root: Path, new_ver: str) -> None:
    init_p = root / "model_generation_engine" / "__init__.py"
    if init_p.is_file():
        txt = init_p.read_text(encoding="utf-8")
        txt_new = re.sub(r'(?m)^__version__\s*=\s*"[^"]+"\s*$', f'__version__ = "{new_ver}"', txt)
        init_p.write_text(txt_new, encoding="utf-8")

    pyproj_p = root / "pyproject.toml"
    if pyproj_p.is_file():
        txt = pyproj_p.read_text(encoding="utf-8")
        txt_new = re.sub(r'(?m)^version\s*=\s*"[^"]+"\s*$', f'version = "{new_ver}"', txt, count=1)
        pyproj_p.write_text(txt_new, encoding="utf-8")


def sync_or_bump_version(
    repo_root: Optional[Path] = None,
    part: Optional[str] = None,
    explicit_version: Optional[str] = None,
    summary: str = "",
    allow_baseline_sync: bool = False,
) -> VersionIntegrityReport:
    """Atomically bump (`patch` | `minor` | `major`) or sync baseline (`1.0.0`) across code, pyproject, VERSION.json & docs."""
    root = get_repo_root(repo_root)
    ledger_path = root / "VERSION.json"
    ledger = load_version_ledger(root)
    cur_ver = str(ledger.get("canonical_version") or "1.0.0")
    cur_sem = parse_semver(cur_ver)

    _, pre_live_files = compute_engine_code_tree_sha256(root)
    recorded_files = dict(ledger.get("file_hashes") or {})
    changed_before = sorted(
        k for k in (set(recorded_files.keys()) | set(pre_live_files.keys())) if recorded_files.get(k) != pre_live_files.get(k)
    )

    if allow_baseline_sync and part is None and explicit_version is None:
        new_sem = cur_sem
    elif explicit_version is not None:
        new_sem = parse_semver(explicit_version)
        if new_sem <= cur_sem and not allow_baseline_sync:
            raise ValueError(f"New version {format_semver(new_sem)} must be strictly > current {cur_ver}.")
    else:
        p = (part or "patch").strip().lower()
        if p == "major":
            new_sem = (cur_sem[0] + 1, 0, 0)
        elif p == "minor":
            new_sem = (cur_sem[0], cur_sem[1] + 1, 0)
        elif p == "patch":
            new_sem = (cur_sem[0], cur_sem[1], cur_sem[2] + 1)
        else:
            raise ValueError(f"Unknown bump part '{part}' (choose 'patch', 'minor', or 'major').")

    if new_sem < MIN_CANONICAL_SEMVER:
        raise ValueError(f"Canonical version {format_semver(new_sem)} cannot be < 1.0.0.")

    new_ver = format_semver(new_sem)
    _write_version_strings(root, new_ver)

    live_sha, live_files = compute_engine_code_tree_sha256(root)
    now_iso = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).isoformat(timespec="seconds")

    ledger["canonical_version"] = new_ver
    ledger["canonical_label"] = "정본 v1.0.0 (Default Baseline)" if new_ver == "1.0.0" else f"릴프로필 v{new_ver}"
    ledger["updated_at"] = now_iso
    ledger["code_tree_sha256"] = live_sha
    ledger["file_hashes"] = live_files

    if not allow_baseline_sync or summary:
        upgrades = list(ledger.get("post_v1_upgrades") or [])
        if summary and (not upgrades or upgrades[-1].get("version") != new_ver):
            upgrades.append(
                {
                    "version": new_ver,
                    "date": now_iso[:10],
                    "timestamp": now_iso,
                    "summary": summary.strip(),
                    "changed_files": changed_before,
                }
            )
            ledger["post_v1_upgrades"] = upgrades

            hist_md = root / "docs" / "VERSION_HISTORY.md"
            if hist_md.is_file():
                md_txt = hist_md.read_text(encoding="utf-8")
                bullet = f"- **`v{new_ver}`** (`{now_iso[:10]}`): {summary.strip()}\n"
                marker = "<!-- POST_V1_UPGRADES -->"
                if marker in md_txt and f"`v{new_ver}`" not in md_txt:
                    md_txt = md_txt.replace(marker, f"{marker}\n{bullet}", 1)
                    hist_md.write_text(md_txt, encoding="utf-8")

    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return check_version_integrity(root)
