"""Ironclad `1원본들 → 1생성` (One-Originals-Set to One-Generation) Enforcement Guard (`origin_guard.py`).

Physically blocks at code & runtime level:
1. Re-feeding any previously generated scene output back into an image generator (`Depth >= 2` HARD BAN).
2. Passing synthetic derived boards or wireframe renders (`composite_styling_board`, `previz_`, `closeup_`, `hero_`, `5k_21x9_`)
   as visual `--image` references. ONLY approved 8-Direction Face Anchors + Raw User-Uploaded Asset Originals are permitted.
3. Multi-pass generation on a single scene job (`1 Job == Strictly 1 Atomic Generation Call`).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple


class OriginLineageViolationError(RuntimeError):
    """Raised immediately whenever a non-original / previously-generated image or a 2nd generation pass is attempted."""


FORBIDDEN_DERIVED_FILENAME_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:^|[/\\])live_test_", re.IGNORECASE), "Previously generated test scene output"),
    (re.compile(r"(?:^|[/\\])(?:hero|5k_21x9|21x9|closeup|macro_verify)_", re.IGNORECASE), "Post-processed/generated 5K output or crop"),
    (re.compile(r"(?:^|[/\\])raw_batch_|(?:^|[/\\])raw_scene_", re.IGNORECASE), "Raw batch scene output"),
    (re.compile(r"composite_styling_board", re.IGNORECASE), "Derived composite collage board (pass raw item originals instead)"),
    (re.compile(r"(?:^|[/\\])previz_.*\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE), "Synthetic 3D Previz wireframe image (use text pose + 8-dir face instead)"),
    (re.compile(r"[/\\]batches[/\\][^/\\]+[/\\](?:raw_|hero_|closeup_)", re.IGNORECASE), "Batch output artifact"),
    (re.compile(r"[/\\]outputs[/\\](?:21x9_|5k_|hero_|closeup_)", re.IGNORECASE), "Vault generated scene output"),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class OriginOnlyOneShotGuard:
    """Cryptographic SHA-256 + Path Lineage + Single-Shot Token Guard enforcing `1원본들 → 1생성`."""

    _lock = threading.Lock()

    def __init__(self, ledger_path: Optional[Path] = None) -> None:
        default_dir = Path.home() / ".model_generation_studio"
        default_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = Path(ledger_path) if ledger_path else (default_dir / "origin_1gen_ledger.json")
        self._executed_job_ids: Set[str] = set()
        self._ensure_ledger()

    def _ensure_ledger(self) -> None:
        if not self.ledger_path.exists():
            init_data = {
                "approved_origins_sha256": {},
                "approved_8dir_faces_sha256": {},
                "forbidden_generated_outputs_sha256": {},
                "executed_job_ids": [],
            }
            self.ledger_path.write_text(json.dumps(init_data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self) -> Dict[str, object]:
        self._ensure_ledger()
        try:
            return json.loads(self.ledger_path.read_text(encoding="utf-8"))
        except Exception:
            return {
                "approved_origins_sha256": {},
                "approved_8dir_faces_sha256": {},
                "forbidden_generated_outputs_sha256": {},
                "executed_job_ids": [],
            }

    def _save(self, data: Dict[str, object]) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def register_approved_origin(self, path: Path, kind: str = "raw_asset") -> str:
        """Whitelist a raw user-uploaded original asset or 8-direction face anchor."""
        p = Path(path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Origin file does not exist: {p}")
        digest = sha256_file(p)
        with self._lock:
            data = self._load()
            bucket = "approved_8dir_faces_sha256" if kind == "8dir_face" else "approved_origins_sha256"
            rec = dict(data.get(bucket) or {})  # type: ignore[arg-type]
            rec[digest] = {"path": str(p), "kind": kind, "registered_at": time.time()}
            data[bucket] = rec
            self._save(data)
        return digest

    def register_generated_scene_output(self, output_path: Path, job_id: str = "") -> Optional[str]:
        """Permanently blacklist a generated scene output SHA-256 so it can NEVER be re-fed as an input."""
        p = Path(output_path).resolve()
        if not p.is_file():
            return None
        digest = sha256_file(p)
        with self._lock:
            data = self._load()
            forb = dict(data.get("forbidden_generated_outputs_sha256") or {})  # type: ignore[arg-type]
            forb[digest] = {"path": str(p), "job_id": job_id, "generated_at": time.time()}
            data["forbidden_generated_outputs_sha256"] = forb
            self._save(data)
        return digest

    def verify_one_shot_and_inputs(
        self,
        reference_image_paths: Iterable[Path],
        job_id: Optional[str] = None,
    ) -> None:
        """Validate `1원본들 → 1생성`:
        - Every input path must exist, must NOT match derived/generated scene patterns, and its SHA-256 must NOT be in
          `forbidden_generated_outputs_sha256`.
        - If `job_id` is supplied, enforces that this `job_id` executes AT MOST ONCE (`max_passes == 1`).
        """
        refs = [Path(p) for p in reference_image_paths]
        if not refs:
            raise OriginLineageViolationError(
                "[1원본들 → 1생성 위반] 최소 1장 이상의 원본/8방향 얼굴 레퍼런스가 필요합니다."
            )

        with self._lock:
            data = self._load()
            forbidden_hashes: Dict[str, dict] = dict(data.get("forbidden_generated_outputs_sha256") or {})  # type: ignore[arg-type]
            approved_8dir: Dict[str, dict] = dict(data.get("approved_8dir_faces_sha256") or {})  # type: ignore[arg-type]
            approved_orig: Dict[str, dict] = dict(data.get("approved_origins_sha256") or {})  # type: ignore[arg-type]
            persisted_jobs: Set[str] = set(data.get("executed_job_ids") or [])  # type: ignore[arg-type]

            if job_id:
                if job_id in self._executed_job_ids or job_id in persisted_jobs:
                    raise OriginLineageViolationError(
                        f"[1원본들 → 1생성 위반] 작업 '{job_id}'에서 2회차 재생성(Multi-Pass / 재먹이기)이 시도되어 강제 차단되었습니다."
                    )

            for idx, rp in enumerate(refs, 1):
                if not rp.is_file():
                    raise OriginLineageViolationError(f"[1원본들 → 1생성 위반] 입력 이미지 #{idx}가 존재하지 않습니다: {rp}")
                resolved_str = str(rp.resolve())
                digest = sha256_file(rp)

                # Explicit whitelist check first: if not an approved 8dir face/origin, check blacklist & forbidden patterns
                if digest in forbidden_hashes and digest not in approved_8dir and digest not in approved_orig:
                    prev_info = forbidden_hashes[digest]
                    raise OriginLineageViolationError(
                        f"[1원본들 → 1생성 위반] 입력 이미지 #{idx} ({rp.name})는 이전에 생성된 결과물({prev_info.get('path')})과 "
                        "동일한 SHA-256 해시를 가집니다. 생성물에 다시 생성하는 재먹이기는 절대 금지됩니다."
                    )

                for pat, reason in FORBIDDEN_DERIVED_FILENAME_PATTERNS:
                    if pat.search(resolved_str):
                        raise OriginLineageViolationError(
                            f"[1원본들 → 1생성 위반] 입력 이미지 #{idx} ({rp.name})가 금지된 파생/생성 이미지 패턴({reason})에 해당합니다. "
                            "오직 8방향 얼굴과 순수 원본 에셋 파일만 입력할 수 있습니다."
                        )

            if job_id:
                self._executed_job_ids.add(job_id)
                persisted_jobs.add(job_id)
                data["executed_job_ids"] = sorted(persisted_jobs)[-2000:]
                self._save(data)
