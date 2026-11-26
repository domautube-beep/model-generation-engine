"""Ironclad `8방향 필수 검증 + 1원본들 → 1생성 + 다단계 체이닝 절대 금지` Red-Team Enforcement Guard (`origin_guard.py` v5.0).

Physically blocks at code, API, and OS `PreToolUse` hook level:
1. **Mandatory 8-Direction Bank Verifier (`Mandatory8DirBankVerifier`)**:
   - Scene composition/generation is STRICTLY BLOCKED unless the subject has a complete, verified 8-Direction Turnaround Bank
     (`left_90`, `left_67`, `left_45`, `left_22`, `right_22`, `right_45`, `right_67`, `right_90`) + `front` (`Gen-0`).
2. **Zero Multi-Hop Chaining (`ChainedGenerationViolationError`)**:
   - **Stage 1 (`Gen-0 -> 8-Dir`)**: Every 8-direction turnaround image MUST be generated in **1 single hop strictly from `Gen-0`**
     (`depth == 0`). Feeding a generated `8dir` image to generate another `8dir` image (`Gen-1 -> Gen-2` chaining) is HARD-BANNED.
   - **Stage 2 (`Gen-0 + 1-Hop 8-Dir + Raw Assets -> Scene`)**: Every scene output is a **Terminal Node (`depth == terminal`)**
     and its SHA-256 is permanently blacklisted so it can NEVER be fed into any image generator again (`v1 -> v2` HARD-BANNED).
3. **Single-Shot Job Token Lock**: `1 Job == Strictly 1 Atomic Generation Call`.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

REQUIRED_8DIR_VIEWS: Tuple[str, ...] = (
    "left_90",
    "left_67",
    "left_45",
    "left_22",
    "right_22",
    "right_45",
    "right_67",
    "right_90",
)

LEGACY_8DIR_ALIASES: Dict[str, str] = {
    "left_profile": "left_90",
    "three_quarter_left": "left_45",
    "three_quarter_right": "right_45",
    "right_profile": "right_90",
}


class OriginLineageViolationError(RuntimeError):
    """Base error raised whenever lineage, 1-shot, or 8-direction rules are violated."""


class Incomplete8DirectionBankError(OriginLineageViolationError):
    """Raised when scene generation is attempted before all 8 turnaround directions (-90°..+90°) are built & verified."""


class ChainedGenerationViolationError(OriginLineageViolationError):
    """Raised when a generated output (including a generated 8-dir image in turnaround mode, or any scene output) is chained into another generation."""


FORBIDDEN_DERIVED_FILENAME_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:^|[/\\])live_test_", re.IGNORECASE), "Previously generated test scene output"),
    (re.compile(r"(?:^|[/\\])(?:hero|5k_21x9|21x9|closeup|macro_verify)_", re.IGNORECASE), "Post-processed/generated 5K output or crop"),
    (re.compile(r"(?:^|[/\\])(?:raw_batch|raw_scene|nb_raw|nb_dop|nb_subj|s03_crimson)_", re.IGNORECASE), "Generated scene output (re-feeding forbidden)"),
    (re.compile(r"composite_styling_board", re.IGNORECASE), "Derived composite collage board (pass raw item originals instead)"),
    (re.compile(r"(?:^|[/\\])previz_.*\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE), "Synthetic 3D Previz wireframe image (use text pose + 8-dir face instead)"),
    (re.compile(r"[/\\]batches[/\\][^/\\]+[/\\](?:raw_|hero_|closeup_)", re.IGNORECASE), "Batch output artifact"),
    (re.compile(r"[/\\]outputs[/\\](?:21x9_|5k_|hero_|closeup_|nb_|subj)", re.IGNORECASE), "Vault generated scene output"),
)

TURNAROUND_GENERATED_FILENAME_PATTERN = re.compile(
    r"(?:^|[/\\])(?:t2_turn_|crisp_gen1_turn_|subj\d+_turn_|s\d+_8dir_|angle_[mp]\d)", re.IGNORECASE
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_visual_16x16_sig(path: Path, bgr_hint: Optional[object] = None) -> Optional[list[int]]:
    """Compute a compression/re-upload-invariant 16x16 grayscale signature to block re-encoded generated images."""
    try:
        import cv2
        import numpy as np

        if isinstance(bgr_hint, np.ndarray) and bgr_hint.size > 0:
            sub = bgr_hint[::4, ::4] if min(bgr_hint.shape[:2]) >= 128 else bgr_hint
            gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY) if sub.ndim == 3 else sub
            small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
            return [int(v) for v in small.ravel()]

        bgr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if bgr is None or bgr.size == 0:
            return None
        small = cv2.resize(bgr, (16, 16), interpolation=cv2.INTER_AREA)
        return [int(v) for v in small.ravel()]
    except Exception:
        return None


def visual_sig_mad(sig_a: Optional[list[int]], sig_b: Optional[list[int]]) -> float:
    if not sig_a or not sig_b or len(sig_a) != 256 or len(sig_b) != 256:
        return 999.0
    return sum(abs(int(a) - int(b)) for a, b in zip(sig_a, sig_b)) / 256.0


class Mandatory8DirBankVerifier:
    """Hard gate guaranteeing STEP 1 (Complete 8-Direction Turnaround Bank + Gen-0 Origin) is NEVER skipped before STEP 2."""

    @classmethod
    def verify_complete_8dir_bank(
        cls,
        canonical_front_path: Path,
        angle_bank: Optional[Dict[str, Path]],
    ) -> Dict[str, Path]:
        front_p = Path(canonical_front_path)
        if not front_p.is_file() or front_p.stat().st_size < 256:
            raise Incomplete8DirectionBankError(
                f"[8방향 필수 검증기 차단] 기준 원본(Gen-0) 얼굴 파일이 유효하지 않습니다: {front_p}"
            )
        if not angle_bank:
            raise Incomplete8DirectionBankError(
                "[8방향 필수 검증기 차단] 8방향 마스터 뱅크(angle_bank)가 비어 있습니다! "
                "반드시 STEP 1에서 좌·우 8방향(-90°~+90°)을 먼저 완성한 뒤 본 장면 생성을 진행해야 합니다."
            )

        normalized: Dict[str, Path] = {"front": front_p}
        for k, p in angle_bank.items():
            if p is None:
                continue
            canon_k = LEGACY_8DIR_ALIASES.get(k, k)
            cand = Path(p)
            if cand.is_file() and cand.stat().st_size >= 256:
                for pat, reason in FORBIDDEN_DERIVED_FILENAME_PATTERNS:
                    if pat.search(str(cand.resolve())):
                        raise ChainedGenerationViolationError(
                            f"[8방향 필수 검증기 차단] 슬롯 '{canon_k}'에 장면 생성물({cand.name}: {reason})을 넣을 수 없습니다."
                        )
                normalized[canon_k] = cand

        missing = [v for v in REQUIRED_8DIR_VIEWS if v not in normalized]
        if missing:
            raise Incomplete8DirectionBankError(
                f"[8방향 필수 검증기 차단] 8방향 마스터 뱅크 미완성 (누락 슬롯 {len(missing)}개: {', '.join(missing)}). "
                "8방향(-90°, -67.5°, -45°, -22.5°, +22.5°, +45°, +67.5°, +90°) 전 슬롯이 모두 검증되어야만 장면 생성이 허용됩니다."
            )
        return normalized


_LEDGER_MEM_CACHE: Dict[str, Tuple[int, int, Dict[str, object]]] = {}
_SIG16_MEM_CACHE: Dict[str, Optional[list[int]]] = {}


class OriginOnlyOneShotGuard:
    """Cryptographic SHA-256 + Lineage Depth + Single-Shot Guard enforcing Zero Multi-Hop Chaining & Mandatory 8-Dir."""

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
                "verified_8dir_subjects": {},
                "executed_job_ids": [],
            }
            self.ledger_path.write_text(json.dumps(init_data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self) -> Dict[str, object]:
        self._ensure_ledger()
        key = str(self.ledger_path)
        try:
            st = self.ledger_path.stat()
            sig = (int(st.st_mtime_ns), int(st.st_size))
            hit = _LEDGER_MEM_CACHE.get(key)
            if hit is not None and hit[0] == sig[0] and hit[1] == sig[1]:
                return hit[2]
            parsed = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            _LEDGER_MEM_CACHE[key] = (sig[0], sig[1], parsed)
            return parsed
        except Exception:
            return {
                "approved_origins_sha256": {},
                "approved_8dir_faces_sha256": {},
                "forbidden_generated_outputs_sha256": {},
                "verified_8dir_subjects": {},
                "executed_job_ids": [],
            }

    def _save(self, data: Dict[str, object]) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger_path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        try:
            st = self.ledger_path.stat()
            _LEDGER_MEM_CACHE[str(self.ledger_path)] = (int(st.st_mtime_ns), int(st.st_size), data)
        except Exception:
            pass

    def register_approved_origin(self, path: Path, kind: str = "raw_asset", depth: int = 0) -> str:
        """Whitelist a pure Gen-0 raw asset (`depth=0`) or a 1-Hop 8-direction face anchor (`kind='8dir_face'`, `depth=1`)."""
        p = Path(path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Origin file does not exist: {p}")
        digest = sha256_file(p)
        sig16 = compute_visual_16x16_sig(p)
        eff_depth = 1 if (kind == "8dir_face" and depth == 0 and TURNAROUND_GENERATED_FILENAME_PATTERN.search(str(p))) else depth
        with self._lock:
            data = self._load()
            bucket = "approved_8dir_faces_sha256" if kind == "8dir_face" else "approved_origins_sha256"
            rec = dict(data.get(bucket) or {})  # type: ignore[arg-type]
            rec[digest] = {"path": str(p), "kind": kind, "depth": eff_depth, "sig16": sig16, "registered_at": time.time()}
            data[bucket] = rec
            # Ensure a genuine approved Gen-0 resource is never poisoned in forbidden_generated_outputs_sha256
            if kind != "8dir_face":
                forb = dict(data.get("forbidden_generated_outputs_sha256") or {})  # type: ignore[arg-type]
                if digest in forb and any(tag in str(p) for tag in ("/resources/", "/.user_uploaded/", "/originals/")):
                    forb.pop(digest, None)
                    data["forbidden_generated_outputs_sha256"] = forb
            self._save(data)
        return digest

    def register_verified_subject_8dir_bank(self, canonical_front_path: Path, angle_bank: Dict[str, Path]) -> str:
        """Verify all 8 directions + Gen-0 and record cryptographic proof in ledger so OS hooks & composers allow Step 2."""
        norm = Mandatory8DirBankVerifier.verify_complete_8dir_bank(canonical_front_path, angle_bank)
        front_hash = self.register_approved_origin(norm["front"], kind="gen0_face", depth=0)
        slot_hashes: Dict[str, str] = {"front": front_hash}
        for v in REQUIRED_8DIR_VIEWS:
            slot_hashes[v] = self.register_approved_origin(norm[v], kind="8dir_face", depth=1)
        with self._lock:
            data = self._load()
            verified = dict(data.get("verified_8dir_subjects") or {})  # type: ignore[arg-type]
            verified[front_hash] = {
                "front_path": str(Path(canonical_front_path).resolve()),
                "slots": slot_hashes,
                "verified_at": time.time(),
            }
            data["verified_8dir_subjects"] = verified
            self._save(data)
        return front_hash

    def register_generated_scene_output(
        self, output_path: Path, job_id: str = "", bgr_hint: Optional[object] = None
    ) -> Optional[str]:
        """Permanently blacklist a generated scene output SHA-256 + 16x16 visual signature (`depth=terminal`) so it can NEVER be re-fed."""
        p = Path(output_path).resolve()
        if not p.is_file():
            return None
        digest = sha256_file(p)
        sig16 = compute_visual_16x16_sig(p, bgr_hint=bgr_hint)
        with self._lock:
            data = self._load()
            app_o = dict(data.get("approved_origins_sha256") or {})  # type: ignore[arg-type]
            if digest in app_o:
                return None
            forb = dict(data.get("forbidden_generated_outputs_sha256") or {})  # type: ignore[arg-type]
            forb[digest] = {
                "path": str(p),
                "job_id": job_id,
                "depth": "terminal",
                "sig16": sig16,
                "generated_at": time.time(),
            }
            data["forbidden_generated_outputs_sha256"] = forb
            self._save(data)
        return digest

    def _find_visual_match_in_forbidden(
        self,
        cand_path: Path,
        forbidden_hashes: Dict[str, dict],
        approved_orig: Dict[str, dict],
        approved_8dir: Dict[str, dict],
        max_mad: float = 5.2,
    ) -> Optional[str]:
        """Detect re-compressed / renamed / re-uploaded generated images via vectorized 16x16 visual L1 MAD <= 5.2."""
        import numpy as np

        digest = sha256_file(cand_path)
        if digest in approved_orig or digest in approved_8dir:
            return None
        if digest in _SIG16_MEM_CACHE:
            cand_sig = _SIG16_MEM_CACHE[digest]
        else:
            cand_sig = compute_visual_16x16_sig(cand_path)
            _SIG16_MEM_CACHE[digest] = cand_sig
        if not cand_sig or len(cand_sig) != 256:
            return None

        c_vec = np.asarray(cand_sig, dtype=np.int16)
        orig_vecs = [
            np.asarray(v["sig16"], dtype=np.int16)
            for v in approved_orig.values()
            if isinstance(v, dict) and isinstance(v.get("sig16"), list) and len(v["sig16"]) == 256
        ]
        orig_mat = np.stack(orig_vecs, axis=0) if orig_vecs else None

        f_vecs = []
        f_labels = []
        for f_sha, f_info in forbidden_hashes.items():
            if f_sha in approved_orig or f_sha in approved_8dir or not isinstance(f_info, dict):
                continue
            f_sig = f_info.get("sig16")
            if not (isinstance(f_sig, list) and len(f_sig) == 256):
                if f_sha in _SIG16_MEM_CACHE:
                    f_sig = _SIG16_MEM_CACHE[f_sha]
                else:
                    p_str = str(f_info.get("path") or "")
                    # Never block synchronously decoding legacy CloudStorage/GoogleDrive archive files without sig16
                    if p_str and "CloudStorage" not in p_str and "GoogleDrive" not in p_str and Path(p_str).is_file():
                        f_sig = compute_visual_16x16_sig(Path(p_str))
                    else:
                        f_sig = None
                    _SIG16_MEM_CACHE[f_sha] = f_sig
                    if isinstance(f_sig, list) and len(f_sig) == 256:
                        f_info["sig16"] = f_sig
            if isinstance(f_sig, list) and len(f_sig) == 256:
                f_vecs.append(f_sig)
                f_labels.append(str(f_info.get("path") or f_sha[:12]))

        if not f_vecs:
            return None
        f_mat = np.asarray(f_vecs, dtype=np.int16)
        mads = np.mean(np.abs(f_mat - c_vec[None, :]), axis=1)
        hits = np.where(mads <= max_mad)[0]
        if len(hits) == 0:
            return None
        min_orig_mad = float(np.min(np.mean(np.abs(orig_mat - c_vec[None, :]), axis=1))) if orig_mat is not None else 999.0
        for idx_h in hits:
            mad_val = float(mads[int(idx_h)])
            if min_orig_mad <= mad_val + 0.5:
                continue
            return f"{f_labels[int(idx_h)]} (시각 지문 MAD={mad_val:.2f})"
        return None

    def verify_turnaround_1hop_inputs(self, reference_image_paths: Iterable[Path], job_id: Optional[str] = None) -> None:
        """Red-Team Stage-1 Guard: 8-Direction Turnaround generation MUST ONLY accept pure Gen-0 (`depth == 0`) photos!
        Feeding a generated `8dir` image into another `8dir` generation (`Hop >= 2` chaining) is strictly forbidden!
        """
        refs = [Path(p) for p in reference_image_paths]
        if not refs:
            raise OriginLineageViolationError("[8방향 1-Hop 위반] 최소 1장의 순수 Gen-0 원본 인물 사진이 필요합니다.")

        with self._lock:
            data = self._load()
            forbidden_hashes: Dict[str, dict] = dict(data.get("forbidden_generated_outputs_sha256") or {})  # type: ignore[arg-type]
            approved_8dir: Dict[str, dict] = dict(data.get("approved_8dir_faces_sha256") or {})  # type: ignore[arg-type]
            approved_orig: Dict[str, dict] = dict(data.get("approved_origins_sha256") or {})  # type: ignore[arg-type]

            for idx, rp in enumerate(refs, 1):
                if not rp.is_file():
                    raise OriginLineageViolationError(f"[8방향 1-Hop 위반] 입력 이미지 #{idx}가 존재하지 않습니다: {rp}")
                resolved_str = str(rp.resolve())
                digest = sha256_file(rp)

                if digest in forbidden_hashes and digest not in approved_orig:
                    raise ChainedGenerationViolationError(
                        f"[레드팀 차단: 생성물 재먹이기 금지] '{rp.name}'은 이미 생성된 출력물입니다."
                    )
                vis_hit_t1 = self._find_visual_match_in_forbidden(rp, forbidden_hashes, approved_orig, approved_8dir)
                if vis_hit_t1:
                    raise ChainedGenerationViolationError(
                        f"[레드팀 차단: 재압축/재업로드 생성물 다중 재생성 금지] '{rp.name}'이(가) 이전 생성물({vis_hit_t1})과 일치합니다! "
                        "반드시 순수 처음 원본(Gen-0)만 입력해야 합니다."
                    )
                for pat, reason in FORBIDDEN_DERIVED_FILENAME_PATTERNS:
                    if pat.search(resolved_str):
                        raise ChainedGenerationViolationError(
                            f"[레드팀 차단: 파생/생성물 재투입 금지] '{rp.name}' ({reason})"
                        )
                if TURNAROUND_GENERATED_FILENAME_PATTERN.search(resolved_str) or (
                    digest in approved_8dir and int(approved_8dir[digest].get("depth", 0)) >= 1
                ):
                    raise ChainedGenerationViolationError(
                        f"[레드팀 차단: 다단계 체이닝(Hop-Chaining) 절대 금지] '{rp.name}'은 이미 생성된 1차 턴어라운드 이미지입니다! "
                        "생성된 이미지를 가지고 다시 다른 각도를 생성하는 연쇄 생성은 절대 금지되며, 오직 순수 Gen-0 원본만 입력할 수 있습니다."
                    )

        if job_id:
            self.verify_one_shot_and_inputs(refs, job_id=job_id)

    def verify_one_shot_and_inputs(
        self,
        reference_image_paths: Iterable[Path],
        job_id: Optional[str] = None,
    ) -> None:
        """Validate `1원본들 → 1생성`: blocks any scene output re-feeding (`depth == terminal`) and blocks 2nd pass on `job_id`."""
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
                    raise ChainedGenerationViolationError(
                        f"[1원본들 → 1생성 위반] 작업 '{job_id}'에서 2회차 재생성(Multi-Pass / 재먹이기)이 시도되어 강제 차단되었습니다."
                    )

            for idx, rp in enumerate(refs, 1):
                if not rp.is_file():
                    raise OriginLineageViolationError(f"[1원본들 → 1생성 위반] 입력 이미지 #{idx}가 존재하지 않습니다: {rp}")
                resolved_str = str(rp.resolve())
                digest = sha256_file(rp)

                if digest in forbidden_hashes and digest not in approved_8dir and digest not in approved_orig:
                    prev_info = forbidden_hashes[digest]
                    raise ChainedGenerationViolationError(
                        f"[1원본들 → 1생성 위반] 입력 이미지 #{idx} ({rp.name})는 이전에 생성된 결과물({prev_info.get('path')})과 "
                        "동일한 SHA-256 해시를 가집니다. 생성물에 다시 생성하는 재먹이기는 절대 금지됩니다."
                    )

                vis_hit = self._find_visual_match_in_forbidden(rp, forbidden_hashes, approved_orig, approved_8dir)
                if vis_hit:
                    raise ChainedGenerationViolationError(
                        f"[1원본들 → 1생성 위반: 재압축/재업로드 생성물 차단] 입력 이미지 #{idx} ({rp.name})가 "
                        f"이전 생성 결과물({vis_hit})과 시각적으로 일치합니다! 생성된 이미지를 다시 생성기에 넣는 다중 재생성은 절대 금지되며, "
                        "반드시 순수 처음 원본(Gen-0)만 입력해야 합니다."
                    )

                for pat, reason in FORBIDDEN_DERIVED_FILENAME_PATTERNS:
                    if pat.search(resolved_str):
                        raise ChainedGenerationViolationError(
                            f"[1원본들 → 1생성 위반] 입력 이미지 #{idx} ({rp.name})가 금지된 파생/생성 이미지 패턴({reason})에 해당합니다. "
                            "오직 검증된 1-Hop 8방향 얼굴과 순수 원본 에셋 파일만 입력할 수 있습니다."
                        )

            if job_id:
                self._executed_job_ids.add(job_id)
                persisted_jobs.add(job_id)
                data["executed_job_ids"] = sorted(persisted_jobs)[-2000:]
                self._save(data)
