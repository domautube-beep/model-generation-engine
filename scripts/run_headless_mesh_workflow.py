#!/usr/bin/env python3
"""CLI for the Commercial-Grade Headless 8.2-Head Body + 1:1 Mesh-Locked Head Workflow (`v13.2`)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_generation_engine.headless_mesh_composer import HeadlessMeshComposer


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless Body + 1:1 Mesh-Locked Head Composer CLI (v13.2)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="Stage 1+2A: Erase placeholder head -> isotropic 8.2-head Gen-0 mesh pre-composite")
    p_prep.add_argument("--gen0", required=True, type=Path)
    p_prep.add_argument("--body-plate", required=True, type=Path)
    p_prep.add_argument("--work-dir", required=True, type=Path)
    p_prep.add_argument("--head-ratio", type=float, default=0.325, help="Target faceToShoulderRatio calibrated to apple_vision.py (0.325 = true 8.2-head)")
    p_prep.add_argument("--upscale-body-4k", action="store_true", help="Upscale sub-2K uploaded scene/body plate to 4K before head erasure and mesh placement")

    p_fin = sub.add_parser("finalize", help="Stage 2C: RBF Outer-Rim Snap + Skin-Masked Relighting + 4-Level Pyramid + Full 10-Gate & 400% Forensic Strips")
    p_fin.add_argument("--gen0", required=True, type=Path)
    p_fin.add_argument("--harmonized", required=True, type=Path)
    p_fin.add_argument("--meta", required=True, type=Path)
    p_fin.add_argument("--output-dir", required=True, type=Path)
    p_fin.add_argument("--stem", required=True, type=str)
    p_fin.add_argument("--no-vault-sync", action="store_true")

    args = parser.parse_args()
    comp = HeadlessMeshComposer()

    if args.cmd == "prepare":
        wdir = args.work_dir.resolve()
        wdir.mkdir(parents=True, exist_ok=True)
        res = comp.prepare_stage2a_precomposite(
            gen0_path=args.gen0,
            body_plate_path=args.body_plate,
            output_headless_path=wdir / "stage1_headless_plate_4k.png",
            output_precomp_path=wdir / "stage2a_mesh_precomposite_4k.png",
            output_meta_path=wdir / "stage2a_meta.json",
            target_head_to_shoulder_ratio=args.head_ratio,
            upscale_body_to_4k=args.upscale_body_4k,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "finalize":
        res = comp.finalize_and_audit_stage2c(
            gen0_path=args.gen0,
            stage2b_harmonized_path=args.harmonized,
            stage2a_meta_path=args.meta,
            output_dir=args.output_dir,
            output_stem=args.stem,
            sync_to_vault=not args.no_vault_sync,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
