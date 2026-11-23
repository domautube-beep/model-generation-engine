#!/usr/bin/env python3
"""Official CLI for the Mandatory 6-Stage Fresh-Origin Editorial Workflow (`v9.0`).

Usage:
  1) Prepare origin bundle (Stages 1 & 2):
     python3 scripts/run_fresh_origin_workflow.py prepare \
       --gen0 /path/to/gen0.jpg --location /path/to/loc.jpg \
       --work-dir /path/to/stage_dir --scene "..." --outfit "..." --aspect 3:4 --best-of 3

  2) Score & select native winner from K fresh candidates (Stage 4):
     python3 scripts/run_fresh_origin_workflow.py rank \
       --manifest /path/to/stage_dir/workflow_manifest.json \
       --candidates /path/to/c1.jpg /path/to/c2.jpg /path/to/c3.jpg

  3) Finalize native winner via single Upscayl 4x + 10-Gate Audit + 400% Eye Strip (Stage 5):
     python3 scripts/run_fresh_origin_workflow.py finalize \
       --manifest /path/to/stage_dir/workflow_manifest.json \
       --winner /path/to/best_cand.jpg --output-dir /path/to/out --stem hero_v9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_generation_engine.fresh_origin_workflow import FreshOriginEditorialWorkflow


def main() -> int:
    parser = argparse.ArgumentParser(description="Fresh-Origin 6-Stage Editorial Workflow CLI (v9.0)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="Stages 1 & 2: Validate Gen-0, extract 25-metric Blueprint + ocular anchor, clean location, emit K prompts")
    p_prep.add_argument("--gen0", required=True, type=Path)
    p_prep.add_argument("--work-dir", required=True, type=Path)
    p_prep.add_argument("--scene", required=True, type=str)
    p_prep.add_argument("--outfit", required=True, type=str)
    p_prep.add_argument("--location", type=Path, default=None)
    p_prep.add_argument("--turnaround-sheet", type=Path, default=None, help="Verified 1-Hop 4K 8-Direction Master Turnaround Sheet (-90..+90 deg)")
    p_prep.add_argument("--extra-assets", nargs="*", type=Path, default=[])
    p_prep.add_argument("--aspect", default="3:4")
    p_prep.add_argument("--best-of", type=int, default=3)
    p_prep.add_argument("--no-match-yaw", action="store_true")
    p_prep.add_argument("--no-ocular-anchor", action="store_true")

    p_rank = sub.add_parser("rank", help="Stage 4: Score K fresh candidates natively across SFace + 18 Craniofacial + 7 Ocular")
    p_rank.add_argument("--manifest", required=True, type=Path)
    p_rank.add_argument("--candidates", required=True, nargs="+", type=Path)

    p_fin = sub.add_parser("finalize", help="Stage 5: Pure Optical Lanczos-4 (v2.1 default) or Upscayl 4x + P88 MatteSkin + Cohesion + 10-Gate Audit")
    p_fin.add_argument("--manifest", required=True, type=Path)
    p_fin.add_argument("--winner", required=True, type=Path)
    p_fin.add_argument("--output-dir", required=True, type=Path)
    p_fin.add_argument("--stem", required=True, type=str)
    p_fin.add_argument("--upscale-mode", choices=("upscayl_hybrid", "optical_lanczos", "upscayl"), default="upscayl_hybrid")
    p_fin.add_argument("--upscayl-model", default="ultramix-balanced-4x", help="Upscayl model name in Vault models/")
    p_fin.add_argument("--reference-scene", type=Path, default=None)
    p_fin.add_argument("--cohesion-mode", type=str, default="auto")
    p_fin.add_argument("--no-vault-sync", action="store_true")
    p_fin.add_argument("--allow-audit-fail", action="store_true")

    args = parser.parse_args()
    wf = FreshOriginEditorialWorkflow()

    if args.cmd == "prepare":
        res = wf.prepare_origin_bundle(
            gen0_portrait_path=args.gen0,
            work_dir=args.work_dir,
            scene_instruction=args.scene,
            outfit_instruction=args.outfit,
            location_reference_path=args.location,
            turnaround_8dir_sheet_path=args.turnaround_sheet,
            extra_raw_asset_paths=args.extra_assets,
            aspect_ratio_label=args.aspect,
            best_of_k=args.best_of,
            match_gen0_yaw=not args.no_match_yaw,
            attach_ocular_detail_anchor=not args.no_ocular_anchor,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "rank":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        res = wf.score_and_select_native_winner(
            gen0_portrait_path=Path(manifest["gen0_portrait_path"]),
            candidate_paths=args.candidates,
            min_created_epoch=float(manifest.get("created_at", 0.0)),
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "finalize":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        res = wf.finalize_native_winner_4k(
            gen0_portrait_path=Path(manifest["gen0_portrait_path"]),
            winner_candidate_path=args.winner,
            output_dir=args.output_dir,
            output_stem=args.stem,
            min_created_epoch=float(manifest.get("created_at", 0.0)),
            sync_to_vault=not args.no_vault_sync,
            strict_10gate=not args.allow_audit_fail,
            upscale_mode=args.upscale_mode,
            upscayl_model_name=args.upscayl_model,
            reference_scene_path=args.reference_scene,
            cohesion_mode=args.cohesion_mode,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
