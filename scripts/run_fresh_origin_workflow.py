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


def _run_rank_and_finalize_single(
    wf: FreshOriginEditorialWorkflow,
    manifest_path: Path,
    candidate_paths: list[Path],
    output_dir: Path,
    stem: str,
    upscale_mode: str = "tonelocked_upscayl_4k",
    upscayl_model: str = "ultramix-balanced-4x",
    reference_scene: Path | None = None,
    cohesion_mode: str = "off",
    sync_to_vault: bool = True,
    no_film_grain: bool = False,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bank_info = manifest.get("real_multishot_bank") or {}
    fl_meta = manifest.get("facelock_precomposite_meta") or {}
    locked_p = Path(fl_meta["locked_real_shot_path"]) if fl_meta.get("locked_real_shot_path") else None
    pure_gen0 = Path(manifest["gen0_portrait_path"])
    ref_for_angle = (
        Path(manifest["matched_8dir_portrait_path"])
        if manifest.get("matched_8dir_portrait_path")
        else pure_gen0
    )
    ts_meta = manifest.get("turnaround_master_sheet_path")
    ts_path = (
        Path(ts_meta["sheet_path"])
        if isinstance(ts_meta, dict) and ts_meta.get("sheet_path")
        else (Path(ts_meta) if isinstance(ts_meta, str) else None)
    )
    min_ep = float(manifest.get("created_at", 0.0))

    existing_cands = [Path(p) for p in candidate_paths if Path(p).is_file()]
    active_cands = existing_cands if existing_cands else [Path(p) for p in candidate_paths]

    if len(active_cands) > 1:
        rank_res = wf.score_and_select_native_winner(
            gen0_portrait_path=ref_for_angle,
            candidate_paths=active_cands,
            min_created_epoch=min_ep,
            turnaround_sheet_path=ts_path,
        )
        ordered_cands = [Path(c["candidate_path"]) for c in rank_res["all_candidates"]]
    else:
        rank_res = None
        ordered_cands = [Path(active_cands[0])]

    last_err: Exception | None = None
    for cand_p in ordered_cands:
        try:
            rcp = wf.finalize_native_winner_4k(
                gen0_portrait_path=ref_for_angle,
                winner_candidate_path=cand_p,
                output_dir=output_dir,
                output_stem=stem,
                min_created_epoch=min_ep,
                sync_to_vault=sync_to_vault,
                strict_10gate=True,
                upscale_mode=upscale_mode,
                upscayl_model_name=upscayl_model,
                reference_scene_path=reference_scene,
                cohesion_mode=cohesion_mode,
                design_mode=str(manifest.get("design_mode") or "zero_touch_4k"),
                no_film_grain=bool(no_film_grain or manifest.get("no_film_grain", False)),
                real_bank_entries=bank_info.get("entries"),
                locked_real_shot_path=locked_p,
                turnaround_sheet_path=ts_path,
                pure_gen0_portrait_path=pure_gen0,
                framing_mode=str(manifest.get("framing_mode") or "balanced_50_50"),
            )
            rcp["selected_candidate_path"] = str(cand_p)
            if rank_res is not None:
                rcp["rank_summary"] = rank_res
            return rcp
        except Exception as exc:
            last_err = exc
    assert last_err is not None
    raise last_err


def main() -> int:
    parser = argparse.ArgumentParser(description="Fresh-Origin 6-Stage Editorial Workflow CLI (v1.0.6)", allow_abbrev=False)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="Stages 1 & 2: Validate Gen-0 + 8-Dir Bank, extract 25-metric Blueprint, emit K prompts")
    p_prep.add_argument("--gen0", required=True, type=Path)
    p_prep.add_argument("--work-dir", required=True, type=Path)
    p_prep.add_argument("--scene", required=True, type=str)
    p_prep.add_argument("--outfit", required=True, type=str)
    p_prep.add_argument("--location", type=Path, default=None)
    p_prep.add_argument("--clean-location-override", type=Path, default=None, help="Optional crisp empty room clean-plate override")
    p_prep.add_argument("--turnaround-sheet", required=True, type=Path, help="Mandatory Verified 1-Hop 4K 8-Direction Master Turnaround Sheet (-90..+90 deg)")
    p_prep.add_argument("--real-originals", nargs="*", type=Path, default=[], help="Additional untouched real studio/candid photos of the same subject")
    p_prep.add_argument(
        "--design-mode",
        choices=("zero_touch_4k", "pure_v21", "pure_character_sheet", "pure_v22", "design_a_relight", "design_b_facelock_outpaint", "design_b_rerender"),
        default="zero_touch_4k",
    )
    p_prep.add_argument("--extra-assets", nargs="*", type=Path, default=[])
    p_prep.add_argument("--aspect", default="3:4")
    p_prep.add_argument("--best-of", type=int, default=5, help="Default 5-Lens (50/65/85/105/135mm) x 5-Distinct-Lighting Matrix")
    p_prep.add_argument("--no-match-yaw", action="store_true")
    p_prep.add_argument("--no-ocular-anchor", action="store_true")
    p_prep.add_argument("--text-only-location", action="store_true", help="Analyze --location mood/lighting for text prompt ONLY without attaching any location image")
    p_prep.add_argument("--no-film-grain", action="store_true", help="Clean base-ISO commercial studio mode with ZERO film grain in prompt and 4K mastering")
    p_prep.add_argument("--focal-mm", type=int, default=None, help="Optional fixed lens focal length in mm across all K cuts (e.g. 90)")
    p_prep.add_argument(
        "--framing-mode",
        choices=("auto", "balanced_50_50", "closeup"),
        default="auto",
        help="Default balanced 50:50 subject-to-space ratio + shallow optical bokeh; switches to closeup only when requested",
    )

    p_rank = sub.add_parser("rank", help="Stage 4: Score K fresh candidates natively across SFace + 18 Craniofacial + 7 Ocular")
    p_rank.add_argument("--manifest", required=True, type=Path)
    p_rank.add_argument("--candidates", required=True, nargs="+", type=Path)

    p_fin = sub.add_parser("finalize", help="Stage 5: Tone-Locked True 4K Super-Resolution (v1.0.6+ default) + Mandatory 10-Gate Lock")
    p_fin.add_argument("--manifest", required=True, type=Path)
    p_fin.add_argument("--winner", required=True, type=Path)
    p_fin.add_argument("--output-dir", required=True, type=Path)
    p_fin.add_argument("--stem", required=True, type=str)
    p_fin.add_argument(
        "--upscale-mode",
        choices=("tonelocked_upscayl_4k", "optical_fullframe"),
        default="tonelocked_upscayl_4k",
    )
    p_fin.add_argument("--upscayl-model", default="ultramix-balanced-4x", help="Upscayl model name in Vault models/")
    p_fin.add_argument("--reference-scene", type=Path, default=None)
    p_fin.add_argument("--cohesion-mode", type=str, default="off")
    p_fin.add_argument("--no-vault-sync", action="store_true")
    p_fin.add_argument("--allow-audit-fail", action="store_true", help=argparse.SUPPRESS)
    p_fin.add_argument("--no-film-grain", action="store_true", help="Disable all synthetic film grain during 4K mastering")

    p_rfin = sub.add_parser(
        "rank-and-finalize",
        help="Stages 4 & 5 in 1 warm process: rank K fresh candidates and finalize the top passing candidate in True 4K",
    )
    p_rfin.add_argument("--manifest", required=True, type=Path)
    p_rfin.add_argument("--candidates", required=True, nargs="+", type=Path)
    p_rfin.add_argument("--output-dir", required=True, type=Path)
    p_rfin.add_argument("--stem", required=True, type=str)
    p_rfin.add_argument("--upscale-mode", choices=("tonelocked_upscayl_4k", "optical_fullframe"), default="tonelocked_upscayl_4k")
    p_rfin.add_argument("--upscayl-model", default="ultramix-balanced-4x")
    p_rfin.add_argument("--reference-scene", type=Path, default=None)
    p_rfin.add_argument("--cohesion-mode", type=str, default="off")
    p_rfin.add_argument("--no-vault-sync", action="store_true")
    p_rfin.add_argument("--allow-audit-fail", action="store_true", help=argparse.SUPPRESS)
    p_rfin.add_argument("--no-film-grain", action="store_true")

    p_pbatch = sub.add_parser(
        "prepare-batch",
        help="Prepare N scene/outfit manifests for 1..M subjects in 1 warm Python process (<0.2s total) and optionally emit Flow & Finalize batch specs",
    )
    p_pbatch.add_argument("--spec-json", required=True, type=Path, help="JSON spec containing gen0, turnaround_sheet, and cuts[]")
    p_pbatch.add_argument("--emit-flow-batch", type=Path, default=None, help="Auto-write ready-to-run Google Flow x4 Native-4K batch JSON")
    p_pbatch.add_argument("--emit-finalize-batch", type=Path, default=None, help="Auto-write ready-to-run finalize-batch JSON")

    p_fbatch = sub.add_parser(
        "finalize-batch",
        help="Rank and finalize N cuts in 1 warm Python process with shared ONNX & Vision caches",
    )
    p_fbatch.add_argument("--spec-json", required=True, type=Path, help="JSON spec containing jobs[{manifest, candidates/winner, output_dir, stem}]")
    p_fbatch.add_argument("--allow-audit-fail", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()
    if getattr(args, "allow_audit_fail", False):
        parser.error(
            "[Anti-Bypass Lock v1.0.6] --allow-audit-fail is permanently banned! "
            "Every 4K output MUST pass all 10 quality gates."
        )
    wf = FreshOriginEditorialWorkflow()

    if args.cmd == "prepare":
        res = wf.prepare_origin_bundle(
            gen0_portrait_path=args.gen0,
            work_dir=args.work_dir,
            scene_instruction=args.scene,
            outfit_instruction=args.outfit,
            location_reference_path=args.location,
            clean_location_override_path=args.clean_location_override,
            turnaround_8dir_sheet_path=args.turnaround_sheet,
            extra_raw_asset_paths=args.extra_assets,
            real_original_paths=args.real_originals,
            design_mode=args.design_mode,
            aspect_ratio_label=args.aspect,
            best_of_k=args.best_of,
            match_gen0_yaw=not args.no_match_yaw,
            attach_ocular_detail_anchor=not args.no_ocular_anchor,
            text_only_location=args.text_only_location,
            no_film_grain=args.no_film_grain,
            fixed_lens_mm=args.focal_mm,
            framing_mode=args.framing_mode,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "rank":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        ref_for_angle = (
            Path(manifest["matched_8dir_portrait_path"])
            if manifest.get("matched_8dir_portrait_path")
            else Path(manifest["gen0_portrait_path"])
        )
        ts_meta = manifest.get("turnaround_master_sheet_path")
        ts_path = Path(ts_meta["sheet_path"]) if isinstance(ts_meta, dict) and ts_meta.get("sheet_path") else (Path(ts_meta) if isinstance(ts_meta, str) else None)
        res = wf.score_and_select_native_winner(
            gen0_portrait_path=ref_for_angle,
            candidate_paths=args.candidates,
            min_created_epoch=float(manifest.get("created_at", 0.0)),
            turnaround_sheet_path=ts_path,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "finalize":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        bank_info = manifest.get("real_multishot_bank") or {}
        fl_meta = manifest.get("facelock_precomposite_meta") or {}
        locked_p = Path(fl_meta["locked_real_shot_path"]) if fl_meta.get("locked_real_shot_path") else None
        pure_gen0 = Path(manifest["gen0_portrait_path"])
        ref_for_angle = (
            Path(manifest["matched_8dir_portrait_path"])
            if manifest.get("matched_8dir_portrait_path")
            else pure_gen0
        )
        ts_meta = manifest.get("turnaround_master_sheet_path")
        ts_path = Path(ts_meta["sheet_path"]) if isinstance(ts_meta, dict) and ts_meta.get("sheet_path") else (Path(ts_meta) if isinstance(ts_meta, str) else None)
        res = wf.finalize_native_winner_4k(
            gen0_portrait_path=ref_for_angle,
            winner_candidate_path=args.winner,
            output_dir=args.output_dir,
            output_stem=args.stem,
            min_created_epoch=float(manifest.get("created_at", 0.0)),
            sync_to_vault=not args.no_vault_sync,
            strict_10gate=True,
            upscale_mode=args.upscale_mode,
            upscayl_model_name=args.upscayl_model,
            reference_scene_path=args.reference_scene,
            cohesion_mode=args.cohesion_mode,
            design_mode=str(manifest.get("design_mode") or "zero_touch_4k"),
            no_film_grain=bool(args.no_film_grain or manifest.get("no_film_grain", False)),
            real_bank_entries=bank_info.get("entries"),
            locked_real_shot_path=locked_p,
            turnaround_sheet_path=ts_path,
            pure_gen0_portrait_path=pure_gen0,
            framing_mode=str(manifest.get("framing_mode") or "balanced_50_50"),
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "rank-and-finalize":
        res = _run_rank_and_finalize_single(
            wf=wf,
            manifest_path=args.manifest,
            candidate_paths=list(args.candidates),
            output_dir=args.output_dir,
            stem=args.stem,
            upscale_mode=args.upscale_mode,
            upscayl_model=args.upscayl_model,
            reference_scene=args.reference_scene,
            cohesion_mode=args.cohesion_mode,
            sync_to_vault=not args.no_vault_sync,
            no_film_grain=args.no_film_grain,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "prepare-batch":
        spec_raw = json.loads(args.spec_json.read_text(encoding="utf-8"))
        defaults = spec_raw if isinstance(spec_raw, dict) else {}
        cuts = spec_raw.get("cuts", []) if isinstance(spec_raw, dict) else spec_raw
        if not cuts:
            raise ValueError("prepare-batch spec-json has no 'cuts' entries.")
        manifests_out = []
        flow_jobs_out = []
        fin_jobs_out = []
        for idx, cut in enumerate(cuts, start=1):
            gen0_c = Path(cut.get("gen0") or defaults["gen0"])
            ts_c = Path(cut.get("turnaround_sheet") or defaults["turnaround_sheet"])
            wdir_c = Path(cut["work_dir"])
            loc_c = Path(cut["location"]) if cut.get("location") else (Path(defaults["location"]) if defaults.get("location") else None)
            cl_ov = Path(cut["clean_location_override"]) if cut.get("clean_location_override") else None
            extra_c = [Path(x) for x in (cut.get("extra_assets") or defaults.get("extra_assets") or [])]
            real_c = [Path(x) for x in (cut.get("real_originals") or defaults.get("real_originals") or [])]
            asp_c = str(cut.get("aspect") or defaults.get("aspect") or "3:4")
            mf = wf.prepare_origin_bundle(
                gen0_portrait_path=gen0_c,
                work_dir=wdir_c,
                scene_instruction=str(cut["scene"]),
                outfit_instruction=str(cut["outfit"]),
                location_reference_path=loc_c,
                clean_location_override_path=cl_ov,
                turnaround_8dir_sheet_path=ts_c,
                extra_raw_asset_paths=extra_c,
                real_original_paths=real_c,
                design_mode=str(cut.get("design_mode") or defaults.get("design_mode") or "zero_touch_4k"),
                aspect_ratio_label=asp_c,
                best_of_k=int(cut.get("best_of") or defaults.get("best_of") or 5),
                match_gen0_yaw=bool(cut.get("match_gen0_yaw", defaults.get("match_gen0_yaw", True))),
                attach_ocular_detail_anchor=bool(cut.get("attach_ocular_detail_anchor", defaults.get("attach_ocular_detail_anchor", True))),
                text_only_location=bool(cut.get("text_only_location", defaults.get("text_only_location", False))),
                no_film_grain=bool(cut.get("no_film_grain", defaults.get("no_film_grain", False))),
                fixed_lens_mm=cut.get("focal_mm", defaults.get("focal_mm")),
                framing_mode=str(cut.get("framing_mode") or defaults.get("framing_mode") or "auto"),
            )
            manifests_out.append(
                {
                    "index": idx,
                    "work_dir": str(wdir_c),
                    "manifest_path": mf["manifest_path"],
                    "framing_mode": mf["framing_mode"],
                    "prompt_count": len(mf["prompts"]),
                }
            )
            stem_c = str(cut.get("stem") or f"cut{idx:02d}")
            raw_out_c = Path(cut.get("raw_output") or (wdir_c / f"{stem_c}_raw.png"))
            flow_cnt = int(cut.get("flow_count") or defaults.get("flow_count") or 4)
            flow_refs: list[str] = []
            for cand_ref in (
                [mf.get("gen0_portrait_path"), mf.get("matched_8dir_portrait_path")]
                + [r.get("path") for r in mf.get("ordered_reference_images", [])]
            ):
                if not cand_ref:
                    continue
                cand_ref_s = str(cand_ref)
                if "master_8angle_turnaround_sheet" in Path(cand_ref_s).name.lower():
                    continue
                if cand_ref_s not in flow_refs and Path(cand_ref_s).is_file():
                    flow_refs.append(cand_ref_s)
            flow_prompt = str(cut.get("flow_prompt_override") or mf["prompts"][0]["prompt"])
            flow_jobs_out.append(
                {
                    "tag": stem_c,
                    "images": flow_refs,
                    "prompt": flow_prompt,
                    "output": str(raw_out_c),
                    "model": str(cut.get("flow_model") or defaults.get("flow_model") or "Nano Banana Pro"),
                    "aspect": asp_c,
                    "tier": "4K",
                    "count": flow_cnt,
                    "manifest": mf["manifest_path"],
                    "max_4k_downloads": int(cut.get("max_4k_downloads") or defaults.get("max_4k_downloads") or 2),
                }
            )
            fin_out_dir = str(cut.get("output_dir") or defaults.get("output_dir") or (wdir_c.parent / "final_4k"))
            cand_list = (
                [str(raw_out_c.with_name(f"{raw_out_c.stem}_c{ci}{raw_out_c.suffix}")) for ci in range(1, flow_cnt + 1)]
                if flow_cnt > 1
                else [str(raw_out_c)]
            )
            fin_jobs_out.append(
                {
                    "manifest": mf["manifest_path"],
                    "candidates": cand_list,
                    "output_dir": fin_out_dir,
                    "stem": stem_c,
                }
            )

        if args.emit_flow_batch is not None:
            args.emit_flow_batch.parent.mkdir(parents=True, exist_ok=True)
            args.emit_flow_batch.write_text(json.dumps(flow_jobs_out, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.emit_finalize_batch is not None:
            args.emit_finalize_batch.parent.mkdir(parents=True, exist_ok=True)
            fin_spec_obj = {
                "upscale_mode": "tonelocked_upscayl_4k",
                "cohesion_mode": "off",
                "sync_to_vault": bool(defaults.get("sync_to_vault", True)),
                "no_film_grain": bool(defaults.get("no_film_grain", True)),
                "jobs": fin_jobs_out,
            }
            args.emit_finalize_batch.write_text(json.dumps(fin_spec_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": "ok",
                    "count": len(manifests_out),
                    "manifests": manifests_out,
                    "emitted_flow_batch": str(args.emit_flow_batch) if args.emit_flow_batch else None,
                    "emitted_finalize_batch": str(args.emit_finalize_batch) if args.emit_finalize_batch else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.cmd == "finalize-batch":
        spec_raw = json.loads(args.spec_json.read_text(encoding="utf-8"))
        defaults = spec_raw if isinstance(spec_raw, dict) else {}
        jobs = spec_raw.get("jobs", []) if isinstance(spec_raw, dict) else spec_raw
        if not jobs:
            raise ValueError("finalize-batch spec-json has no 'jobs' entries.")
        passed_jobs = []
        failed_jobs = []
        for jb in jobs:
            stem_j = str(jb["stem"])
            out_d_j = Path(jb["output_dir"])
            mf_j = Path(jb["manifest"])
            cands_j = [Path(x) for x in jb["candidates"]] if jb.get("candidates") else [Path(jb["winner"])]
            try:
                rcp_j = _run_rank_and_finalize_single(
                    wf=wf,
                    manifest_path=mf_j,
                    candidate_paths=cands_j,
                    output_dir=out_d_j,
                    stem=stem_j,
                    upscale_mode=str(jb.get("upscale_mode") or defaults.get("upscale_mode") or "tonelocked_upscayl_4k"),
                    upscayl_model=str(jb.get("upscayl_model") or defaults.get("upscayl_model") or "ultramix-balanced-4x"),
                    reference_scene=Path(jb["reference_scene"]) if jb.get("reference_scene") else None,
                    cohesion_mode=str(jb.get("cohesion_mode") or defaults.get("cohesion_mode") or "off"),
                    sync_to_vault=bool(jb.get("sync_to_vault", defaults.get("sync_to_vault", True))),
                    no_film_grain=bool(jb.get("no_film_grain", defaults.get("no_film_grain", False))),
                )
                passed_jobs.append(
                    {
                        "stem": stem_j,
                        "status": "passed_10gates",
                        "hero_4k": rcp_j["deliverables"]["hero_4k"],
                        "receipt_path": rcp_j["receipt_path"],
                        "sface_similarity": rcp_j["sface_similarity"],
                        "face_laplacian_variance": rcp_j["face_laplacian_variance"],
                    }
                )
            except Exception as exc:
                failed_jobs.append({"stem": stem_j, "status": "failed_10gates", "error": str(exc)})
        summary = {
            "status": "ok" if passed_jobs else "all_failed",
            "passed_count": len(passed_jobs),
            "failed_count": len(failed_jobs),
            "passed": passed_jobs,
            "failed": failed_jobs,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if passed_jobs else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
