from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from .config import (
    CameraRigSpec,
    CanvasSpec,
    EngineConfig,
    LocationSpec,
    SubjectProfile,
    WardrobeSpec,
)
from .pipeline import ModelGenerationPipeline
from .prompt_builder import PromptFirewall
from .workflow import SinglePassGenerationGuard, UniversalVariationPlanner


def _register_plan_arguments(p_plan: argparse.ArgumentParser) -> None:
    p_plan.add_argument("--ref-face", type=Path, required=True, help="Path to single identity reference photo")
    p_plan.add_argument(
        "--subject-label",
        "--identity-label",
        dest="subject_label",
        type=str,
        default="human subject in the reference portrait",
        help="Universal identity label for the subject",
    )
    p_plan.add_argument("--height-cm", type=float, default=None, help="Optional subject height in cm (applies below neck only)")
    p_plan.add_argument("--weight-kg", type=float, default=None, help="Optional subject weight in kg (applies below neck only)")
    p_plan.add_argument("--hairstyle", type=str, default=None, help="Optional hairstyle override (None = inherit 1:1 from reference)")
    p_plan.add_argument("--eyewear", type=str, default=None, help="Optional eyewear override (None = inherit 1:1 from reference)")
    p_plan.add_argument(
        "--expression",
        type=str,
        default="natural composed expression matching the reference portrait 1:1",
        help="Facial expression descriptor",
    )
    p_plan.add_argument(
        "--facial-trait",
        dest="facial_traits",
        action="append",
        default=[],
        help="Additional facial identity traits (repeatable)",
    )
    p_plan.add_argument(
        "--wardrobe-preset",
        type=str,
        default="tailored_editorial",
        choices=list(UniversalVariationPlanner.WARDROBE_CATALOG.keys()),
    )
    p_plan.add_argument(
        "--location-preset",
        type=str,
        default="studio_editorial",
        choices=list(UniversalVariationPlanner.LOCATION_CATALOG.keys()),
    )
    p_plan.add_argument(
        "--camera-preset",
        type=str,
        default="seated_editorial",
        choices=list(UniversalVariationPlanner.CAMERA_CATALOG.keys()),
    )
    p_plan.add_argument("--custom-outfit", type=str, default=None, help="Custom outfit description override")
    p_plan.add_argument("--custom-accessories", type=str, default=None, help="Custom accessories description")
    p_plan.add_argument("--custom-location", type=str, default=None, help="Custom location description override")
    p_plan.add_argument("--custom-palette-hex", type=str, default=None, help="Custom backdrop/environment hex color")
    p_plan.add_argument("--custom-props", type=str, default=None, help="Custom props and architectural objects")
    p_plan.add_argument("--custom-lighting", type=str, default=None, help="Custom lighting description")
    p_plan.add_argument("--custom-angle", type=str, default=None, help="Custom camera angle override")
    p_plan.add_argument("--custom-shot-scale", type=str, default=None, help="Custom camera shot scale override")
    p_plan.add_argument("--custom-pose", type=str, default=None, help="Custom subject pose description")
    p_plan.add_argument("--focal-length", type=int, default=None, help="Custom lens focal length in mm (e.g. 35, 50, 85, 105)")
    p_plan.add_argument(
        "--negative-space-side",
        type=str,
        default=None,
        help="Custom negative space placement ('left', 'right', 'center', 'top', 'none')",
    )
    p_plan.add_argument("--aspect-ratio-label", type=str, default=None, help="Custom aspect ratio label (e.g. '16:9', '9:16', '21:9')")
    p_plan.add_argument("--max-count", type=int, default=6)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="model-engine",
        description="모델생성 (Universal Model Generation & 5K Super-Resolution Engine v2.0 CLI)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. plan-variations (and 'plan' alias)
    p_plan = sub.add_parser(
        "plan-variations",
        aliases=["plan"],
        help="Plan a universal variation matrix for ANY model, outfit preset/custom, location preset/custom, and camera angle",
    )
    _register_plan_arguments(p_plan)

    # 2. validate-lineage
    p_lin = sub.add_parser("validate-lineage", help="Verify Single-Pass From-Scratch lineage (depth=0, 1 image only)")
    p_lin.add_argument("--ref-images", type=Path, nargs="+", required=True)
    p_lin.add_argument("--depth", type=int, default=0)

    # 3. single-pass-5k
    p_sp = sub.add_parser(
        "single-pass-5k",
        help="Run Single-Pass 5K pipeline (4xNomos8kSC + Adaptive MatteSkinProcessor + Auto-Pad Compositor + 6-Gate Audit)",
    )
    p_sp.add_argument("--scene-image", type=Path, required=True)
    p_sp.add_argument("--output-hero", type=Path, required=True)
    p_sp.add_argument("--output-closeup", type=Path, default=None)
    p_sp.add_argument("--target-width", type=int, default=5040)
    p_sp.add_argument("--target-height", type=int, default=2160)
    p_sp.add_argument("--pad-left-ratio", type=float, default=None)
    p_sp.add_argument("--seed", type=int, default=42)

    # 4. build-prompts
    p_prompt = sub.add_parser("build-prompts", help="Generate validated universal prompts through PromptFirewall")
    p_prompt.add_argument(
        "--subject-label",
        "--identity-label",
        dest="subject_label",
        type=str,
        default="human subject in the reference portrait",
    )
    p_prompt.add_argument("--height-cm", type=float, default=None)
    p_prompt.add_argument("--weight-kg", type=float, default=None)
    p_prompt.add_argument("--hairstyle", type=str, default=None)
    p_prompt.add_argument("--eyewear", type=str, default=None)
    p_prompt.add_argument(
        "--expression",
        type=str,
        default="natural composed expression matching the reference portrait 1:1",
    )
    p_prompt.add_argument("--outfit", type=str, default="contemporary minimalist tailored attire")
    p_prompt.add_argument("--accessories", type=str, default=None)
    p_prompt.add_argument("--scene", type=str, required=True, help="Location/environment description")
    p_prompt.add_argument("--props", type=str, default=None)
    p_prompt.add_argument(
        "--lighting",
        type=str,
        default="soft diffused natural-matte lighting with zero harsh specular glare",
    )
    p_prompt.add_argument("--angle", type=str, default="eye-level camera angle")
    p_prompt.add_argument("--shot-scale", type=str, default="three-quarter portrait shot")
    p_prompt.add_argument("--pose", type=str, default="natural composed editorial posture")
    p_prompt.add_argument("--focal-length", type=int, default=85)
    p_prompt.add_argument("--negative-space-side", type=str, default="left")
    p_prompt.add_argument("--aspect-ratio-label", type=str, default="16:9")

    args = parser.parse_args(argv)

    if args.command in ("plan-variations", "plan"):
        subj = SubjectProfile(
            subject_identity_label=args.subject_label,
            height_cm=args.height_cm,
            weight_kg=args.weight_kg,
            hairstyle=args.hairstyle,
            eyewear=args.eyewear,
            expression=args.expression,
            facial_traits=tuple(args.facial_traits or ()),
        )

        has_custom_wardrobe = bool(args.custom_outfit or args.custom_accessories)
        has_custom_location = bool(
            args.custom_location or args.custom_palette_hex or args.custom_props or args.custom_lighting
        )
        has_custom_camera = bool(
            args.custom_angle
            or args.custom_shot_scale
            or args.custom_pose
            or args.focal_length is not None
            or args.negative_space_side
            or args.aspect_ratio_label
        )

        custom_vars = None
        if has_custom_wardrobe or has_custom_location or has_custom_camera:
            if has_custom_wardrobe:
                w_list = [
                    WardrobeSpec(
                        style_category="custom_wardrobe",
                        outfit_description=args.custom_outfit or "contemporary minimalist tailored attire",
                        accessories=args.custom_accessories,
                    )
                ]
            else:
                w_list = list(UniversalVariationPlanner.WARDROBE_CATALOG[args.wardrobe_preset])

            if has_custom_location:
                l_list = [
                    LocationSpec(
                        environment_category="custom_location",
                        location_description=args.custom_location or "clean minimalist studio backdrop",
                        color_palette_hex=args.custom_palette_hex,
                        props_and_objects=args.custom_props,
                        lighting_description=(
                            args.custom_lighting
                            or "soft diffused natural-matte lighting with zero harsh specular glare"
                        ),
                    )
                ]
            else:
                l_list = list(UniversalVariationPlanner.LOCATION_CATALOG[args.location_preset])

            if has_custom_camera:
                c_list = [
                    CameraRigSpec(
                        shot_scale=args.custom_shot_scale or "three-quarter portrait shot",
                        camera_angle=args.custom_angle or "eye-level camera angle",
                        focal_length_mm=args.focal_length if args.focal_length is not None else 85,
                        pose_description=args.custom_pose or "natural composed editorial posture",
                        negative_space_side=args.negative_space_side or "left",
                        aspect_ratio_label=args.aspect_ratio_label or "16:9",
                    )
                ]
            else:
                c_list = list(UniversalVariationPlanner.CAMERA_CATALOG[args.camera_preset])

            custom_vars = UniversalVariationPlanner.compose_custom_variations(
                w_list, l_list, c_list, max_count=args.max_count
            )

        plans = UniversalVariationPlanner.plan_variation_batch(
            reference_face_path=args.ref_face,
            subject=subj,
            wardrobe_preset=args.wardrobe_preset,
            location_preset=args.location_preset,
            camera_preset=args.camera_preset,
            custom_variations=custom_vars,
            max_count=args.max_count,
        )
        print(json.dumps(plans, ensure_ascii=False, indent=2))
        return 0

    if args.command == "validate-lineage":
        valid_ref = SinglePassGenerationGuard.validate_reference_lineage(
            args.ref_images, generation_depth=args.depth
        )
        print(json.dumps({"valid": True, "reference_image": str(valid_ref), "depth": args.depth}, indent=2))
        return 0

    if args.command == "single-pass-5k":
        canvas_spec = CanvasSpec(
            target_width=args.target_width,
            target_height=args.target_height,
            pad_left_ratio=args.pad_left_ratio,
        )
        cfg = EngineConfig(canvas=canvas_spec)
        pipe = ModelGenerationPipeline(cfg)
        res = pipe.run_single_pass_5k(
            pure_scratch_scene_path=args.scene_image,
            output_hero_path=args.output_hero,
            output_closeup_path=args.output_closeup,
            seed=args.seed,
        )
        res["quality_report"] = asdict(res["quality_report"])
        res["skin_metrics"] = asdict(res["skin_metrics"])
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-prompts":
        subj = SubjectProfile(
            subject_identity_label=args.subject_label,
            height_cm=args.height_cm,
            weight_kg=args.weight_kg,
            hairstyle=args.hairstyle,
            eyewear=args.eyewear,
            expression=args.expression,
        )
        w = WardrobeSpec(outfit_description=args.outfit, accessories=args.accessories)
        loc = LocationSpec(
            location_description=args.scene,
            props_and_objects=args.props,
            lighting_description=args.lighting,
        )
        cam = CameraRigSpec(
            shot_scale=args.shot_scale,
            camera_angle=args.angle,
            focal_length_mm=args.focal_length,
            pose_description=args.pose,
            negative_space_side=args.negative_space_side,
            aspect_ratio_label=args.aspect_ratio_label,
        )
        p = PromptFirewall.build_universal_single_pass_prompt(subj, w, loc, cam)
        print(json.dumps({"universal_single_pass_prompt": p}, ensure_ascii=False, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
