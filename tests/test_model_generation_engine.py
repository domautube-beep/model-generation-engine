"""Comprehensive Unit & Integration Test Suite for Universal Model Generation Engine (모델생성 v2.0)."""

from __future__ import annotations

from pathlib import Path
import unittest

import cv2
import numpy as np

from model_generation_engine import (
    AdaptiveSkinToneProfile,
    CameraRigSpec,
    CanvasSpec,
    DynamicFaceDetector,
    EngineConfig,
    GenerationalDegradationError,
    LocationSpec,
    MatteSkinProcessor,
    ModelGenerationPipeline,
    PromptContaminationError,
    PromptFirewall,
    QualityGateAuditor,
    ReferenceIsolationError,
    Seamless5KCompositor,
    SinglePassGenerationGuard,
    SubjectProfile,
    SuperResolutionEngine,
    UniversalVariationPlanner,
    VariationSpec,
    WardrobeSpec,
)


class TestUniversalModelGenerationEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = EngineConfig()
        cls.detector = DynamicFaceDetector(cls.config.resolve_model_path("yunet.onnx"))
        cls.ref_face = cls.config.vault_dir / "resources" / "ref_person_face.jpg"
        cls.outputs_dir = cls.config.vault_dir / "outputs"

    def test_01_universal_subject_profiles_zero_hardcoding(self) -> None:
        """Verify SubjectProfile supports any model without hardcoded gender, glasses, or height/weight."""
        # 1. Default completely neutral profile
        default_subj = SubjectProfile()
        self.assertIsNone(default_subj.height_cm)
        self.assertIsNone(default_subj.weight_kg)
        self.assertIsNone(default_subj.eyewear)
        self.assertIsNone(default_subj.hairstyle)
        self.assertIn(
            "preserving the exact eyewear presence or absence",
            default_subj.facial_identity_descriptor(),
        )
        self.assertIn(
            "preserving the exact hairstyle",
            default_subj.facial_identity_descriptor(),
        )

        wardrobe = WardrobeSpec(
            style_category="summer_linen",
            outfit_description="ivory unlined linen blazer over a soft silk camisole top and high-waisted wide-leg linen trousers",
            accessories="minimalist silver cuff",
        )
        location = LocationSpec(
            environment_category="outdoor_courtyard",
            location_description="sun-dappled travertine courtyard wall with honed stone pavers",
            props_and_objects="minimalist limestone bench and terra-cotta planter with olive branch",
            lighting_description="golden hour diffused directional sunlight",
        )
        camera = CameraRigSpec(
            shot_scale="medium three-quarter standing portrait",
            camera_angle="subtle low-angle heroic perspective looking slightly upward",
            focal_length_mm=85,
            pose_description="standing with relaxed posture, head turned 20 degrees toward camera",
            negative_space_side="right",
        )

        # 2. Female model without glasses
        female_subj = SubjectProfile(
            subject_identity_label="Female model in her late 20s",
            height_cm=168,
            weight_kg=50,
            hairstyle="long wavy dark brown hair",
            eyewear="no glasses",
            expression="serene confident editorial gaze",
        )
        self.assertIn("slender", female_subj.body_frame_descriptor())
        prompt_female = PromptFirewall.build_universal_single_pass_prompt(
            subject=female_subj,
            wardrobe=wardrobe,
            location=location,
            camera=camera,
        )
        PromptFirewall.validate_prompt(prompt_female)
        self.assertIn("Female model in her late 20s", prompt_female)
        self.assertIn("168cm, 50kg", prompt_female)
        self.assertIn("wearing no glasses (bare eyes)", prompt_female)
        self.assertNotIn("browline glasses", prompt_female.lower())
        self.assertNotIn("korean male", prompt_female.lower())

        # 3. Male model with glasses and sturdy build
        male_subj = SubjectProfile(
            subject_identity_label="East Asian man in his 30s",
            height_cm=175,
            weight_kg=80,
            hairstyle="short dark hair with neat side part",
            eyewear="black-and-gold browline glasses",
        )
        self.assertIn("sturdy", male_subj.body_frame_descriptor())
        prompt_male = PromptFirewall.build_universal_single_pass_prompt(
            subject=male_subj,
            wardrobe=wardrobe,
            location=location,
            camera=camera,
        )
        PromptFirewall.validate_prompt(prompt_male)
        self.assertIn("175cm, 80kg", prompt_male)
        self.assertIn("black-and-gold browline glasses", prompt_male)

    def test_02_universal_variation_planner_across_4_axes(self) -> None:
        """Verify UniversalVariationPlanner plans across any wardrobe, location, and camera angle category."""
        subj = SubjectProfile(subject_identity_label="Universal editorial model")

        for w_cat, l_cat, c_cat in [
            ("tailored_editorial", "studio_editorial", "seated_editorial"),
            ("smart_casual", "architectural_interior", "multi_angle_portfolio"),
            ("architectural_outerwear", "outdoor_natural", "multi_angle_portfolio"),
            ("tailored_editorial", "urban_cinematic", "multi_angle_portfolio"),
        ]:
            plans = UniversalVariationPlanner.plan_variation_batch(
                reference_face_path=self.ref_face,
                subject=subj,
                wardrobe_preset=w_cat,
                location_preset=l_cat,
                camera_preset=c_cat,
                max_count=3,
            )
            self.assertEqual(len(plans), 3)
            for p in plans:
                PromptFirewall.validate_prompt(p["prompt"])
                self.assertEqual(p["reference_images"], [str(self.ref_face)])

        # Test custom user-supplied WardrobeSpec, LocationSpec, CameraRigSpec lists
        custom_vars = UniversalVariationPlanner.compose_custom_variations(
            wardrobes=[
                WardrobeSpec(
                    style_category="custom_avant_garde",
                    outfit_description="structured slate-grey gabardine trench coat over black mock-neck knit and pleated charcoal wool trousers",
                )
            ],
            locations=[
                LocationSpec(
                    environment_category="rooftop_terrace",
                    location_description="panoramic minimalist concrete rooftop skyline at dusk",
                    props_and_objects="architectural glass-and-steel parapet and minimalist concrete bollard",
                    lighting_description="cool twilight ambient sky with warm key light",
                )
            ],
            cameras=[
                CameraRigSpec(
                    shot_scale="full-body wide editorial shot",
                    camera_angle="dynamic 45-degree side-profile angle",
                    focal_length_mm=35,
                    pose_description="walking confidently along the terrace edge",
                    negative_space_side="right",
                )
            ],
            max_count=1,
        )
        custom_plans = UniversalVariationPlanner.plan_variation_batch(
            reference_face_path=self.ref_face,
            subject=subj,
            custom_variations=custom_vars,
            max_count=1,
        )
        self.assertEqual(len(custom_plans), 1)
        self.assertIn("slate-grey gabardine trench coat", custom_plans[0]["prompt"])
        self.assertIn("rooftop skyline at dusk", custom_plans[0]["prompt"])
        self.assertIn("35mm", custom_plans[0]["prompt"])

    def test_03_prompt_firewall_blocks_forbidden_patterns(self) -> None:
        """Verify PromptFirewall rejects face-bloating, skin-blur, plastic-gloss, and forced-wrinkle prompts."""
        valid_prompt = PromptFirewall.build_universal_single_pass_prompt(
            subject=SubjectProfile(),
            wardrobe=UniversalVariationPlanner.WARDROBE_CATALOG["tailored_editorial"][0],
            location=UniversalVariationPlanner.LOCATION_CATALOG["studio_editorial"][0],
            camera=UniversalVariationPlanner.CAMERA_CATALOG["seated_editorial"][0],
        )
        PromptFirewall.validate_prompt(valid_prompt)

        for bad_snippet in [
            "chubby cheeks and round face",
            "airbrushed skin with beauty-cam filter",
            "dewy skin with glossy skin finish",
            "deep wrinkles and heavy creases",
        ]:
            with self.assertRaises(PromptContaminationError):
                PromptFirewall.validate_prompt(valid_prompt + " " + bad_snippet)

    def test_04_reference_isolation_and_single_pass_guard(self) -> None:
        """Verify SinglePassGenerationGuard blocks re-feeding and multi-person reference mixing."""
        validated = SinglePassGenerationGuard.validate_reference_lineage(
            [self.ref_face], generation_depth=0
        )
        self.assertEqual(validated, self.ref_face)

        with self.assertRaises(GenerationalDegradationError):
            SinglePassGenerationGuard.validate_reference_lineage(
                [self.ref_face], generation_depth=1
            )

        with self.assertRaises(GenerationalDegradationError):
            SinglePassGenerationGuard.validate_reference_lineage(
                [self.outputs_dir / "21x9_studio_seated_01_terracotta_cane.jpg"],
                generation_depth=0,
            )

        mood_ref = self.config.vault_dir / "resources" / "ref_studio_seated.jpg"
        with self.assertRaises(ReferenceIsolationError):
            SinglePassGenerationGuard.validate_reference_lineage(
                [self.ref_face, mood_ref],
                generation_depth=0,
            )
        with self.assertRaises(ReferenceIsolationError):
            SinglePassGenerationGuard.validate_reference_lineage(
                [mood_ref],
                generation_depth=0,
            )

    def test_05_adaptive_skin_tone_profile_and_matte_skin_processor(self) -> None:
        """Verify AdaptiveSkinToneProfile adapts dynamically to fair, medium, and deep skin tones without hardcoding."""
        img = cv2.imread(str(self.ref_face), cv2.IMREAD_COLOR)
        self.assertIsNotNone(img)
        det = self.detector.detect(img)

        # 1. Sample actual skin profile from reference portrait
        profile_real = self.detector.sample_adaptive_skin_profile(img, det)
        self.assertGreater(profile_real.l_median, 80.0)

        # 2. Simulate a darker/deeper skin tone by shifting L downward in LAB space
        lab_dark = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_dark[:, :, 0] = np.clip(lab_dark[:, :, 0] * 0.55, 25, 255)
        img_dark = cv2.cvtColor(lab_dark.astype(np.uint8), cv2.COLOR_LAB2BGR)
        profile_dark = self.detector.sample_adaptive_skin_profile(img_dark, det)
        self.assertLess(profile_dark.l_median, profile_real.l_median * 0.70)

        # Verify MatteSkinProcessor processes both fair and dark skin tones cleanly
        skin_proc = MatteSkinProcessor(self.detector)
        out_real, metrics_real = skin_proc.process(img)
        _, core_dark = skin_proc.build_dynamic_skin_mask(img_dark, det)
        self.assertEqual(out_real.shape, img.shape)
        self.assertGreater(metrics_real.skin_pixel_count, 500)
        self.assertGreater(int(np.sum(core_dark)), 500)

    def test_06_auto_balanced_5k_compositor_dynamic_padding(self) -> None:
        """Verify Seamless5KCompositor auto-detects subject position when pad_left_ratio=None."""
        img = cv2.imread(str(self.ref_face), cv2.IMREAD_COLOR)
        compositor = Seamless5KCompositor(self.config, self.detector)
        canvas, pad_left, new_w = compositor.build_ultrawide_canvas(img)
        self.assertEqual(canvas.shape, (2160, 5040, 3))
        self.assertGreaterEqual(pad_left, 0)
        self.assertGreater(new_w, 1000)

    def test_07_similarity_alignment_validation(self) -> None:
        """Verify SuperResolutionEngine validates similarity transforms and rejects skew/shear/reflection."""
        src = np.array(
            [[100.0, 100.0], [160.0, 100.0], [130.0, 135.0], [110.0, 165.0], [150.0, 165.0]],
            dtype=np.float32,
        )
        dst = src * 1.02 + np.array([4.0, -3.0], dtype=np.float32)
        m, _ = cv2.estimateAffinePartial2D(src, dst)
        SuperResolutionEngine.validate_similarity_transform(m)

        # Skewed matrix must raise RuntimeError
        skew_m = np.array([[1.0, 0.4, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        with self.assertRaises(RuntimeError):
            SuperResolutionEngine.validate_similarity_transform(skew_m)

    def test_08_all_11_deliverables_pass_6_quality_gates(self) -> None:
        """Verify all 11 5K 21:9 deliverables in Google Drive Vault pass all 6 quality gates."""
        auditor = QualityGateAuditor(self.config, self.detector)
        expected_files = [
            "21x9_studio_seated_01_terracotta_cane.jpg",
            "21x9_studio_seated_02_olive_bauhaus.jpg",
            "21x9_studio_seated_03_taupe_camel.jpg",
            "21x9_studio_seated_04_navy_waistcoat.jpg",
            "21x9_studio_seated_05_burgundy_suede.jpg",
            "21x9_studio_seated_06_sand_suit.jpg",
            "21x9_custom_01_vermilion_red_touch_frame.jpg",
            "21x9_custom_02_vermilion_red_front_stride.jpg",
            "21x9_custom_03_navy_wall_orange_door.jpg",
            "21x9_custom_04_green_door_orange_glow.jpg",
            "21x9_custom_05_periwinkle_cutout.jpg",
        ]
        for fname in expected_files:
            fpath = self.outputs_dir / fname
            self.assertTrue(fpath.exists(), f"Missing 5K deliverable: {fpath}")
            img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
            face = self.detector.detect(img)
            crop_box = self.detector.compute_upper_body_crop_box(img.shape[:2], face)
            report = auditor.audit(
                canvas_5k_bgr=img,
                crop_box_5k=crop_box,
                seam_l1_diff=1.2,
            )
            self.assertTrue(
                report.passed,
                f"Deliverable {fname} failed quality gates: {report.message}",
            )

    def test_09_adversarial_skin_tone_spectrum_ebony_olive_pale(self) -> None:
        """Red Team Audit Axis 3: Verify deep ebony (L~18..42), olive/tan, and pale skin tones are never clipped."""
        img = cv2.imread(str(self.ref_face), cv2.IMREAD_COLOR)
        self.assertIsNotNone(img)
        det = self.detector.detect(img)
        skin_proc = MatteSkinProcessor(self.detector)

        # 1. Deep ebony skin tone (L compressed into ~18..45)
        lab_ebony = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_ebony[:, :, 0] = np.clip(lab_ebony[:, :, 0] * 0.22, 16.0, 65.0)
        img_ebony = cv2.cvtColor(lab_ebony.astype(np.uint8), cv2.COLOR_LAB2BGR)
        prof_ebony = self.detector.sample_adaptive_skin_profile(img_ebony, det)
        self.assertLess(prof_ebony.l_median, 45.0)
        self.assertGreater(prof_ebony.l_median, 14.0)
        _, core_ebony = skin_proc.build_dynamic_skin_mask(img_ebony, det)
        self.assertGreater(
            int(np.sum(core_ebony)),
            500,
            "Deep ebony skin tone (L<35) was clipped by AdaptiveSkinToneProfile!",
        )

        # 2. Olive / bronze tan skin tone
        lab_olive = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_olive[:, :, 0] = np.clip(lab_olive[:, :, 0] * 0.72, 40.0, 220.0)
        lab_olive[:, :, 1] = np.clip(lab_olive[:, :, 1] + 4.0, 0.0, 255.0)
        lab_olive[:, :, 2] = np.clip(lab_olive[:, :, 2] + 10.0, 0.0, 255.0)
        img_olive = cv2.cvtColor(lab_olive.astype(np.uint8), cv2.COLOR_LAB2BGR)
        _, core_olive = skin_proc.build_dynamic_skin_mask(img_olive, det)
        self.assertGreater(int(np.sum(core_olive)), 500)

        # 3. Very pale / Nordic porcelain-fair skin tone
        lab_pale = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_pale[:, :, 0] = np.clip(lab_pale[:, :, 0] * 1.18 + 18.0, 0.0, 248.0)
        img_pale = cv2.cvtColor(lab_pale.astype(np.uint8), cv2.COLOR_LAB2BGR)
        _, core_pale = skin_proc.build_dynamic_skin_mask(img_pale, det)
        self.assertGreater(int(np.sum(core_pale)), 500)

    def test_10_adversarial_universal_canvas_aspect_ratios_and_offcenter_subjects(self) -> None:
        """Red Team Audit Axis 4: Verify 21:9, 16:9, 9:16 vertical, and 1:1 square canvases preserve off-center subjects."""
        src_21x9 = cv2.imread(
            str(self.outputs_dir / "21x9_studio_seated_01_terracotta_cane.jpg"),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(src_21x9)
        # Downscale 2x for fast multi-aspect verification while keeping crisp face detection
        scene_wide = cv2.resize(src_21x9, (2520, 1080), interpolation=cv2.INTER_AREA)

        # Shift subject to left, center, and right in synthetic wide scenes
        face_orig = self.detector.detect(scene_wide)
        cx_orig = int(round(face_orig.center[0]))
        w_wide = scene_wide.shape[1]

        shift_left = int(round(0.22 * w_wide)) - cx_orig
        shift_right = int(round(0.78 * w_wide)) - cx_orig
        scenes = {
            "left": np.roll(scene_wide, shift_left, axis=1),
            "center": scene_wide,
            "right": np.roll(scene_wide, shift_right, axis=1),
        }

        for ratio_name, (tw, th) in [
            ("21:9_5K", (5040, 2160)),
            ("16:9_4K", (3840, 2160)),
            ("9:16_Vertical", (1080, 1920)),
            ("1:1_Square", (1080, 1080)),
        ]:
            cfg = EngineConfig(canvas=CanvasSpec(target_width=tw, target_height=th))
            comp = Seamless5KCompositor(cfg, self.detector)
            for pos_name, sc_img in scenes.items():
                canvas, _, _ = comp.build_ultrawide_canvas(sc_img)
                self.assertEqual(canvas.shape, (th, tw, 3), f"Mismatch for {ratio_name} ({pos_name})")
                # Ensure the subject's face was NOT cropped out in 9:16 or 1:1!
                detected = self.detector.detect(canvas)
                self.assertGreater(
                    detected.confidence,
                    0.60,
                    f"Subject face lost or clipped in {ratio_name} with subject at {pos_name}!",
                )

    def test_11_adversarial_cli_custom_model_outfit_location_angle(self) -> None:
        """Red Team Audit Axis 2: Verify CLI supports arbitrary custom model, outfit, location, props, and angle."""
        import io
        import json
        from contextlib import redirect_stdout
        from model_generation_engine.cli import main as cli_main

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli_main(
                [
                    "plan",
                    "--ref-face",
                    str(self.ref_face),
                    "--identity-label",
                    "Scandinavian female architect in her 30s",
                    "--height-cm",
                    "172",
                    "--weight-kg",
                    "56",
                    "--hairstyle",
                    "sleek platinum-blonde low bun",
                    "--eyewear",
                    "no glasses",
                    "--custom-outfit",
                    "emerald silk trench dress with wide sculptural lapel",
                    "--custom-accessories",
                    "brushed titanium cuff bracelet",
                    "--custom-location",
                    "sunlit Nordic fjord glass pavilion",
                    "--custom-props",
                    "minimalist birchwood architectural drafting table",
                    "--custom-lighting",
                    "crisp high-latitude morning daylight",
                    "--custom-angle",
                    "45-degree elevated high-angle editorial perspective",
                    "--custom-shot-scale",
                    "full-body architectural portrait shot",
                    "--custom-pose",
                    "leaning lightly against the glass balustrade",
                    "--focal-length",
                    "50",
                    "--negative-space-side",
                    "right",
                    "--max-count",
                    "1",
                ]
            )
        self.assertEqual(rc, 0)
        plans = json.loads(buf.getvalue())
        self.assertEqual(len(plans), 1)
        p0 = plans[0]["prompt"]
        self.assertIn("Scandinavian female architect in her 30s", p0)
        self.assertIn("172cm, 56kg", p0)
        self.assertIn("sleek platinum-blonde low bun", p0)
        self.assertIn("wearing no glasses (bare eyes)", p0)
        self.assertIn("emerald silk trench dress", p0)
        self.assertIn("brushed titanium cuff bracelet", p0)
        self.assertIn("sunlit Nordic fjord glass pavilion", p0)
        self.assertIn("minimalist birchwood architectural drafting table", p0)
        self.assertIn("45-degree elevated high-angle editorial perspective", p0)
        self.assertIn("50mm", p0)

        # Partial custom override: custom outfit combined with outdoor_natural preset & multi_angle_portfolio preset
        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            rc2 = cli_main(
                [
                    "plan-variations",
                    "--ref-face",
                    str(self.ref_face),
                    "--custom-outfit",
                    "bespokeburgundy velvet dinner jacket over black silk shirt",
                    "--location-preset",
                    "outdoor_natural",
                    "--camera-preset",
                    "multi_angle_portfolio",
                    "--max-count",
                    "3",
                ]
            )
        self.assertEqual(rc2, 0)
        plans2 = json.loads(buf2.getvalue())
        self.assertEqual(len(plans2), 3)
        self.assertIn("salt flat", plans2[0]["location"])
        self.assertIn("coastal basalt cliff", plans2[1]["location"])

    def test_12_static_ast_and_regex_zero_hardcoding_audit(self) -> None:
        """Red Team Audit Axes 1 & 5: Static code scan verifying zero hardcoded gender/glasses/LAB thresholds + face-aware lineage guard."""
        import re

        pkg_dir = Path(__file__).resolve().parent.parent / "model_generation_engine"
        py_files = sorted(pkg_dir.rglob("*.py"))
        self.assertGreaterEqual(len(py_files), 14)

        forbidden_code_regexes = [
            (r"Korean\s+male", "Hardcoded 'Korean male' subject label"),
            (r"browline\s+glasses", "Hardcoded 'browline glasses' eyewear"),
            (r"\bhis\s+(?:face|hair|eyes|head|body|jaw|skin|cheeks)\b", "Hardcoded male pronoun 'his'"),
            (r"L\s*>\s*125", "Hardcoded CIE LAB skin L>125 threshold"),
            (r"A\s*>\s*131", "Hardcoded CIE LAB skin A>131 threshold"),
            (r"B\s*>\s*133", "Hardcoded CIE LAB skin B>133 threshold"),
        ]

        for py_path in py_files:
            content = py_path.read_text(encoding="utf-8")
            for pattern, reason in forbidden_code_regexes:
                self.assertIsNone(
                    re.search(pattern, content, flags=re.IGNORECASE),
                    f"Red Team static audit violation in {py_path.name}: {reason} (pattern: {pattern})",
                )

        # Also verify SinglePassGenerationGuard face-aware auxiliary reference rejection
        with self.assertRaises(ReferenceIsolationError) as ctx:
            SinglePassGenerationGuard.validate_reference_lineage(
                reference_image_paths=[self.ref_face],
                generation_depth=0,
                auxiliary_reference_paths=[self.ref_face],
                face_detector=self.detector,
            )
        self.assertIn("contains a human face", str(ctx.exception))

    def test_13_anti_smudge_per_organ_sharpness_and_anatomy_gate(self) -> None:
        """Verify Anti-Smudge Organ Masking, Per-Organ Gradient Gate, and Anatomical Landmark Geometry Gate."""
        auditor = QualityGateAuditor(self.config, self.detector)
        fpath = self.outputs_dir / "21x9_studio_seated_01_terracotta_cane.jpg"
        img_5k = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        face = self.detector.detect(img_5k)
        crop_box = self.detector.compute_upper_body_crop_box(img_5k.shape[:2], face)

        # 1. Clean 5K hero image passes all 7 gates including per-organ anti-smudge gate
        report_clean = auditor.audit(img_5k, crop_box, seam_l1_diff=1.0)
        self.assertTrue(report_clean.passed, report_clean.message)
        self.assertGreaterEqual(report_clean.min_organ_gradient, self.config.min_organ_gradient_energy)
        self.assertTrue(report_clean.anatomical_geometry_valid)

        # 2. Simulate localized smudge on the left eye ONLY (while hair, glasses rim, and rest of face remain sharp)
        smudged_5k = img_5k.copy()
        x1, y1, x2, y2 = crop_box
        patch = smudged_5k[y1:y2, x1:x2]
        face_in_patch = self.detector.detect(patch)
        le = face_in_patch.landmarks_5pt[0]
        r_blur = max(18, int(round(0.32 * face_in_patch.inter_ocular_distance)))
        cx, cy = int(round(le[0])), int(round(le[1]))
        eye_box = patch[max(0, cy - r_blur) : min(patch.shape[0], cy + r_blur), max(0, cx - r_blur) : min(patch.shape[1], cx + r_blur)]
        patch[max(0, cy - r_blur) : min(patch.shape[0], cy + r_blur), max(0, cx - r_blur) : min(patch.shape[1], cx + r_blur)] = (
            cv2.GaussianBlur(eye_box, (61, 61), 25.0)
        )
        min_grad_smudged, scores_smudged = auditor.measure_per_organ_sharpness(patch, face_in_patch)
        self.assertLess(scores_smudged["left_eye"], self.config.min_organ_gradient_energy)
        self.assertLess(min_grad_smudged, self.config.min_organ_gradient_energy)

    def test_14_sface_identity_gate_3d_proportions_and_tonematched_synthesis(self) -> None:
        """Verify SFace 128-D Identity Gate rejects different person & all 11 Tone-Matched 3D Teacher 02 5K deliverables pass 9 gates."""
        auditor = QualityGateAuditor(self.config, self.detector)
        ref_t2 = cv2.imread(str(self.config.vault_dir / "resources" / "ref_teacher_02_face.jpg"))
        self.assertIsNotNone(ref_t2)
        ref_t1 = cv2.imread(str(self.ref_face))
        f_t2 = self.detector.detect(ref_t2)
        f_t1 = self.detector.detect(ref_t1)

        # 1. Different person must score well below min_identity_cosine_sim (0.65)
        sim_diff = auditor.measure_identity_similarity(ref_t2, ref_t1, f_t2, f_t1)
        self.assertLess(sim_diff, 0.35)

        # 2. Tone-Matched 3D Refinement Prompt must enforce eyewear temple continuity and scene tone & manner harmony
        subj_eyewear = SubjectProfile(
            subject_identity_label="East Asian instructor in his 40s",
            eyewear="dark tortoiseshell-and-metal combination glasses",
        )
        tm_prompt = PromptFirewall.build_tonematched_3d_refinement_prompt(
            subj_eyewear, "warm terracotta studio portrait"
        )
        PromptFirewall.validate_prompt(tm_prompt)
        self.assertIn("temple arms", tm_prompt)
        self.assertIn("Tone & Manner Harmony", tm_prompt)
        self.assertEqual(self.config.auto_facelock_below_iod_5k_px, 0.0)

        # 3. All 11 Tone-Matched 3D Teacher 02 5K deliverables must pass all 9 Quality Gates
        teacher02_files = [
            "21x9_teacher02_seated_01_terracotta_cane.jpg",
            "21x9_teacher02_seated_02_olive_bauhaus.jpg",
            "21x9_teacher02_seated_03_taupe_camel.jpg",
            "21x9_teacher02_seated_04_navy_waistcoat.jpg",
            "21x9_teacher02_seated_05_burgundy_suede.jpg",
            "21x9_teacher02_seated_06_sand_suit.jpg",
            "21x9_teacher02_door_01_vermilion_touch.jpg",
            "21x9_teacher02_door_02_vermilion_stride.jpg",
            "21x9_teacher02_door_03_navy_orange.jpg",
            "21x9_teacher02_door_04_green_orange.jpg",
            "21x9_teacher02_door_05_periwinkle_cutout.jpg",
        ]
        for fname in teacher02_files:
            fpath = self.outputs_dir / fname
            self.assertTrue(fpath.exists(), f"Missing Teacher 02 5K deliverable: {fpath}")
            hero_t2 = cv2.imread(str(fpath))
            f_hero = self.detector.detect(hero_t2)
            crop_box = self.detector.compute_upper_body_crop_box(hero_t2.shape[:2], f_hero)
            rep = auditor.audit(hero_t2, crop_box, seam_l1_diff=1.5, reference_face_bgr=ref_t2)
            self.assertTrue(rep.passed, f"{fname} failed 9-Gate audit: {rep.message}")
            self.assertGreaterEqual(
                rep.identity_cosine_sim, self.config.min_identity_cosine_sim, f"{fname} low SFace similarity"
            )

    def test_15_real_person_3d_turnaround_and_angle_routing(self) -> None:
        """Verify RealPersonTurnaroundEngine canonical framing, craniofacial blueprint, 2-hop prompts, sheet assembly, and angle routing."""
        import tempfile
        from model_generation_engine import ModelGenerationPipeline, TURNAROUND_VIEWS

        pipe = ModelGenerationPipeline(self.config)
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            canon_path, bp = pipe.turnaround.prepare_canonical_identity_anchor(
                self.ref_face, tmp / "canonical_front.jpg", target_size=(600, 750)
            )
            self.assertTrue(canon_path.exists())
            self.assertGreater(bp.eye_to_chin_ratio, 1.0)
            self.assertGreater(bp.width_to_length_ratio, 0.7)

            subj = SubjectProfile(subject_identity_label="Adult editorial subject")
            # Hop 1 + Hop 2 with dual-anchor 3/4 bridges
            fake_3q_l = tmp / "3q_left.jpg"
            fake_3q_r = tmp / "3q_right.jpg"
            fake_p_l = tmp / "prof_left.jpg"
            fake_p_r = tmp / "prof_right.jpg"
            for fp in (fake_3q_l, fake_3q_r, fake_p_l, fake_p_r):
                fp.write_bytes(canon_path.read_bytes())

            plans = pipe.turnaround.plan_progressive_5angle_turnaround(
                canon_path, subj, approved_3q_left_path=fake_3q_l, approved_3q_right_path=fake_3q_r
            )
            self.assertEqual(set(plans.keys()), set(TURNAROUND_VIEWS))
            self.assertEqual(len(plans["three_quarter_left"]["reference_images"]), 1)
            self.assertEqual(len(plans["left_profile"]["reference_images"]), 2)
            for v in ("three_quarter_left", "three_quarter_right", "left_profile", "right_profile"):
                PromptFirewall.validate_prompt(plans[v]["prompt"])

            angle_bank = {
                "left_profile": fake_p_l,
                "three_quarter_left": fake_3q_l,
                "front": canon_path,
                "three_quarter_right": fake_3q_r,
                "right_profile": fake_p_r,
            }
            sheet_path = pipe.turnaround.assemble_master_turnaround_sheet(
                angle_bank, tmp / "master_sheet.jpg", cell_size=(240, 300), include_labels=False
            )
            self.assertTrue(sheet_path.exists())

            refs_3q, resolved = pipe.turnaround.select_angle_matched_references(
                canon_path, angle_bank, target_view="three_quarter_left"
            )
            self.assertEqual(resolved, "three_quarter_left")
            self.assertEqual(refs_3q, [canon_path, fake_3q_l])

    def test_16_8angle_360_compass_foreign_face_masking_and_3d_previz(self) -> None:
        """Verify 3-hop 8-Angle (0..315 deg) turnaround, continuous azimuth compass, foreign-face grey masking, and 3D Previz."""
        import tempfile
        from model_generation_engine import (
            ModelGenerationPipeline,
            PosePrevizSpec,
            Previz3DRenderer,
            TURNAROUND_8_VIEWS,
            resolve_8angle_view_from_azimuth,
        )

        # 1. Continuous 0..360 deg azimuth -> 8 canonical views
        expected_compass = [
            (0.0, "front"),
            (18.0, "front"),
            (45.0, "three_quarter_left"),
            (90.0, "left_profile"),
            (135.0, "back_left_3q"),
            (180.0, "back"),
            (225.0, "back_right_3q"),
            (270.0, "right_profile"),
            (315.0, "three_quarter_right"),
            (350.0, "front"),
        ]
        for az, exp_view in expected_compass:
            self.assertEqual(resolve_8angle_view_from_azimuth(az), exp_view)

        pipe = ModelGenerationPipeline(self.config)
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            canon_path, _ = pipe.turnaround.prepare_canonical_identity_anchor(
                self.ref_face, tmp / "canon_0.jpg", target_size=(600, 750)
            )
            bank = {}
            for v in TURNAROUND_8_VIEWS:
                p = tmp / f"{v}.jpg"
                p.write_bytes(canon_path.read_bytes())
                bank[v] = p

            subj = SubjectProfile(subject_identity_label="Universal subject")
            plans8 = pipe.turnaround.plan_progressive_8angle_turnaround(canon_path, subj, existing_angles=bank)
            self.assertEqual(set(plans8.keys()), set(TURNAROUND_8_VIEWS))
            self.assertEqual(plans8["three_quarter_left"]["hop"], 1)
            self.assertEqual(plans8["left_profile"]["hop"], 2)
            self.assertEqual(plans8["back_left_3q"]["hop"], 3)
            self.assertEqual(plans8["back"]["hop"], 3)
            for v in TURNAROUND_8_VIEWS:
                if v != "front":
                    PromptFirewall.validate_prompt(str(plans8[v]["prompt"]))

            sheet8 = pipe.turnaround.assemble_master_turnaround_sheet(
                bank, tmp / "sheet_8.jpg", cell_size=(200, 250), include_labels=True, columns=4
            )
            self.assertTrue(sheet8.exists())

            # 2. Foreign-face grey masking on reference images containing another human face
            ref_bgr = cv2.imread(str(self.ref_face))
            masked_bgr, masked_cnt = pipe.turnaround.mask_foreign_faces_in_reference(
                ref_bgr, output_masked_path=tmp / "masked_ref.jpg"
            )
            self.assertGreaterEqual(masked_cnt, 1)
            self.assertFalse(np.array_equal(ref_bgr, masked_bgr))

            # 3. 3D Previz Pose Renderer + joint overrides
            pose = PosePrevizSpec(
                preset_id="seated_armchair",
                head_azimuth_deg=225.0,
                torso_azimuth_deg=210.0,
                camera_elevation_deg=-12.0,
                focal_length_mm=50,
                joints_override={"r_wrist": (0.18, 0.35, 0.22)},
            )
            self.assertEqual(pose.matched_8angle_view, "back_right_3q")
            previz_img = Previz3DRenderer.render_pose_guide(pose, canvas_size=(640, 360), output_path=tmp / "previz.jpg")
            self.assertEqual(previz_img.shape, (360, 640, 3))
            self.assertTrue((tmp / "previz.jpg").exists())

    def test_17_multi_item_multi_location_batch_composer_and_studio_app_api(self) -> None:
        """Verify MultiSceneBatchComposer (clothes+accessories+bag+shoes x N locations x 3D Previz) and Studio FastAPI endpoints."""
        import tempfile
        from fastapi.testclient import TestClient
        from model_generation_engine import (
            AttachmentItem,
            ModelGenerationPipeline,
            PosePrevizSpec,
            SceneLocationInput,
        )
        from model_generation_engine.studio_app import StudioImageRuntime, create_studio_app

        pipe = ModelGenerationPipeline(self.config)
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            canon_path, _ = pipe.turnaround.prepare_canonical_identity_anchor(
                self.ref_face, tmp / "front.jpg", target_size=(600, 750)
            )
            fake_270 = tmp / "right_profile.jpg"
            fake_270.write_bytes(canon_path.read_bytes())

            # Create dummy product & location images (including one with a foreign face to verify auto-masking)
            solid_item = np.full((320, 320, 3), (180, 140, 110), dtype=np.uint8)
            garment_p = tmp / "jacket.jpg"
            acc_p = tmp / "watch.jpg"
            bag_p = tmp / "briefcase.jpg"
            shoe_p = tmp / "loafers.jpg"
            for p in (garment_p, acc_p, bag_p, shoe_p):
                cv2.imwrite(str(p), solid_item)

            loc1_p = tmp / "loc_with_foreign_model.jpg"
            loc1_p.write_bytes(self.ref_face.read_bytes())
            loc2_p = tmp / "loc_gallery.jpg"
            cv2.imwrite(str(loc2_p), np.full((360, 640, 3), (210, 205, 198), dtype=np.uint8))

            jobs = pipe.batch_composer.compose_batch(
                canonical_front_path=canon_path,
                subject=SubjectProfile(subject_identity_label="Editorial subject"),
                locations=[
                    SceneLocationInput(name="lounge", image_path=loc1_p, mood_or_place_prompt="warm walnut lounge"),
                    SceneLocationInput(name="gallery", image_path=loc2_p, scope="style_mood_only"),
                    SceneLocationInput(name="terrace", mood_or_place_prompt="sunlit stone terrace"),
                ],
                poses=[PosePrevizSpec(preset_id="walking_stride", head_azimuth_deg=270.0, torso_azimuth_deg=260.0)],
                angle_bank={"front": canon_path, "right_profile": fake_270},
                items=[
                    AttachmentItem(category="garment", image_path=garment_p, sub_role="outerwear", description="linen jacket"),
                    AttachmentItem(category="accessory", image_path=acc_p, sub_role="watch", description="chronograph watch"),
                    AttachmentItem(category="bag", image_path=bag_p, description="leather tote bag"),
                    AttachmentItem(category="shoe", image_path=shoe_p, description="suede loafers"),
                ],
                work_dir=tmp / "staging",
            )
            # 3 locations -> 3 independent jobs!
            self.assertEqual(len(jobs), 3)
            for j in jobs:
                self.assertEqual(j.matched_8angle_view, "right_profile")
                PromptFirewall.validate_prompt(j.prompt)
                roles = [r["role"] for r in j.ordered_reference_images]
                self.assertIn("identity_primary_front", roles)
                self.assertIn("identity_angle_right_profile", roles)
                self.assertIn("item_garment", roles)
                self.assertIn("item_accessory", roles)
                self.assertIn("item_bag", roles)
                self.assertIn("item_shoe", roles)
                self.assertIn("pose_previz_guide", roles)

            # Test Studio Web App FastAPI Server end-to-end
            ws_dir = tmp / "studio_ws"
            app = create_studio_app(workspace_dir=ws_dir, runtime=StudioImageRuntime())
            client = TestClient(app)

            r_idx = client.get("/")
            self.assertEqual(r_idx.status_code, 200)
            self.assertIn("모델생성 스튜디오", r_idx.text)

            r_st = client.get("/api/status")
            self.assertEqual(r_st.status_code, 200)
            self.assertEqual(len(r_st.json()["turnaround_views"]), 8)

            with open(self.ref_face, "rb") as fh:
                r_sub = client.post(
                    "/api/subjects",
                    data={"name": "테스트 인물", "identity_label": "adult subject"},
                    files={"photo": ("face.jpg", fh, "image/jpeg")},
                )
            self.assertEqual(r_sub.status_code, 200)
            sid = r_sub.json()["id"]

            r_8p = client.post(f"/api/subjects/{sid}/generate-8angles", json={"provider": "manifest"})
            self.assertEqual(r_8p.status_code, 200)
            self.assertEqual(len(r_8p.json()["plans"]), 8)

            with open(garment_p, "rb") as fg:
                r_up_g = client.post(
                    "/api/assets/upload",
                    data={"category": "garment", "sub_role": "jacket", "description": "olive blazer"},
                    files=[("files", ("jacket.jpg", fg, "image/jpeg"))],
                )
            self.assertEqual(r_up_g.status_code, 200)
            gid = r_up_g.json()["uploaded"][0]["id"]

            with open(loc1_p, "rb") as fl1, open(loc2_p, "rb") as fl2:
                r_up_l = client.post(
                    "/api/assets/upload",
                    data={"category": "location"},
                    files=[
                        ("files", ("loc1.jpg", fl1, "image/jpeg")),
                        ("files", ("loc2.jpg", fl2, "image/jpeg")),
                    ],
                )
            self.assertEqual(r_up_l.status_code, 200)
            self.assertGreaterEqual(r_up_l.json()["uploaded"][0]["foreign_faces_masked"], 1)
            loc_ids = [u["id"] for u in r_up_l.json()["uploaded"]]

            r_pv = client.post(
                "/api/previz/render",
                json={"preset_id": "standing_left_45", "head_azimuth_deg": 45, "torso_azimuth_deg": 45},
            )
            self.assertEqual(r_pv.status_code, 200)
            self.assertEqual(r_pv.json()["matched_8angle_view"], "three_quarter_left")

            r_bat = client.post(
                "/api/batches",
                json={
                    "subject_id": sid,
                    "item_asset_ids": [gid],
                    "location_asset_ids": loc_ids,
                    "text_locations": ["sunlit travertine courtyard"],
                    "shared_mood_prompt": "warm architectural daylight",
                    "provider": "manifest",
                    "pose": {"preset_id": "standing_left_45", "head_azimuth_deg": 45, "torso_azimuth_deg": 45},
                },
            )
            self.assertEqual(r_bat.status_code, 200)
            # 2 uploaded location images + 1 text location = 3 jobs!
            self.assertEqual(r_bat.json()["total_jobs"], 3)


if __name__ == "__main__":
    unittest.main()
