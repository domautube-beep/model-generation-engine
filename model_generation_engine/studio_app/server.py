"""FastAPI Local Production Server for 모델생성 스튜디오 v3.0 (`127.0.0.1:18830`).

Supports:
1. Real-person Face Upload -> Pure Optical Canonical Anchor + 8-Angle (`0°..315°`) Turnaround Registration & Master Sheet
2. Multi-Category Drag & Drop Attachments: Clothing (`garment`), Accessories (`accessory`), Bags (`bag`), Shoes (`shoe`)
   with automatic foreign-face grey masking
3. Multi-Location & Style Reference Batching (`locations`: N uploaded places/styles -> N parallel scene jobs)
4. Interactive 3D Previz Pose Rig (`PosePrevizSpec` + live 3D joint overrides + automatic 8-angle compass routing)
5. Parallel Multi-Scene Generation + Pure Optical `Lanczos-4` 5K (`5040x2160` / `3840x2160`) + 9-Gate Quality Audit
"""

from __future__ import annotations

import asyncio
import json
import secrets
import shutil
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import CanvasSpec, EngineConfig, SubjectProfile
from ..identity_turnaround import (
    TURNAROUND_8_VIEWS,
    TURNAROUND_FULL_SWEEP,
    VIEW_AZIMUTH_DEG,
    VIEW_LABELS_KO,
)
from ..multi_asset_composer import AttachmentItem, SceneLocationInput
from ..pipeline import ModelGenerationPipeline
from ..previz_pose import GAZE_MODES, POSE_PRESETS, PosePrevizSpec, Previz3DRenderer
from ..scene_radiometry import (
    APERTURE_MODES,
    CAMERA_PROFILES,
    COHESION_MASTERING_MODES,
    FILM_PROFILES,
    LIGHTING_MODES,
    SceneRadiometryAnalyzer,
)
from .runtime import StudioImageRuntime

STUDIO_PORT = 18830
STATIC_DIR = Path(__file__).with_name("static")

CANVAS_PRESETS: Dict[str, Tuple[int, int, str, str]] = {
    "5k_21x9": (5040, 2160, "21:9", "3840x2160"),
    "4k_16x9": (3840, 2160, "16:9", "3840x2160"),
    "4k_9x16": (2160, 3840, "9:16", "2160x3840"),
    "3x4_editorial": (3584, 4800, "3:4", "1024x1536"),
    "1x1_square": (2048, 2048, "1:1", "2048x2048"),
}


def create_studio_app(
    workspace_dir: Optional[Path] = None,
    runtime: Optional[StudioImageRuntime] = None,
) -> FastAPI:
    base_dir = Path(workspace_dir) if workspace_dir else (Path.home() / ".model_generation_studio")
    for sub in ("subjects", "assets", "batches", "previz"):
        (base_dir / sub).mkdir(parents=True, exist_ok=True)

    state_file = base_dir / "studio_state.json"
    if state_file.exists():
        try:
            db_state: Dict[str, dict] = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            db_state = {"subjects": {}, "assets": {}, "batches": {}}
    else:
        db_state = {"subjects": {}, "assets": {}, "batches": {}}

    def save_state() -> None:
        state_file.write_text(json.dumps(db_state, ensure_ascii=False, indent=2), encoding="utf-8")

    pipe = ModelGenerationPipeline()
    rt = runtime or StudioImageRuntime()
    csrf_token = secrets.token_urlsafe(24)
    bg_tasks: set = set()

    app = FastAPI(title="모델생성 스튜디오 v3.0")
    app.state.workspace_dir = base_dir
    app.state.pipeline = pipe
    app.state.runtime = rt
    app.state.db = db_state
    app.state.csrf = csrf_token

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        if request.url.hostname not in {"127.0.0.1", "localhost", "testserver"}:
            return JSONResponse({"detail": "Localhost access only."}, status_code=403)
        if (
            request.url.path.startswith("/api/")
            and request.method not in {"GET", "HEAD"}
            and request.url.hostname != "testserver"
        ):
            if request.headers.get("x-studio-csrf") != csrf_token:
                return JSONResponse({"detail": "CSRF session expired. Refresh page."}, status_code=403)
        resp = await call_next(request)
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        return resp

    def rel_url(path: Path) -> str:
        rel = Path(path).resolve().relative_to(base_dir.resolve())
        return f"/api/files/{rel.as_posix()}"

    @app.get("/")
    def index_page():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/api/files/{rel_path:path}")
    def serve_workspace_file(rel_path: str):
        full = (base_dir / rel_path).resolve()
        if not str(full).startswith(str(base_dir.resolve())) or not full.is_file():
            raise HTTPException(status_code=404, detail="File not found.")
        return FileResponse(full)

    @app.get("/api/status")
    def get_status():
        return {
            "csrf": csrf_token,
            "version": "3.0.0",
            "workspace": str(base_dir),
            "vault_dir": str(pipe.config.vault_dir) if pipe.config.vault_dir else None,
            "runtime": rt.capabilities(),
            "turnaround_views": [
                {
                    "view": v,
                    "label": VIEW_LABELS_KO[v],
                    "azimuth_deg": VIEW_AZIMUTH_DEG[v],
                    "is_origin": (v == "front"),
                }
                for v in TURNAROUND_FULL_SWEEP
            ],
            "pose_presets": [
                {
                    "id": k,
                    "label": v["label_ko"],
                    "shot_scale": v["shot_scale"],
                    "torso_azimuth_deg": v["torso_azimuth_deg"],
                    "head_azimuth_deg": v["head_azimuth_deg"],
                    "head_pitch_deg": v.get("head_pitch_deg", 0.0),
                    "head_roll_deg": v.get("head_roll_deg", 0.0),
                    "gaze_mode": v.get("gaze_mode", "auto"),
                    "joints": v["joints"],
                }
                for k, v in POSE_PRESETS.items()
            ],
            "camera_profiles": [{"id": "auto", "label": "자동 촬영감독 (무드·조도·레퍼런스 맞춰 바디 찰떡 선택)"}]
            + [{"id": k, "label": str(v["label_ko"])} for k, v in CAMERA_PROFILES.items()],
            "film_profiles": [{"id": "auto", "label": "자동 촬영감독 (화이트밸런스·색감·톤·무드 찰떡 필름 매칭)"}]
            + [{"id": k, "label": str(v["label_ko"])} for k, v in FILM_PROFILES.items()],
            "lighting_modes": [
                {"id": k, "label": str(v["label_ko"])} for k, v in LIGHTING_MODES.items()
            ],
            "cohesion_modes": [
                {"id": k, "label": str(v["label_ko"])} for k, v in COHESION_MASTERING_MODES.items()
            ],
            "canvas_presets": [
                {"id": k, "width": v[0], "height": v[1], "aspect": v[2]}
                for k, v in CANVAS_PRESETS.items()
            ],
            "subjects": list(db_state["subjects"].values()),
            "assets": list(db_state["assets"].values()),
            "batches": sorted(db_state["batches"].values(), key=lambda b: b["created_at"], reverse=True)[:25],
        }

    # ── 1. Subject Registration & 8-Angle Turnaround Bank ──
    @app.post("/api/subjects")
    async def create_subject(
        name: str = Form("신규 인물"),
        identity_label: str = Form("human subject in the reference portrait"),
        height_cm: Optional[float] = Form(None),
        weight_kg: Optional[float] = Form(None),
        eyewear: str = Form(""),
        hairstyle: str = Form(""),
        photo: UploadFile = File(...),
    ):
        sid = f"subj_{uuid.uuid4().hex[:8]}"
        sdir = base_dir / "subjects" / sid
        sdir.mkdir(parents=True, exist_ok=True)
        ext = Path(photo.filename or "face.jpg").suffix.lower() or ".jpg"
        raw_path = sdir / f"raw_upload{ext}"
        raw_path.write_bytes(await photo.read())

        canon_path = sdir / "angle_000_front.jpg"
        try:
            _, blueprint = pipe.turnaround.prepare_canonical_identity_anchor(raw_path, canon_path)
        except Exception as exc:
            shutil.rmtree(sdir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=f"얼굴 검출 실패: {exc}") from exc

        record = {
            "id": sid,
            "name": name.strip() or "신규 인물",
            "identity_label": identity_label.strip() or "human subject in the reference portrait",
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "eyewear": eyewear.strip() or None,
            "hairstyle": hairstyle.strip() or None,
            "canonical_front_path": str(canon_path),
            "canonical_front_url": rel_url(canon_path),
            "blueprint": asdict(blueprint),
            "angles": {
                "front": {"view": "front", "path": str(canon_path), "url": rel_url(canon_path), "status": "ready"}
            },
            "master_sheet_url": None,
            "turnaround_state": "front_ready",
            "created_at": time.time(),
        }
        db_state["subjects"][sid] = record
        save_state()
        return record

    @app.post("/api/subjects/{subject_id}/angles/{view}")
    async def upload_subject_angle(subject_id: str, view: str, photo: UploadFile = File(...)):
        if subject_id not in db_state["subjects"]:
            raise HTTPException(status_code=404, detail="인물을 찾을 수 없습니다.")
        if view not in TURNAROUND_FULL_SWEEP:
            raise HTTPException(status_code=400, detail=f"지원하지 않는 각도입니다: {view}")
        rec = db_state["subjects"][subject_id]
        sdir = base_dir / "subjects" / subject_id
        ang_tag = f"{VIEW_AZIMUTH_DEG[view]:+05.1f}".replace("+", "p").replace("-", "m").replace(".", "_")
        ang_path = sdir / f"angle_{ang_tag}_{view}.jpg"
        ang_path.write_bytes(await photo.read())
        rec["angles"][view] = {"view": view, "path": str(ang_path), "url": rel_url(ang_path), "status": "ready"}

        ready_map = {v: Path(info["path"]) for v, info in rec["angles"].items() if Path(info["path"]).exists()}
        if len(ready_map) >= 2:
            sheet_p = sdir / "master_8angle_sheet.jpg"
            pipe.turnaround.assemble_master_turnaround_sheet(ready_map, sheet_p, include_labels=True)
            rec["master_sheet_url"] = rel_url(sheet_p) + f"?t={int(time.time())}"
        save_state()
        return rec

    @app.post("/api/subjects/{subject_id}/generate-8angles")
    async def generate_subject_8angles(subject_id: str, data: Optional[dict] = None):
        if subject_id not in db_state["subjects"]:
            raise HTTPException(status_code=404, detail="인물을 찾을 수 없습니다.")
        rec = db_state["subjects"][subject_id]
        provider = (data or {}).get("provider", "auto")
        sdir = base_dir / "subjects" / subject_id
        subj_prof = SubjectProfile(
            subject_identity_label=rec["identity_label"],
            height_cm=rec.get("height_cm"),
            weight_kg=rec.get("weight_kg"),
            eyewear=rec.get("eyewear"),
            hairstyle=rec.get("hairstyle"),
        )
        canon_path = Path(rec["canonical_front_path"])

        # Always save the 8-angle prompt manifest immediately
        existing_map = {v: Path(info["path"]) for v, info in rec["angles"].items() if Path(info["path"]).exists()}
        plans_initial = pipe.turnaround.plan_progressive_8angle_turnaround(canon_path, subj_prof, existing_map)
        (sdir / "turnaround_8angle_plans.json").write_text(
            json.dumps(plans_initial, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        caps = rt.capabilities()
        eff_provider = str(caps["default_provider"]) if provider == "auto" else provider
        if eff_provider == "manifest":
            rec["turnaround_state"] = "plans_saved"
            save_state()
            return {"subject": rec, "plans": plans_initial, "mode": "manifest"}

        rec["turnaround_state"] = "generating"
        save_state()

        async def _run_1hop_parallel_turnaround() -> None:
            try:
                cur_bank = {
                    v: Path(info["path"])
                    for v, info in rec["angles"].items()
                    if info.get("status") == "ready" and Path(info["path"]).exists()
                }
                all_plans = pipe.turnaround.plan_progressive_8angle_turnaround(canon_path, subj_prof, cur_bank)
                pending_views = [
                    v for v, p in all_plans.items() if p["hop"] == 1 and v not in cur_bank
                ]

                async def _gen_one_view(v_name: str, plan_obj: dict) -> None:
                    ang_tag = (
                        f"{VIEW_AZIMUTH_DEG[v_name]:+05.1f}".replace("+", "p").replace("-", "m").replace(".", "_")
                    )
                    out_p = sdir / f"angle_{ang_tag}_{v_name}.jpg"
                    rec["angles"][v_name] = {"view": v_name, "path": str(out_p), "url": "", "status": "running"}
                    save_state()
                    # STRICT 1-HOP DIRECT FROM GEN-0: verify NO generated 8dir image is chained into another!
                    refs = [Path(rp) for rp in plan_obj["reference_images"]]
                    rt.origin_guard.verify_turnaround_1hop_inputs(refs)
                    await rt.generate_image(
                        prompt=str(plan_obj["prompt"]),
                        reference_image_paths=refs,
                        output_path=out_p,
                        size="1024x1536",
                        provider=eff_provider,
                    )
                    rt.origin_guard.register_approved_origin(out_p, kind="8dir_face", depth=1)
                    rec["angles"][v_name] = {
                        "view": v_name,
                        "path": str(out_p),
                        "url": rel_url(out_p) + f"?t={int(time.time())}",
                        "status": "ready",
                    }
                    save_state()

                if pending_views:
                    await asyncio.gather(*[_gen_one_view(v, all_plans[v]) for v in pending_views])

                final_map = {
                    v: Path(info["path"])
                    for v, info in rec["angles"].items()
                    if info.get("status") == "ready" and Path(info["path"]).exists()
                }
                rt.origin_guard.register_verified_subject_8dir_bank(canon_path, final_map)
                sheet_p = sdir / "master_8angle_sheet.jpg"
                pipe.turnaround.assemble_master_turnaround_sheet(final_map, sheet_p, include_labels=True)
                rec["master_sheet_url"] = rel_url(sheet_p) + f"?t={int(time.time())}"
                rec["turnaround_state"] = "completed"
                save_state()
            except Exception as exc:
                rec["turnaround_state"] = f"error: {exc}"
                save_state()

        task = asyncio.create_task(_run_1hop_parallel_turnaround())
        bg_tasks.add(task)
        task.add_done_callback(bg_tasks.discard)
        return {"subject": rec, "mode": eff_provider, "status": "started"}

    @app.get("/api/subjects/{subject_id}")
    def get_subject(subject_id: str):
        if subject_id not in db_state["subjects"]:
            raise HTTPException(status_code=404, detail="Subject not found.")
        return db_state["subjects"][subject_id]

    @app.delete("/api/subjects/{subject_id}")
    def delete_subject(subject_id: str):
        if subject_id in db_state["subjects"]:
            del db_state["subjects"][subject_id]
            shutil.rmtree(base_dir / "subjects" / subject_id, ignore_errors=True)
            save_state()
        return {"deleted": subject_id}

    # ── 2. Multi-Category Asset Upload (Clothes, Accessories, Bags, Shoes, Multi-Locations) ──
    @app.post("/api/assets/upload")
    async def upload_assets(
        category: str = Form(...),  # garment | accessory | bag | shoe | location
        sub_role: str = Form(""),
        description: str = Form(""),
        wearing_style: str = Form("auto"),
        scale_mode: str = Form("medium_standard"),
        size_cm: Optional[float] = Form(None),
        carry_mode: str = Form("hand_right"),
        files: List[UploadFile] = File(...),
    ):
        if category not in {"garment", "accessory", "bag", "shoe", "location"}:
            raise HTTPException(status_code=400, detail=f"지원하지 않는 카테고리입니다: {category}")
        created = []
        for uf in files:
            aid = f"{category}_{uuid.uuid4().hex[:8]}"
            adir = base_dir / "assets" / category
            adir.mkdir(parents=True, exist_ok=True)
            ext = Path(uf.filename or "item.jpg").suffix.lower() or ".jpg"
            raw_p = adir / f"{aid}{ext}"
            raw_p.write_bytes(await uf.read())

            bgr = cv2.imread(str(raw_p))
            if bgr is None:
                raw_p.unlink(missing_ok=True)
                continue

            auto_pose_info = None
            auto_radiometry = None
            if category == "location":
                auto_pose_info = pipe.turnaround.analyze_reference_pose_before_masking(bgr)
                rad = SceneRadiometryAnalyzer.analyze(scene_bgr=bgr)
                auto_radiometry = {
                    "kelvin_est": rad.kelvin_est,
                    "lighting_regime": rad.lighting_regime,
                    "contrast_ratio_stops": rad.contrast_ratio_stops,
                    "recommended_aperture": rad.recommended_aperture,
                }

            masked_p = adir / f"{aid}_clean.jpg"
            if category == "location":
                _, erased_frac = pipe.turnaround.erase_foreign_person_completely(
                    bgr, output_clean_path=masked_p, remove_overlay_text=True
                )
                masked_faces = 1 if erased_frac > 0.005 else 0
            else:
                masked_p = adir / f"{aid}_masked.jpg"
                _, masked_faces = pipe.turnaround.mask_foreign_faces_in_reference(bgr, output_masked_path=masked_p)
            active_p = masked_p if masked_faces > 0 else raw_p
            if masked_faces == 0:
                masked_p.unlink(missing_ok=True)

            pipe.batch_composer.origin_guard.register_approved_origin(active_p, kind="raw_asset")

            item_rec = {
                "id": aid,
                "category": category,
                "filename": uf.filename or raw_p.name,
                "sub_role": sub_role.strip(),
                "description": description.strip(),
                "wearing_style": wearing_style.strip() or "auto",
                "scale_mode": scale_mode.strip() or "medium_standard",
                "size_cm": size_cm,
                "carry_mode": carry_mode.strip() or "hand_right",
                "override_yaw_deg": None,
                "auto_pose_info": auto_pose_info,
                "auto_radiometry": auto_radiometry,
                "raw_path": str(raw_p),
                "active_path": str(active_p),
                "preview_url": rel_url(raw_p),
                "masked_url": rel_url(active_p),
                "foreign_faces_masked": masked_faces,
                "created_at": time.time(),
            }
            db_state["assets"][aid] = item_rec
            created.append(item_rec)
        save_state()
        return {"uploaded": created, "count": len(created)}

    @app.patch("/api/assets/{asset_id}")
    def update_asset_metadata(asset_id: str, data: dict):
        rec = db_state["assets"].get(asset_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Asset not found.")
        for field_k in ("sub_role", "description", "wearing_style", "scale_mode", "carry_mode"):
            if field_k in data and data[field_k] is not None:
                rec[field_k] = str(data[field_k]).strip()
        if "size_cm" in data:
            rec["size_cm"] = float(data["size_cm"]) if data["size_cm"] not in (None, "") else None
        if "override_yaw_deg" in data:
            rec["override_yaw_deg"] = (
                float(data["override_yaw_deg"]) if data["override_yaw_deg"] not in (None, "", "auto") else None
            )
        save_state()
        return rec

    @app.delete("/api/assets/{asset_id}")
    def delete_asset(asset_id: str):
        rec = db_state["assets"].pop(asset_id, None)
        if rec:
            for k in ("raw_path", "active_path"):
                if rec.get(k):
                    Path(rec[k]).unlink(missing_ok=True)
            save_state()
        return {"deleted": asset_id}

    # ── 3. Interactive 3D Previz Pose Preview ──
    @app.post("/api/previz/render")
    def render_previz_preview(data: dict):
        spec = PosePrevizSpec(
            preset_id=str(data.get("preset_id") or "standing_editorial"),
            head_azimuth_deg=float(data.get("head_azimuth_deg", 0.0)),
            head_pitch_deg=float(data.get("head_pitch_deg", 0.0)),
            torso_azimuth_deg=float(data.get("torso_azimuth_deg", 0.0)),
            camera_elevation_deg=float(data.get("camera_elevation_deg", 0.0)),
            focal_length_mm=int(data.get("focal_length_mm", 85)),
            subject_x_norm=float(data.get("subject_x_norm", 0.58)),
            shot_scale=str(data.get("shot_scale") or "three-quarter portrait shot"),
            custom_pose_note=str(data.get("custom_pose_note") or ""),
            joints_override={
                k: (float(v[0]), float(v[1]), float(v[2]))
                for k, v in (data.get("joints_override") or {}).items()
            }
            or None,
            bag_carry_mode=str(data.get("bag_carry_mode") or ""),
            bag_size=str(data.get("bag_size") or "medium_standard"),
        )
        out_p = base_dir / "previz" / "latest_previz.jpg"
        Previz3DRenderer.render_pose_guide(spec, canvas_size=(1280, 720), output_path=out_p)
        return {
            "matched_8angle_view": spec.matched_8angle_view,
            "matched_8angle_label": VIEW_LABELS_KO[spec.matched_8angle_view],
            "pose_clause": spec.to_prompt_clause(),
            "previz_url": rel_url(out_p) + f"?t={int(time.time() * 1000)}",
        }

    # ── 4. Multi-Location × Multi-Item × 3D Previz Parallel Batch Execution ──
    @app.post("/api/batches")
    async def launch_batch(data: dict):
        sid = data.get("subject_id")
        if not sid or sid not in db_state["subjects"]:
            raise HTTPException(status_code=400, detail="먼저 얼굴 사진(인물)을 선택하거나 등록하세요.")
        subj_rec = db_state["subjects"][sid]

        subj_prof = SubjectProfile(
            subject_identity_label=subj_rec["identity_label"],
            height_cm=subj_rec.get("height_cm"),
            weight_kg=subj_rec.get("weight_kg"),
            eyewear=subj_rec.get("eyewear"),
            hairstyle=subj_rec.get("hairstyle"),
        )
        angle_bank = {
            v: Path(info["path"])
            for v, info in subj_rec.get("angles", {}).items()
            if info.get("status") == "ready" and Path(info["path"]).exists()
        }

        # Gather attached garments, accessories, bags, shoes with full layering, scale, and carry metadata
        items: List[AttachmentItem] = []
        for aid in data.get("item_asset_ids", []):
            arec = db_state["assets"].get(aid)
            if arec and arec["category"] in {"garment", "accessory", "bag", "shoe"}:
                items.append(
                    AttachmentItem(
                        category=arec["category"],
                        image_path=Path(arec["active_path"]),
                        description=arec.get("description", ""),
                        sub_role=arec.get("sub_role", ""),
                        wearing_style=arec.get("wearing_style", "auto"),
                        scale_mode=arec.get("scale_mode", "medium_standard"),
                        size_cm=arec.get("size_cm"),
                        carry_mode=arec.get("carry_mode", "hand_right"),
                    )
                )
        for txt_item in data.get("text_items", []):
            if txt_item.get("description", "").strip():
                items.append(
                    AttachmentItem(
                        category=txt_item.get("category", "garment"),
                        description=txt_item["description"].strip(),
                        sub_role=txt_item.get("sub_role", ""),
                        wearing_style=txt_item.get("wearing_style", "auto"),
                        scale_mode=txt_item.get("scale_mode", "medium_standard"),
                        size_cm=txt_item.get("size_cm"),
                        carry_mode=txt_item.get("carry_mode", "hand_right"),
                    )
                )

        # Gather 1..N locations (each uploaded location image + optional text mood creates an independent shot!)
        loc_inputs: List[SceneLocationInput] = []
        shared_mood = str(data.get("shared_mood_prompt") or "").strip()
        shared_light = str(data.get("shared_lighting_note") or "").strip()
        loc_scope = str(data.get("location_scope") or "scene_and_mood")

        loc_asset_ids: List[str] = data.get("location_asset_ids") or []
        for idx, laid in enumerate(loc_asset_ids, 1):
            lrec = db_state["assets"].get(laid)
            if lrec:
                per_note = str((data.get("location_notes") or {}).get(laid) or lrec.get("description") or shared_mood)
                loc_inputs.append(
                    SceneLocationInput(
                        name=lrec.get("filename") or f"장소_{idx:02d}",
                        image_path=Path(lrec["active_path"]),
                        raw_unmasked_path=Path(lrec["raw_path"]) if lrec.get("raw_path") else None,
                        mood_or_place_prompt=per_note,
                        lighting_note=shared_light,
                        scope=loc_scope,
                        override_yaw_deg=lrec.get("override_yaw_deg"),
                    )
                )

        for extra_prompt in data.get("text_locations") or []:
            if str(extra_prompt).strip():
                loc_inputs.append(
                    SceneLocationInput(
                        name=f"텍스트장소_{len(loc_inputs) + 1:02d}",
                        mood_or_place_prompt=str(extra_prompt).strip(),
                        lighting_note=shared_light,
                        scope="scene_and_mood",
                    )
                )

        if not loc_inputs:
            loc_inputs.append(
                SceneLocationInput(
                    name="기본_분위기_장면",
                    mood_or_place_prompt=shared_mood or "clean architectural editorial space",
                    lighting_note=shared_light,
                )
            )

        pose_raw = data.get("pose") or {}
        pose_spec = PosePrevizSpec(
            preset_id=str(pose_raw.get("preset_id") or "standing_editorial"),
            head_azimuth_deg=float(pose_raw.get("head_azimuth_deg", 0.0)),
            head_pitch_deg=float(pose_raw.get("head_pitch_deg", 0.0)),
            head_roll_deg=float(pose_raw.get("head_roll_deg", 0.0)),
            gaze_mode=str(pose_raw.get("gaze_mode") or "auto"),
            torso_azimuth_deg=float(pose_raw.get("torso_azimuth_deg", 0.0)),
            camera_elevation_deg=float(pose_raw.get("camera_elevation_deg", 0.0)),
            focal_length_mm=int(pose_raw.get("focal_length_mm", 85)),
            subject_x_norm=float(pose_raw.get("subject_x_norm", 0.58)),
            shot_scale=str(pose_raw.get("shot_scale") or "three-quarter portrait shot"),
            custom_pose_note=str(pose_raw.get("custom_pose_note") or ""),
            joints_override={
                k: (float(v[0]), float(v[1]), float(v[2]))
                for k, v in (pose_raw.get("joints_override") or {}).items()
            }
            or None,
        )

        canvas_key = str(data.get("canvas_preset") or "4k_16x9")
        tw, th, aspect_lbl, gen_size = CANVAS_PRESETS.get(canvas_key, CANVAS_PRESETS["4k_16x9"])
        provider = str(data.get("provider") or "auto")
        lighting_mode = str(data.get("lighting_mode") or "auto")
        aperture_mode = str(data.get("aperture_mode") or "auto")
        camera_id = str(data.get("camera_id") or "auto")
        film_id = str(data.get("film_id") or "auto")
        cohesion_mode = str(data.get("cohesion_mode") or film_id)

        bid = f"batch_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        bdir = base_dir / "batches" / bid
        bdir.mkdir(parents=True, exist_ok=True)

        sdir_subj = base_dir / "subjects" / sid
        sheet_cand = sdir_subj / "master_8angle_sheet.jpg"
        face_dir_mode = str(data.get("face_direction_mode") or "auto")

        try:
            composed_jobs = pipe.batch_composer.compose_batch(
                canonical_front_path=Path(subj_rec["canonical_front_path"]),
                subject=subj_prof,
                locations=loc_inputs,
                poses=[pose_spec],
                angle_bank=angle_bank,
                master_8dir_sheet_path=sheet_cand if sheet_cand.exists() else None,
                items=items,
                work_dir=bdir / "staging",
                aspect_ratio_label=aspect_lbl,
                face_direction_mode=face_dir_mode,
                lighting_mode=lighting_mode,
                aperture_mode=aperture_mode,
                camera_id=camera_id,
                film_id=film_id,
                cohesion_mode=cohesion_mode,
                attach_previz_image=False,  # Strict `1원본들 → 1생성`: ONLY 8-dir faces + raw original assets!
                require_complete_8dir_bank=True,
            )
        except Exception as exc:
            shutil.rmtree(bdir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        caps = rt.capabilities()
        eff_provider = str(caps["default_provider"]) if provider == "auto" else provider

        batch_rec = {
            "id": bid,
            "subject_id": sid,
            "subject_name": subj_rec["name"],
            "canvas_preset": canvas_key,
            "target_size": [tw, th],
            "provider": eff_provider,
            "camera_id": camera_id,
            "film_id": film_id,
            "lighting_mode": lighting_mode,
            "aperture_mode": aperture_mode,
            "cohesion_mode": cohesion_mode,
            "total_jobs": len(composed_jobs),
            "status": "manifest_ready" if eff_provider == "manifest" else "running",
            "created_at": time.time(),
            "jobs": [
                {
                    "job_index": j.job_index,
                    "job_tag": j.job_tag,
                    "location_name": j.location_name,
                    "matched_8angle_view": j.matched_8angle_view,
                    "matched_8angle_label": VIEW_LABELS_KO.get(j.matched_8angle_view, j.matched_8angle_view),
                    "head_azimuth_deg": j.head_azimuth_deg,
                    "head_pitch_deg": j.head_pitch_deg,
                    "head_roll_deg": j.head_roll_deg,
                    "gaze_mode": j.gaze_mode,
                    "cohesion_mode": j.cohesion_mode,
                    "reference_scene_path": j.reference_scene_path,
                    "radiometry": asdict(j.radiometric_spec) if j.radiometric_spec else None,
                    "prompt": j.prompt,
                    "reference_images": j.ordered_reference_images,
                    "previz_url": rel_url(Path(j.previz_guide_path)) if j.previz_guide_path else None,
                    "status": "manifest_saved" if eff_provider == "manifest" else "queued",
                    "raw_path": None,
                    "hero_url": None,
                    "closeup_url": None,
                    "quality": None,
                    "error": None,
                }
                for j in composed_jobs
            ],
        }
        (bdir / "batch_manifest.json").write_text(
            json.dumps(batch_rec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        db_state["batches"][bid] = batch_rec
        save_state()

        if eff_provider != "manifest":
            upscale_mode_req = str(data.get("upscale_mode") or "upscayl_auto")
            reattach_mode_req = str(data.get("reattach_mode") or "none")
            custom_pipe = ModelGenerationPipeline(
                EngineConfig(
                    upscale_mode=upscale_mode_req,
                    canvas=CanvasSpec(target_width=tw, target_height=th),
                )
            )

            async def _execute_parallel_batch() -> None:
                async def _run_job(j_obj, j_dict: dict) -> None:
                    j_dict["status"] = "generating"
                    save_state()
                    try:
                        raw_out = bdir / f"raw_{j_obj.job_tag}.png"
                        hero_out = bdir / f"hero_{tw}x{th}_{j_obj.job_tag}.jpg"
                        cu_out = bdir / f"closeup_{j_obj.job_tag}.jpg"
                        ref_paths = [Path(r["path"]) for r in j_obj.ordered_reference_images]
                        await rt.generate_image(
                            prompt=j_obj.prompt,
                            reference_image_paths=ref_paths,
                            output_path=raw_out,
                            size=gen_size,
                            provider=eff_provider,
                            job_id=f"{bid}_{j_obj.job_tag}",
                        )
                        j_dict["raw_path"] = str(raw_out)
                        res = await asyncio.to_thread(
                            custom_pipe.run_single_pass_5k,
                            raw_out,
                            hero_out,
                            cu_out,
                            42,
                            Path(subj_rec["canonical_front_path"]),
                            reattach_mode_req,
                            None,
                            j_obj.cohesion_mode,
                            Path(j_obj.reference_scene_path) if j_obj.reference_scene_path else None,
                            j_obj.radiometric_spec,
                        )
                        if pipe.config.vault_dir:
                            v_out = pipe.config.vault_dir / "outputs"
                            v_out.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(hero_out, v_out / f"{bid}_{hero_out.name}")
                        rep = res["quality_report"]
                        j_dict["hero_url"] = rel_url(hero_out) + f"?t={int(time.time())}"
                        j_dict["closeup_url"] = rel_url(cu_out) + f"?t={int(time.time())}"
                        j_dict["quality"] = asdict(rep)
                        j_dict["status"] = "done"
                    except Exception as exc:
                        j_dict["status"] = "failed"
                        j_dict["error"] = str(exc)
                    save_state()

                await asyncio.gather(*[_run_job(c_job, batch_rec["jobs"][i]) for i, c_job in enumerate(composed_jobs)])
                batch_rec["status"] = (
                    "done"
                    if all(j["status"] == "done" for j in batch_rec["jobs"])
                    else "completed_with_errors"
                )
                save_state()

            task = asyncio.create_task(_execute_parallel_batch())
            bg_tasks.add(task)
            task.add_done_callback(bg_tasks.discard)

        return batch_rec

    @app.post("/api/batches/{batch_id}/jobs/{job_index}/remaster")
    async def remaster_job_cohesion(batch_id: str, job_index: int, data: dict):
        """Instant zero-regeneration post-optical mood/tone/contrast/color remastering on an existing batch shot."""
        batch_rec = db_state["batches"].get(batch_id)
        if not batch_rec:
            raise HTTPException(status_code=404, detail="배치를 찾을 수 없습니다.")
        target_job = next((j for j in batch_rec.get("jobs", []) if int(j["job_index"]) == int(job_index)), None)
        if not target_job:
            raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")

        bdir = base_dir / "batches" / batch_id
        raw_p = Path(target_job.get("raw_path") or (bdir / f"raw_{target_job['job_tag']}.png"))
        if not raw_p.exists():
            raise HTTPException(status_code=400, detail="원본 생성 파일이 없어 재마스터링할 수 없습니다.")

        new_mode = str(data.get("cohesion_mode") or "auto")
        tw, th = batch_rec["target_size"]
        hero_out = bdir / f"hero_{tw}x{th}_{target_job['job_tag']}.jpg"
        cu_out = bdir / f"closeup_{target_job['job_tag']}.jpg"
        subj_rec = db_state["subjects"].get(batch_rec["subject_id"])
        ref_face_p = Path(subj_rec["canonical_front_path"]) if subj_rec else None
        ref_scene_p = Path(target_job["reference_scene_path"]) if target_job.get("reference_scene_path") else None

        custom_pipe = ModelGenerationPipeline(EngineConfig(canvas=CanvasSpec(target_width=tw, target_height=th)))
        res = await asyncio.to_thread(
            custom_pipe.run_single_pass_5k,
            raw_p,
            hero_out,
            cu_out,
            42,
            ref_face_p,
            "none",
            None,
            new_mode,
            ref_scene_p,
            None,
        )
        target_job["cohesion_mode"] = new_mode
        target_job["hero_url"] = rel_url(hero_out) + f"?t={int(time.time() * 1000)}"
        target_job["closeup_url"] = rel_url(cu_out) + f"?t={int(time.time() * 1000)}"
        target_job["quality"] = asdict(res["quality_report"])
        save_state()
        return target_job

    @app.get("/api/batches/{batch_id}")
    def get_batch(batch_id: str):
        if batch_id not in db_state["batches"]:
            raise HTTPException(status_code=404, detail="Batch not found.")
        return db_state["batches"][batch_id]

    return app
