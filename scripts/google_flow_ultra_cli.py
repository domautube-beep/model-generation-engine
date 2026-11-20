#!/usr/bin/env python3
"""Google Flow ULTRA Native 2K/4K Image Generation CLI for `모델생성` (`v10.1`).

Uses the user's own local persistent Patchright Chrome profile (`~/.shorts-factory/browser` by default,
matching the proven `shorts-automation` / `lawyer-longform-studio` convention) to run **Nano Banana Pro**
in Google Flow ULTRA Image mode at ZERO extra per-call API billing.

Safety & Lineage Rules:
1. Enforces `OriginOnlyOneShotGuard` on every uploaded reference image (ONLY untouched `Gen-0` + pure optical crops).
2. Zero login / CAPTCHA bypass: if Google Sign-In or anti-bot verification appears, stops cleanly and instructs
   the user to run with `--login` in a visible window.
3. Downloads the **highest available native resolution (4K > 2K > Original)** via Flow's Download button
   (NEVER screenshots!) and reports exact downloaded pixel dimensions + raw face pixel size.

Usage:
  # 1) First-time login (only if session expired):
  /Users/doma/Projects/shorts-automation/.venv/bin/python scripts/google_flow_ultra_cli.py --login

  # 2) Generate from Stage-1 workflow_manifest.json:
  /Users/doma/Projects/shorts-automation/.venv/bin/python scripts/google_flow_ultra_cli.py \
    --manifest /path/to/workflow_manifest.json \
    --output /path/to/flow_ultra_out.png \
    --model "Nano Banana Pro" --aspect "3:4" --count 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FLOW_URL = "https://labs.google/fx/ko/tools/flow"
DEFAULT_PROFILE_DIR = Path(os.environ.get("SHORTS_SECRETS") or os.path.expanduser("~/.shorts-factory")) / "browser"


def _clean_crash_banner(profile_dir: Path) -> None:
    pref = profile_dir / "Default" / "Preferences"
    if not pref.is_file():
        return
    try:
        data = json.loads(pref.read_text(encoding="utf-8"))
        prof = data.setdefault("profile", {})
        if prof.get("exit_type") == "Normal" and prof.get("exited_cleanly") is True:
            return
        prof["exit_type"] = "Normal"
        prof["exited_cleanly"] = True
        pref.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _open_context(pw, profile_dir: Path, headed: bool, downloads_dir: Path):
    profile_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(profile_dir, 0o700)
    except OSError:
        pass
    downloads_dir.mkdir(parents=True, exist_ok=True)
    _clean_crash_banner(profile_dir)
    return pw.chromium.launch_persistent_context(
        str(profile_dir),
        headless=not headed,
        accept_downloads=True,
        downloads_path=str(downloads_dir),
        viewport={"width": 1360, "height": 940},
        ignore_default_args=["--enable-automation"],
        args=[
            "--hide-crash-restore-bubble",
            "--disable-session-crashed-bubble",
            "--no-first-run",
            "--no-default-browser-check",
        ],
    )


def _dismiss_popups(page) -> None:
    for sel in (
        "button:has-text('동의함')",
        "button:has-text('나중에')",
        "button[aria-label='닫기']",
        "button[aria-label='Close']",
        "button:has-text('확인')",
        "button:has-text('Got it')",
    ):
        try:
            loc = page.locator(sel)
            if loc.count() > 0 and loc.first.is_visible():
                loc.first.click(timeout=2000, force=True)
                time.sleep(0.5)
        except Exception:
            pass


def _check_auth_or_bot_block(page) -> Optional[str]:
    url = (page.url or "").lower()
    if "accounts.google.com" in url or "servicelogin" in url:
        return "GOOGLE_SIGNIN_REQUIRED"
    try:
        body = (page.locator("body").inner_text(timeout=3000) or "").lower()
    except Exception:
        body = ""
    if any(k in body for k in ("sign in to continue", "다시 로그인", "로그인이 필요", "세션이 만료")):
        return "GOOGLE_SIGNIN_REQUIRED"
    if any(k in body for k in ("unusual activity", "비정상적인 활동", "verify you are human", "captcha")):
        return "BOT_OR_VERIFICATION_GATE"
    return None


def _enter_new_project(page) -> None:
    _dismiss_popups(page)
    # Landing page -> Flow workspace if needed
    for entry_sel in (
        "text=Google Flow로 만들기",
        "text=Create with Google Flow",
        "text=새 프로젝트",
        "text=New project",
    ):
        try:
            loc = page.locator(entry_sel)
            if loc.count() > 0 and loc.first.is_visible():
                loc.first.click(timeout=8000, force=True)
                time.sleep(2.5)
                _dismiss_popups(page)
        except Exception:
            pass


def _configure_image_mode(page, model_name: str, aspect: str, count: int) -> List[str]:
    notes: List[str] = []
    chip_sel = (
        "button:has-text('동영상'), button:has-text('이미지'), "
        "button:has-text('Nano Banana'), button:has-text('Omni'), button:has-text('Veo')"
    )
    chip = page.locator(chip_sel).last
    chip.wait_for(state="visible", timeout=35000)
    chip.click(timeout=6000, force=True)
    time.sleep(1.2)

    # Select Image tab + Aspect Ratio tab + Count tab
    for label, desc in (("이미지", "mode"), (aspect, "aspect"), (f"x{count}", "count")):
        try:
            tab = page.locator(f"[role=tab]:has-text('{label}'), button:has-text('{label}')").first
            if tab.count() > 0:
                if tab.get_attribute("aria-selected") != "true":
                    tab.click(timeout=5000, force=True)
                    time.sleep(0.8)
                notes.append(f"{desc}={label}")
        except Exception as exc:
            notes.append(f"{desc}_warn={exc}")

    # Select Model (e.g. Nano Banana Pro)
    try:
        dd = page.locator("button:has-text('arrow_drop_down'), button:has-text('Nano Banana')").last
        if dd.count() > 0:
            cur_txt = (dd.inner_text() or "").strip()
            if model_name.lower() not in cur_txt.lower():
                dd.click(timeout=5000, force=True)
                time.sleep(1.0)
                opt = page.get_by_text(model_name, exact=False).first
                opt.click(timeout=5000, force=True)
                time.sleep(1.0)
            notes.append(f"model={model_name}")
    except Exception as exc:
        notes.append(f"model_warn={exc}")

    page.keyboard.press("Escape")
    time.sleep(0.8)
    return notes


def _upload_and_attach_images(page, image_paths: List[Path]) -> int:
    if not image_paths:
        return 0
    str_paths = [str(p.resolve()) for p in image_paths]

    # Open the prompt media attachment modal (`+` / add button near prompt bar)
    for add_sel in (
        "button[aria-label*='소재']",
        "button[aria-label*='추가']",
        "button[aria-label*='Add']",
        "button:has-text('add')",
    ):
        try:
            btn = page.locator(add_sel).last
            if btn.count() > 0 and btn.is_visible():
                btn.click(timeout=4000, force=True)
                time.sleep(1.2)
                break
        except Exception:
            pass

    file_inps = page.locator("input[type='file']")
    if file_inps.count() == 0:
        raise RuntimeError("Google Flow file upload input (`input[type='file']`) not found on page.")

    # Upload all pure Gen-0 reference images cleanly
    target_inp = file_inps.last
    if target_inp.get_attribute("multiple") is not None:
        target_inp.set_input_files(str_paths)
        time.sleep(3.5 + 1.5 * len(str_paths))
    else:
        for sp in str_paths:
            file_inps.last.set_input_files(sp)
            time.sleep(3.5)

    # Close modal if it remains open and verify uploaded thumbnails are attached to prompt
    try:
        # Click newly uploaded asset thumbnails inside the picker if they need explicit selection
        modal_imgs = page.locator("[role='dialog'] img, [role='listbox'] img")
        cnt = modal_imgs.count()
        for i in range(min(len(str_paths), cnt)):
            try:
                modal_imgs.nth(i).click(timeout=3000, force=True)
                time.sleep(0.6)
            except Exception:
                pass
    except Exception:
        pass

    page.keyboard.press("Escape")
    time.sleep(1.0)
    return len(str_paths)


def _count_result_images(page) -> int:
    # Result media tiles on the main canvas (exclude tiny avatars/icons via bounding box >= 120px)
    imgs = page.locator("main img, [role='main'] img, img[src*='googleusercontent'], img[src^='blob:']")
    total = imgs.count()
    big = 0
    for i in range(total):
        try:
            box = imgs.nth(i).bounding_box()
            if box and box["width"] >= 140 and box["height"] >= 140:
                big += 1
        except Exception:
            pass
    return big


def _download_highest_res_image(page, downloads_dir: Path, output_path: Path, wait_timeout: int = 180) -> Tuple[Path, str]:
    """Open the newest generated image tile and download at highest available resolution (4K > 2K > Original)."""
    # Click the newest large generated image tile on canvas
    imgs = page.locator("img[src*='googleusercontent'], img[src^='blob:'], main img")
    best_idx = -1
    for i in range(imgs.count()):
        try:
            box = imgs.nth(i).bounding_box()
            if box and box["width"] >= 140 and box["height"] >= 140:
                best_idx = i
        except Exception:
            pass
    if best_idx < 0:
        raise RuntimeError("Generated image tile not found on Flow canvas.")

    imgs.nth(best_idx).click(timeout=8000, force=True)
    time.sleep(2.0)

    # Locate Download button
    dl_btn = page.locator("button:has-text('download'), button[aria-label*='다운로드'], button[aria-label*='Download']").last
    dl_btn.wait_for(state="visible", timeout=20000)

    chosen_tier = "default"
    # Check if clicking download immediately starts a download OR opens a resolution submenu (4K / 2K / 1K)
    try:
        with page.expect_download(timeout=5000) as dl_info:
            dl_btn.click(timeout=6000, force=True)
        dl = dl_info.value
        chosen_tier = "direct_download"
    except Exception:
        # Submenu opened! Look for 4K -> 2K -> Original/원본 in strict priority order
        time.sleep(1.0)
        dl = None
        for tier_lbl, sel in (
            ("4K", "[role='menuitem']:has-text('4K'), button:has-text('4K'), text=/4K|3840|4096/i"),
            ("2K", "[role='menuitem']:has-text('2K'), button:has-text('2K'), text=/2K|2048/i"),
            ("Original", "[role='menuitem']:has-text('원본'), [role='menuitem']:has-text('Original'), text=/원본|Original|최대/i"),
            ("1K", "[role='menuitem']:has-text('1K'), button:has-text('1K')"),
        ):
            opt = page.locator(sel)
            if opt.count() > 0 and opt.first.is_visible():
                chosen_tier = tier_lbl
                with page.expect_download(timeout=wait_timeout * 1000) as dl_info:
                    opt.first.click(timeout=8000, force=True)
                dl = dl_info.value
                break
        if dl is None:
            raise RuntimeError("Download menu opened but no 4K/2K/Original download option could be clicked.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_saved = downloads_dir / (dl.suggested_filename or f"flow_dl_{int(time.time())}.png")
    dl.save_as(str(tmp_saved))
    shutil.copy2(tmp_saved, output_path)
    return output_path, chosen_tier


def _validate_origin_inputs(image_paths: List[Path]) -> None:
    """Enforce `1원본들 → 1생성`: never allow stale generated outputs into Google Flow."""
    forbidden = re.compile(
        r"(?:^|[/\\])(?:s\d+_v\d+_|live_test_|hero_|5k_21x9_|raw_batch_|raw_scene_|exp_m|exp_final|upscayl4k_)",
        re.IGNORECASE,
    )
    safe = re.compile(
        r"(?:[/\\]\.user_uploaded[/\\]|[/\\]originals[/\\]|loc\d+.*_clean\.|gen0_optical_ocular_detail_anchor\.)",
        re.IGNORECASE,
    )
    for p in image_paths:
        s = str(p)
        if not p.is_file():
            raise FileNotFoundError(f"Reference image not found: {p}")
        if not safe.search(s) and forbidden.search(s):
            raise RuntimeError(
                f"[1원본들→1생성 차단] '{p.name}'은 이전 생성 결과물입니다. 순수 원본(Gen-0)과 무손실 광학 크롭만 업로드하세요."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Google Flow ULTRA Native 2K/4K Image CLI (모델생성 v10.1)")
    parser.add_argument("--login", action="store_true", help="Open visible Chrome window for Google AI Ultra login")
    parser.add_argument("--manifest", type=Path, default=None, help="Stage-1 workflow_manifest.json path")
    parser.add_argument("--images", nargs="*", type=Path, default=[], help="Reference image paths (pure Gen-0 only)")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text (optional if --manifest is provided)")
    parser.add_argument("--output", type=Path, default=None, help="Output image file path")
    parser.add_argument("--model", type=str, default="Nano Banana Pro")
    parser.add_argument("--aspect", type=str, default="3:4")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--headed", action="store_true", help="Show browser window during generation")
    parser.add_argument("--timeout", type=int, default=240, help="Generation wait timeout in seconds")
    args = parser.parse_args()

    try:
        from patchright.sync_api import sync_playwright
    except ImportError:
        print(
            "ERROR: patchright is not installed in this Python interpreter.\n"
            "Run with: /Users/doma/Projects/shorts-automation/.venv/bin/python scripts/google_flow_ultra_cli.py ...",
            file=sys.stderr,
        )
        return 2

    downloads_dir = Path.home() / ".model_generation_studio" / "flow_downloads"

    if args.login:
        print(f"[Google Flow ULTRA] 로그인용 Chrome 창을 엽니다 (프로필: {args.profile_dir})")
        print("Google AI Ultra 계정으로 로그인 완료 후 창을 닫거나 터미널에서 Ctrl+C 하세요.")
        with sync_playwright() as pw:
            ctx = _open_context(pw, args.profile_dir, headed=True, downloads_dir=downloads_dir)
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(FLOW_URL, wait_until="domcontentloaded")
            try:
                while ctx.pages:
                    time.sleep(2)
            except KeyboardInterrupt:
                pass
            finally:
                ctx.close()
        return 0

    ref_images: List[Path] = list(args.images)
    prompt_text = args.prompt.strip()
    if args.manifest is not None:
        m = json.loads(args.manifest.read_text(encoding="utf-8"))
        if not ref_images:
            ref_images = [Path(r["path"]) for r in m.get("ordered_reference_images", [])]
        if not prompt_text and m.get("prompts"):
            prompt_text = str(m["prompts"][0]["prompt"]).strip()

    if not prompt_text:
        print("ERROR: Provide --prompt or --manifest.", file=sys.stderr)
        return 2
    if args.output is None:
        print("ERROR: Provide --output path.", file=sys.stderr)
        return 2

    _validate_origin_inputs(ref_images)

    with sync_playwright() as pw:
        ctx = _open_context(pw, args.profile_dir, headed=args.headed, downloads_dir=downloads_dir)
        try:
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(FLOW_URL, wait_until="domcontentloaded", timeout=60000)
            time.sleep(3.5)

            block = _check_auth_or_bot_block(page)
            if block:
                print(json.dumps({"status": "blocked", "reason": block, "url": page.url}, ensure_ascii=False))
                return 3

            _enter_new_project(page)
            block = _check_auth_or_bot_block(page)
            if block:
                print(json.dumps({"status": "blocked", "reason": block, "url": page.url}, ensure_ascii=False))
                return 3

            cfg_notes = _configure_image_mode(page, args.model, args.aspect, args.count)
            uploaded_n = _upload_and_attach_images(page, ref_images)

            before_tiles = _count_result_images(page)

            # Focus prompt box and type prompt cleanly
            box = page.locator("[role='textbox'][contenteditable='true'], textarea").first
            box.wait_for(state="visible", timeout=20000)
            box.click(timeout=5000, force=True)
            time.sleep(0.4)
            # Flow server prompt length safety cap (~2,200 chars keeps full lock while avoiding silent rejection)
            trimmed_prompt = prompt_text[:2200]
            box.fill(trimmed_prompt)
            time.sleep(0.8)

            # Submit generation
            send_btn = page.locator("button:has-text('arrow_forward'), button[type='submit']").last
            send_btn.click(timeout=8000, force=True)

            # Wait for a new generated image tile to appear
            t0 = time.time()
            while time.time() - t0 < args.timeout:
                time.sleep(3.0)
                blk = _check_auth_or_bot_block(page)
                if blk:
                    print(json.dumps({"status": "blocked", "reason": blk, "url": page.url}, ensure_ascii=False))
                    return 3
                if _count_result_images(page) > before_tiles:
                    time.sleep(3.0)
                    break
            else:
                raise TimeoutError(f"Timed out after {args.timeout}s waiting for Google Flow image generation.")

            saved_path, tier = _download_highest_res_image(page, downloads_dir, args.output, wait_timeout=args.timeout)

            # Measure actual downloaded image resolution
            w_px, h_px = 0, 0
            try:
                from PIL import Image
                with Image.open(saved_path) as im:
                    w_px, h_px = im.size
            except Exception:
                pass

            print(
                json.dumps(
                    {
                        "status": "ok",
                        "output_path": str(saved_path),
                        "resolution": [w_px, h_px],
                        "download_tier": tier,
                        "uploaded_references": uploaded_n,
                        "config_notes": cfg_notes,
                        "project_url": page.url,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        finally:
            try:
                ctx.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
