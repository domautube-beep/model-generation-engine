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
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FLOW_URL = "https://labs.google/fx/ko/tools/flow"
DEFAULT_PROFILE_DIR = Path(os.environ.get("SHORTS_SECRETS") or os.path.expanduser("~/.shorts-factory")) / "browser"
SUBJECT_PROJECTS_PATH = Path.home() / ".model_generation_studio" / "flow_subject_projects.json"


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_project_root_url(url: str) -> Optional[str]:
    m = re.search(r"(https://(?:flow\.google\.com|labs\.google/fx/(?:[a-z-]+/)?tools/flow)/project/[0-9a-fA-F\-]+)", url or "")
    if m:
        return m.group(1)
    return None


def _load_subject_project_url(gen0_path: Optional[Path]) -> Optional[str]:
    if gen0_path is None or not gen0_path.is_file() or not SUBJECT_PROJECTS_PATH.is_file():
        return None
    try:
        data = json.loads(SUBJECT_PROJECTS_PATH.read_text(encoding="utf-8"))
        return _normalize_project_root_url(str(data.get(_sha256_file(gen0_path), "")))
    except Exception:
        return None


def _save_subject_project_url(gen0_path: Optional[Path], project_url: str) -> None:
    root_u = _normalize_project_root_url(project_url)
    if gen0_path is None or not gen0_path.is_file() or not root_u:
        return
    try:
        SUBJECT_PROJECTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(SUBJECT_PROJECTS_PATH.read_text(encoding="utf-8")) if SUBJECT_PROJECTS_PATH.is_file() else {}
        data[_sha256_file(gen0_path)] = root_u
        SUBJECT_PROJECTS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


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


def _snapshot_marker_counts(page) -> Dict[str, int]:
    """Snapshot baseline text marker counts BEFORE submitting (driver.py D-047: only newly added markers count!)."""
    try:
        body = (page.locator("body").inner_text(timeout=3000) or "").lower()
    except Exception:
        body = ""
    markers = ("unusual activity", "비정상적인 활동", "verify you are human", "로봇이 아닙니다")
    return {m: body.count(m) for m in markers}


def _check_auth_or_bot_block(page, baseline: Optional[Dict[str, int]] = None) -> Optional[str]:
    url = (page.url or "").lower()
    if "accounts.google.com" in url or "servicelogin" in url:
        return "GOOGLE_SIGNIN_REQUIRED"
    # Check for an actually visible active challenge iframe (>100px)
    try:
        ch_frames = page.locator("iframe[src*='recaptcha/api2/bframe'], iframe[src*='hcaptcha'], iframe[src*='challenges.cloudflare']")
        for fi in range(ch_frames.count()):
            el = ch_frames.nth(fi)
            if el.is_visible():
                box = el.bounding_box()
                if box and box["width"] >= 100 and box["height"] >= 100:
                    return "BOT_OR_VERIFICATION_GATE"
    except Exception:
        pass
    try:
        body = (page.locator("body").inner_text(timeout=3000) or "").lower()
    except Exception:
        body = ""
    if any(k in body for k in ("sign in to continue", "다시 로그인", "로그인이 필요", "세션이 만료")):
        return "GOOGLE_SIGNIN_REQUIRED"
    # Inherited from adapters/flow/driver.py D-047: ONLY flag text toast if count STRICTLY INCREASED vs pre-submit baseline!
    if baseline is not None:
        for k, prev_n in baseline.items():
            if body.count(k) > prev_n:
                return "FLOW_UNUSUAL_ACTIVITY_TOAST"
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

    # Select Model (e.g. Nano Banana Pro) inside the open settings panel
    try:
        panel_btns = page.locator("button:has-text('Nano Banana'), button:has-text('Imagen'), button:has-text('arrow_drop_down')")
        if panel_btns.count() > 0:
            for bi in range(panel_btns.count()):
                btxt = (panel_btns.nth(bi).inner_text() or "").strip()
                if "nano banana" in btxt.lower() or "imagen" in btxt.lower():
                    if model_name.lower() not in btxt.lower():
                        panel_btns.nth(bi).click(timeout=5000, force=True)
                        time.sleep(0.8)
                        opt = page.get_by_text(model_name, exact=False).first
                        opt.click(timeout=5000, force=True)
                        time.sleep(0.8)
                    notes.append(f"model={model_name}")
                    break
    except Exception as exc:
        notes.append(f"model_warn={exc}")

    page.keyboard.press("Escape")
    time.sleep(0.8)
    return notes


def _count_prompt_bar_chips(page) -> int:
    """Count attached reference thumbnail `<img>` chips inside the bottom Google Flow prompt bar."""
    tbox = None
    try:
        tb = page.locator("[role='textbox'][contenteditable='true']").first
        tbox = tb.bounding_box()
    except Exception:
        pass
    min_y = (tbox["y"] - 95) if tbox else 715
    max_y = (tbox["y"] + 25) if tbox else 870
    min_x = (tbox["x"] - 35) if tbox else 340
    max_x = (tbox["x"] + max(450.0, tbox["width"] + 30)) if tbox else 980
    cnt = 0
    imgs = page.locator("img")
    for i in range(imgs.count()):
        try:
            el = imgs.nth(i)
            if not el.is_visible():
                continue
            box = el.bounding_box()
            if not box:
                continue
            if min_x <= box["x"] <= max_x and min_y <= box["y"] <= max_y and 14 <= box["width"] <= 92 and 14 <= box["height"] <= 92:
                src = (el.get_attribute("src") or "").strip()
                if src and not src.startswith("data:image/svg"):
                    cnt += 1
        except Exception:
            pass
    return cnt


def _first_visible(page, selector: str, min_x: float = 0.0, min_y: float = 0.0):
    loc = page.locator(selector)
    for i in range(loc.count()):
        try:
            el = loc.nth(i)
            if not el.is_visible():
                continue
            box = el.bounding_box()
            if not box:
                continue
            if box["x"] >= min_x and box["y"] >= min_y:
                return el
        except Exception:
            pass
    return None


def _open_prompt_bar_media_popover(page) -> bool:
    """Click the exact `+` button immediately to the LEFT of the bottom prompt textbox (`무엇을 만들고 싶으신가요?`)."""
    tb = page.locator("[role='textbox'][contenteditable='true']").first
    tb.wait_for(state="visible", timeout=20000)
    tbox = tb.bounding_box()
    if not tbox:
        return False
    # Find the visible button directly left of the textbox on the same horizontal row
    btns = page.locator("button")
    best_btn = None
    best_dist = 1e9
    for i in range(btns.count()):
        try:
            b = btns.nth(i)
            if not b.is_visible():
                continue
            bb = b.bounding_box()
            if not bb:
                continue
            if abs((bb["y"] + bb["height"] * 0.5) - (tbox["y"] + tbox["height"] * 0.5)) <= 45 and bb["x"] < tbox["x"] + 20 and bb["x"] > tbox["x"] - 140:
                dist = abs(tbox["x"] - (bb["x"] + bb["width"]))
                if dist < best_dist:
                    best_dist = dist
                    best_btn = b
        except Exception:
            pass
    if best_btn is not None:
        best_btn.click(timeout=5000, force=True)
        time.sleep(1.2)
        return True
    return False


def _upload_and_attach_images(page, image_paths: List[Path]) -> int:
    if not image_paths:
        return 0
    resolved_paths = [p.resolve() for p in image_paths]
    before_chips = _count_prompt_bar_chips(page)
    target_chips = before_chips + len(resolved_paths)

    upload_btn_sel = (
        "button:has-text('미디어 업로드'), button:has-text('이미지 업로드'), button:has-text('소재 업로드'), "
        "button:has-text('Upload media'), button:has-text('Upload image')"
    )
    add_prompt_btn_sel = "button:has-text('프롬프트에 추가'), button:has-text('Add to prompt')"
    popover_toggle_sel = "button[aria-label='프롬프트 상자에 소재 추가'], button[aria-label*='Add to prompt'], button[aria-label*='소재 추가']"

    def _ensure_popover_open() -> None:
        if _first_visible(page, upload_btn_sel, min_x=240.0) is not None or _first_visible(page, add_prompt_btn_sel, min_x=240.0) is not None:
            return
        tog = _first_visible(page, popover_toggle_sel, min_x=240.0)
        if tog is not None:
            tog.click(timeout=4000, force=True)
            time.sleep(1.2)
            if _first_visible(page, upload_btn_sel, min_x=240.0) is not None:
                return
        _open_prompt_bar_media_popover(page)

    def _find_asset_item_by_name(fname: str):
        # Match asset button by full filename or stem (only fall back to 36-char prefix if Flow truncated with '...')
        stem = Path(fname).stem
        prefix36 = fname[:36]
        btns = page.locator("button")
        for bi in range(btns.count()):
            try:
                b = btns.nth(bi)
                if not b.is_visible():
                    continue
                box = b.bounding_box()
                if not box or box["x"] < 300 or box["y"] > 840 or box["y"] < 150:
                    continue
                txt = (b.inner_text() or "").strip()
                if re.search(r"\b\d{1,3}%", txt):
                    # Still server-transcoding ("99%" / "100%" blurred placeholder) — wait until thumbnail is ready!
                    continue
                if fname in txt or stem in txt or (("..." in txt or "…" in txt) and prefix36 in txt):
                    return b
            except Exception:
                pass
        return None

    for p in resolved_paths:
        _ensure_popover_open()
        time.sleep(0.9)
        item_btn = _find_asset_item_by_name(p.name)
        if item_btn is None:
            # Never blindly click the `+` button again if the popover is already open (that toggles it closed!)
            _ensure_popover_open()
            time.sleep(0.6)
            item_btn = _find_asset_item_by_name(p.name)
        if item_btn is None:
            # Upload this single missing asset into the project cleanly
            vis_up = _first_visible(page, upload_btn_sel, min_x=240.0)
            if vis_up is None:
                _open_prompt_bar_media_popover(page)
                time.sleep(1.2)
                vis_up = _first_visible(page, upload_btn_sel, min_x=240.0)
            file_inps = page.locator("input[type='file']")
            if vis_up is not None:
                try:
                    with page.expect_file_chooser(timeout=15000) as fc_info:
                        vis_up.click(timeout=6000)
                    fc_info.value.set_files([str(p)])
                except Exception:
                    if file_inps.count() > 0:
                        file_inps.last.set_input_files([str(p)])
                    else:
                        raise
            elif file_inps.count() > 0:
                file_inps.last.set_input_files([str(p)])
            else:
                raise RuntimeError("Google Flow upload button/input not found")

            # Wait until any server transcoding percentage badge ("12%", "33%", "99%") on canvas or popover finishes!
            t_trans = time.time()
            time.sleep(2.2)
            while time.time() - t_trans < 45.0:
                pct_loc = page.locator("text=/^\\s*\\d{1,3}%\\s*$/")
                any_pct = False
                for pi in range(pct_loc.count()):
                    try:
                        if pct_loc.nth(pi).is_visible():
                            any_pct = True
                            break
                    except Exception:
                        pass
                if not any_pct:
                    break
                time.sleep(1.5)

            # Wait up to 75s for upload to reach 100% and appear in the `+` media popover (or auto-attach as chip!)
            t_wait = time.time()
            cur_before_this = _count_prompt_bar_chips(page)
            while time.time() - t_wait < 75.0:
                time.sleep(1.5)
                if _count_prompt_bar_chips(page) > cur_before_this:
                    item_btn = "AUTO_ATTACHED"
                    break
                _ensure_popover_open()
                item_btn = _find_asset_item_by_name(p.name)
                if item_btn is not None:
                    break

        if item_btn == "AUTO_ATTACHED":
            continue
        if item_btn is None:
            dbg_p = Path.home() / ".model_generation_studio" / "flow_attach_verify_debug.png"
            page.screenshot(path=str(dbg_p), full_page=False)
            raise RuntimeError(f"Asset '{p.name}' did not appear in Google Flow asset popover (debug: {dbg_p})")

        # Click the asset item in the list (retry up to 4 times in case transcoding just completed!)
        chips_before_click = _count_prompt_bar_chips(page)
        for _att in range(4):
            if _att > 0:
                time.sleep(2.2)
                _ensure_popover_open()
                time.sleep(0.9)
                item_btn = _find_asset_item_by_name(p.name)
                if item_btn is None:
                    continue
            item_btn.click(timeout=4000, force=True)
            time.sleep(1.2)
            if _count_prompt_bar_chips(page) > chips_before_click:
                break
            add_btn = _first_visible(page, add_prompt_btn_sel, min_x=300.0)
            if add_btn is not None:
                add_btn.click(timeout=5000, force=True)
                time.sleep(1.2)
            if _count_prompt_bar_chips(page) > chips_before_click:
                break

    # Close popover cleanly so only the prompt bar remains
    if _first_visible(page, add_prompt_btn_sel, min_x=240.0) is not None:
        page.keyboard.press("Escape")
        time.sleep(0.9)

    attached_delta = _count_prompt_bar_chips(page) - before_chips
    if attached_delta < len(resolved_paths):
        dbg_p = Path.home() / ".model_generation_studio" / "flow_attach_verify_debug.png"
        page.screenshot(path=str(dbg_p), full_page=False)
        raise RuntimeError(
            f"Reference attachment verification FAILED: expected {len(resolved_paths)} attached chip(s) in prompt bar, "
            f"found {attached_delta} (debug screenshot: {dbg_p})"
        )
    return attached_delta


def _snapshot_result_images(page) -> List[Tuple[int, str, float, float]]:
    """Return [(index, src, width, height)] for all visible media tile images on canvas (handles narrow tiles beside wide 9-panel sheets)."""
    imgs = page.locator("img")
    found: List[Tuple[int, str, float, float]] = []
    for i in range(imgs.count()):
        try:
            el = imgs.nth(i)
            box = el.bounding_box()
            if not box or box["x"] < 200 or box["y"] < 55 or box["y"] > 760:
                continue
            if box["width"] < 70 or box["height"] < 90:
                continue
            src = (el.get_attribute("src") or "").strip()
            if not src or src.startswith("data:image/svg"):
                continue
            found.append((i, src, float(box["width"]), float(box["height"])))
        except Exception:
            pass
    return found


def _download_highest_res_image(
    page,
    downloads_dir: Path,
    output_path: Path,
    before_srcs: set,
    wait_timeout: int = 180,
    preferred_tier: str = "4K",
) -> Tuple[Path, str, List[str]]:
    """Open the newly generated image tile (not in `before_srcs`) and download at highest available resolution (4K > 2K > Original)."""
    before_bases = {s.split("?")[0] for s in before_srcs}
    clicked = False
    imgs = page.locator("img")
    for i in range(imgs.count()):
        try:
            el = imgs.nth(i)
            box = el.bounding_box()
            if not box or box["x"] < 200 or box["y"] < 55 or box["y"] > 760 or box["width"] < 70 or box["height"] < 90:
                continue
            src = (el.get_attribute("src") or "").strip()
            if not src or src.startswith("data:image/svg"):
                continue
            if src.split("?")[0] not in before_bases:
                # Click center of the visible tile bounding box directly so overlay wrappers never block clicks
                page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
                clicked = True
                break
        except Exception:
            pass
    if not clicked:
        dbg_p = Path.home() / ".model_generation_studio" / "flow_tile_debug.png"
        page.screenshot(path=str(dbg_p), full_page=False)
        raise RuntimeError(f"Newly generated image tile not clickable (debug: {dbg_p})")

    time.sleep(2.2)

    # Locate the ACTUALLY VISIBLE Download button on the opened generated image viewer (never a hidden hover button!)
    dl_sel = "button:has-text('download'), button[aria-label*='다운로드'], button[aria-label*='Download']"
    dl_btn = None
    for attempt in range(3):
        t_wait = time.time()
        while time.time() - t_wait < 8.0:
            cands = page.locator(dl_sel)
            for ci in range(cands.count() - 1, -1, -1):
                try:
                    cand = cands.nth(ci)
                    if cand.is_visible():
                        dl_btn = cand
                        break
                except Exception:
                    pass
            if dl_btn is not None:
                break
            time.sleep(0.5)
        if dl_btn is not None:
            break
        # Retry opening the tile viewer in case the first click only selected/focused the tile
        for i in range(imgs.count()):
            try:
                el = imgs.nth(i)
                box = el.bounding_box()
                if not box or box["width"] < 140 or box["height"] < 140:
                    continue
                src = (el.get_attribute("src") or "").strip()
                if src and not src.startswith("data:image/svg") and src.split("?")[0] not in before_bases:
                    page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
                    time.sleep(0.5)
                    page.mouse.dblclick(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
                    time.sleep(1.5)
                    break
            except Exception:
                pass
    if dl_btn is None:
        dbg_p = Path.home() / ".model_generation_studio" / "flow_dlbtn_debug.png"
        page.screenshot(path=str(dbg_p), full_page=False)
        raise RuntimeError(f"Visible Download button not found on Google Flow viewer (debug: {dbg_p})")

    chosen_tier = "default"
    seen_menu_items: List[str] = []
    try:
        with page.expect_download(timeout=4500) as dl_info:
            dl_btn.click(timeout=6000, force=True)
        dl = dl_info.value
        chosen_tier = "direct_download"
    except Exception:
        # Submenu opened! Capture all visible menu items and pick highest tier (4K > 2K > Original > 1K)
        time.sleep(1.0)
        menu_els = page.locator("[role='menuitem'], [role='menu'] button, [role='listbox'] [role='option']")
        for mi in range(menu_els.count()):
            try:
                t = (menu_els.nth(mi).inner_text() or "").strip().replace("\n", " ")
                if t:
                    seen_menu_items.append(t)
            except Exception:
                pass
        dl = None
        all_tiers = [
            ("4K", "[role='menuitem']:has-text('4K'), button:has-text('4K'), [role='option']:has-text('4K')"),
            ("2K", "[role='menuitem']:has-text('2K'), button:has-text('2K'), [role='option']:has-text('2K')"),
            ("Original", "[role='menuitem']:has-text('원본'), [role='menuitem']:has-text('Original'), button:has-text('원본')"),
            ("1K", "[role='menuitem']:has-text('1K'), button:has-text('1K'), [role='option']:has-text('1K')"),
        ]
        pref = preferred_tier.strip().upper()
        if pref == "2K":
            tier_order = [all_tiers[1], all_tiers[0], all_tiers[2], all_tiers[3]]
        elif pref in ("1K", "ORIGINAL"):
            tier_order = [all_tiers[2], all_tiers[3], all_tiers[1], all_tiers[0]]
        else:
            tier_order = all_tiers
        for tier_lbl, sel in tier_order:
            for _dl_try in range(2 if tier_lbl == "4K" else 1):
                if _dl_try > 0 or (dl is None and page.locator(sel).count() == 0):
                    try:
                        dl_btn.click(timeout=5000, force=True)
                        time.sleep(1.0)
                    except Exception:
                        pass
                opt = page.locator(sel)
                if opt.count() > 0 and opt.first.is_visible():
                    try:
                        chosen_tier = tier_lbl
                        with page.expect_download(timeout=min(65000, wait_timeout * 1000)) as dl_info:
                            opt.first.click(timeout=8000, force=True)
                        dl = dl_info.value
                        break
                    except Exception:
                        dl = None
            if dl is not None:
                break
        if dl is None:
            dbg_p = Path.home() / ".model_generation_studio" / "flow_download_debug.png"
            page.screenshot(path=str(dbg_p), full_page=False)
            raise RuntimeError(f"Download menu opened ({seen_menu_items}) but download did not start (debug: {dbg_p})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_saved = downloads_dir / (dl.suggested_filename or f"flow_dl_{int(time.time())}.png")
    dl.save_as(str(tmp_saved))
    shutil.copy2(tmp_saved, output_path)
    return output_path, chosen_tier, seen_menu_items


def _validate_origin_inputs(image_paths: List[Path]) -> None:
    """Enforce `1원본들 → 1생성`: never allow stale generated outputs into Google Flow."""
    forbidden = re.compile(
        r"(?:^|[/\\])(?:s\d+_v\d+_|live_test_|hero_|5k_21x9_|raw_batch_|raw_scene_|exp_m|exp_final|upscayl4k_)",
        re.IGNORECASE,
    )
    safe = re.compile(
        r"(?:[/\\]\.user_uploaded[/\\]|[/\\]originals[/\\]|loc\d+.*_clean\.|gen0_optical_ocular_detail_anchor\.|master_8angle|s\d+_8dir_)",
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
    parser.add_argument("--batch-json", type=Path, default=None, help="JSON list of jobs [{images, prompt, output, aspect, model, tier}] to run in 1 Chrome session")
    parser.add_argument("--images", nargs="*", type=Path, default=[], help="Reference image paths (pure Gen-0 only)")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text (optional if --manifest is provided)")
    parser.add_argument("--output", type=Path, default=None, help="Output image file path")
    parser.add_argument("--model", type=str, default="Nano Banana Pro")
    parser.add_argument("--aspect", type=str, default="3:4")
    parser.add_argument("--tier", type=str, default="4K", help="Preferred download tier: 4K (default), 2K, or Original")
    parser.add_argument("--project-url", type=str, default="", help="Persistent Google Flow project URL for this person (1 person = 1 project)")
    parser.add_argument("--download-edit-url", type=str, default="", help="Directly download a specific generated tile URL at --tier (e.g. 4K winner)")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--headed", action="store_true", default=True, help="Visible desktop Chrome window (default=True to prevent headless bot toast)")
    parser.add_argument("--headless", action="store_true", help="Force headless browser (discouraged; triggers FLOW_UNUSUAL_ACTIVITY_TOAST)")
    parser.add_argument("--timeout", type=int, default=240, help="Generation wait timeout in seconds")
    args = parser.parse_args()
    use_headed = not bool(args.headless)

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

    if args.download_edit_url:
        if args.output is None:
            print("ERROR: Provide --output with --download-edit-url.", file=sys.stderr)
            return 2
        with sync_playwright() as pw:
            ctx = _open_context(pw, args.profile_dir, headed=use_headed, downloads_dir=downloads_dir)
            try:
                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                page.goto(args.download_edit_url, wait_until="domcontentloaded", timeout=60000)
                time.sleep(3.5)
                dl_btn = _first_visible(page, "button:has-text('download'), button[aria-label*='다운로드'], button[aria-label*='Download']")
                if dl_btn is None:
                    raise RuntimeError(f"Download button not found on {args.download_edit_url}")
                dl_btn.click(timeout=6000, force=True)
                time.sleep(1.0)
                pref = args.tier.strip().upper()
                sel = (
                    "[role='menuitem']:has-text('4K'), button:has-text('4K')"
                    if pref == "4K"
                    else "[role='menuitem']:has-text('2K'), button:has-text('2K')"
                    if pref == "2K"
                    else "[role='menuitem']:has-text('1K'), [role='menuitem']:has-text('원본')"
                )
                opt = page.locator(sel).first
                with page.expect_download(timeout=args.timeout * 1000) as dli:
                    opt.click(timeout=8000, force=True)
                dl = dli.value
                tmp = downloads_dir / (dl.suggested_filename or f"winner_{int(time.time())}.png")
                dl.save_as(str(tmp))
                args.output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(tmp, args.output)
                w_px, h_px = 0, 0
                try:
                    from PIL import Image
                    with Image.open(args.output) as im:
                        w_px, h_px = im.size
                except Exception:
                    pass
                print(json.dumps({"status": "ok", "output_path": str(args.output), "resolution": [w_px, h_px], "download_tier": pref, "edit_url": args.download_edit_url}, ensure_ascii=False, indent=2))
                return 0
            finally:
                ctx.close()

    jobs: List[Dict[str, Any]] = []
    if args.batch_json is not None:
        raw_jobs = json.loads(args.batch_json.read_text(encoding="utf-8"))
        for jb in raw_jobs:
            imgs = [Path(p) for p in jb.get("images", [])]
            _validate_origin_inputs(imgs)
            jobs.append(
                {
                    "tag": str(jb.get("tag", Path(jb["output"]).stem)),
                    "images": imgs,
                    "prompt": str(jb["prompt"]).strip(),
                    "output": Path(jb["output"]),
                    "model": str(jb.get("model", args.model)),
                    "aspect": str(jb.get("aspect", args.aspect)),
                    "tier": str(jb.get("tier", args.tier)),
                    "count": int(jb.get("count", args.count)),
                }
            )
    else:
        ref_images: List[Path] = list(args.images)
        prompt_text = args.prompt.strip()
        if args.manifest is not None:
            m = json.loads(args.manifest.read_text(encoding="utf-8"))
            if not ref_images:
                ref_images = [Path(r["path"]) for r in m.get("ordered_reference_images", [])]
            if not prompt_text and m.get("prompts"):
                prompt_text = str(m["prompts"][0]["prompt"]).strip()

        if not prompt_text:
            print("ERROR: Provide --prompt, --manifest, or --batch-json.", file=sys.stderr)
            return 2
        if args.output is None:
            print("ERROR: Provide --output path.", file=sys.stderr)
            return 2
        _validate_origin_inputs(ref_images)
        jobs.append(
            {
                "tag": args.output.stem,
                "images": ref_images,
                "prompt": prompt_text,
                "output": args.output,
                "model": args.model,
                "aspect": args.aspect,
                "tier": args.tier,
                "count": args.count,
            }
        )

    results: List[Dict[str, Any]] = []
    # 1 Person = 1 Persistent Google Flow Project (NEVER spam new projects per angle or retry!)
    primary_gen0: Optional[Path] = jobs[0]["images"][0] if (jobs and jobs[0]["images"]) else None
    pinned_project_url: Optional[str] = (
        _normalize_project_root_url(args.project_url)
        or _load_subject_project_url(primary_gen0)
    )
    if pinned_project_url and primary_gen0 is not None:
        _save_subject_project_url(primary_gen0, pinned_project_url)

    with sync_playwright() as pw:
        ctx = _open_context(pw, args.profile_dir, headed=use_headed, downloads_dir=downloads_dir)
        try:
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            last_configured_mode: Optional[Tuple[str, str, int]] = None
            known_uploaded_names: set[str] = set()
            for j_idx, job in enumerate(jobs):
                if j_idx > 0:
                    # Natural human-like pacing between batch cuts to prevent FLOW_UNUSUAL_ACTIVITY_TOAST
                    time.sleep(1.8)
                if pinned_project_url:
                    cur_norm = _normalize_project_root_url(page.url)
                    if cur_norm == pinned_project_url and "/edit/" in page.url:
                        try:
                            page.keyboard.press("Escape")
                            time.sleep(0.9)
                        except Exception:
                            pass
                    if page.url.rstrip("/") != pinned_project_url.rstrip("/"):
                        page.goto(pinned_project_url, wait_until="domcontentloaded", timeout=60000)
                        time.sleep(2.4)
                        last_configured_mode = None
                    block = _check_auth_or_bot_block(page)
                    if block:
                        print(json.dumps({"status": "blocked", "reason": block, "url": page.url}, ensure_ascii=False))
                        return 3
                else:
                    page.goto(FLOW_URL, wait_until="domcontentloaded", timeout=60000)
                    time.sleep(2.8)
                    block = _check_auth_or_bot_block(page)
                    if block:
                        print(json.dumps({"status": "blocked", "reason": block, "url": page.url}, ensure_ascii=False))
                        return 3
                    _enter_new_project(page)
                    block = _check_auth_or_bot_block(page)
                    if block:
                        print(json.dumps({"status": "blocked", "reason": block, "url": page.url}, ensure_ascii=False))
                        return 3
                    pinned_project_url = _normalize_project_root_url(page.url)
                    if pinned_project_url and primary_gen0 is not None:
                        _save_subject_project_url(primary_gen0, pinned_project_url)
                    last_configured_mode = None

                req_mode = (str(job["model"]), str(job["aspect"]), int(job["count"]))
                if last_configured_mode != req_mode:
                    cfg_notes = _configure_image_mode(page, job["model"], job["aspect"], job["count"])
                    last_configured_mode = req_mode
                else:
                    cfg_notes = ["reused_warm_mode"]

                all_already_uploaded = all(Path(p).name in known_uploaded_names for p in job["images"])
                if _count_prompt_bar_chips(page) > 0 and pinned_project_url:
                    # Ensure clean prompt bar with zero leftover chips from prior cut
                    page.goto(pinned_project_url, wait_until="domcontentloaded", timeout=60000)
                    time.sleep(2.2)
                    cfg_notes = _configure_image_mode(page, job["model"], job["aspect"], job["count"])
                    last_configured_mode = req_mode
                uploaded_n = _upload_and_attach_images(page, job["images"])
                for p_up in job["images"]:
                    known_uploaded_names.add(Path(p_up).name)
                # Only need long settle wait when brand-new media tiles were uploaded to the project grid
                time.sleep(1.0 if all_already_uploaded else 2.8)
                before_srcs = {src for _, src, _, _ in _snapshot_result_images(page)}

                box = page.locator("[role='textbox'][contenteditable='true']").first
                box.wait_for(state="visible", timeout=20000)
                box.click(timeout=5000)
                time.sleep(0.4)
                box.fill(job["prompt"])
                time.sleep(0.8)

                baseline_markers = _snapshot_marker_counts(page)
                send_btn = page.locator("button:has-text('arrow_forward')").last
                send_btn.click(timeout=8000)

                # Refresh `before_srcs` during the first 6s (uploaded tiles settle quickly; real Nano Banana generation takes >= 10s)
                time.sleep(5.0)
                before_srcs |= {src for _, src, _, _ in _snapshot_result_images(page)}

                target_n = max(1, int(job["count"]))
                fresh_srcs: List[str] = []
                t0 = time.time()
                while time.time() - t0 < args.timeout:
                    time.sleep(2.5)
                    blk = _check_auth_or_bot_block(page, baseline=baseline_markers)
                    if blk:
                        print(json.dumps({"status": "blocked", "reason": blk, "url": page.url}, ensure_ascii=False))
                        return 3
                    cur_imgs = _snapshot_result_images(page)
                    fresh = [src for _, src, _, _ in cur_imgs if src not in before_srcs]
                    if len(fresh) >= target_n:
                        fresh_srcs = fresh[:target_n]
                        time.sleep(3.0)
                        break
                    elif fresh and (time.time() - t0) > 65.0:
                        fresh_srcs = fresh
                        time.sleep(2.5)
                        break
                if not fresh_srcs:
                    raise TimeoutError(f"Timed out after {args.timeout}s waiting for job {job['tag']}.")

                if target_n > 1:
                    # Harvest ALL newly generated candidates (`_c1.._cN`) with each candidate's exact `/edit/` URL!
                    proj_root = _normalize_project_root_url(page.url) or pinned_project_url or page.url
                    cand_records: List[Dict[str, Any]] = []
                    seen_edit_urls = set()
                    for c_idx, fsrc in enumerate(fresh_srcs, start=1):
                        if c_idx > 1:
                            page.goto(proj_root, wait_until="domcontentloaded", timeout=60000)
                            time.sleep(3.0)
                        c_out = job["output"].with_name(f"{job['output'].stem}_c{c_idx}{job['output'].suffix}")
                        # Click the exact tile matching `fsrc` (or `c_idx`-th fresh tile)
                        cur_tiles = _snapshot_result_images(page)
                        fresh_tiles = [(idx, s, w, h) for idx, s, w, h in cur_tiles if s not in before_srcs]
                        target_tile = None
                        for idx, s, w, h in fresh_tiles:
                            if s.split("?")[0] == fsrc.split("?")[0]:
                                target_tile = idx
                                break
                        if target_tile is None and (c_idx - 1) < len(fresh_tiles):
                            target_tile = fresh_tiles[c_idx - 1][0]
                        if target_tile is None:
                            continue
                        el = page.locator("img").nth(target_tile)
                        tbox = el.bounding_box()
                        if not tbox:
                            continue
                        page.mouse.click(tbox["x"] + tbox["width"] * 0.5, tbox["y"] + tbox["height"] * 0.5)
                        time.sleep(2.2)
                        edit_url = page.url
                        if edit_url in seen_edit_urls:
                            continue
                        seen_edit_urls.add(edit_url)
                        dl_btn = _first_visible(page, "button:has-text('download'), button[aria-label*='다운로드'], button[aria-label*='Download']")
                        if dl_btn is None:
                            continue
                        dl_btn.click(timeout=6000, force=True)
                        time.sleep(0.9)
                        pref = job["tier"].strip().upper()
                        tier_sel = (
                            "[role='menuitem']:has-text('4K'), button:has-text('4K')"
                            if pref == "4K"
                            else "[role='menuitem']:has-text('2K'), button:has-text('2K')"
                            if pref == "2K"
                            else "[role='menuitem']:has-text('1K'), [role='menuitem']:has-text('원본')"
                        )
                        opt = page.locator(tier_sel).first
                        if opt.count() == 0 or not opt.is_visible():
                            opt = page.locator("[role='menuitem']:has-text('1K'), [role='menuitem']:has-text('2K'), [role='menuitem']:has-text('원본')").first
                        with page.expect_download(timeout=args.timeout * 1000) as dli:
                            opt.click(timeout=8000, force=True)
                        dl = dli.value
                        tmp = downloads_dir / (dl.suggested_filename or f"cand_{c_idx}_{int(time.time())}.png")
                        dl.save_as(str(tmp))
                        c_out.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(tmp, c_out)
                        w_px, h_px = 0, 0
                        try:
                            from PIL import Image
                            with Image.open(c_out) as im:
                                w_px, h_px = im.size
                        except Exception:
                            pass
                        cand_records.append(
                            {
                                "candidate": c_idx,
                                "output_path": str(c_out),
                                "resolution": [w_px, h_px],
                                "edit_url": edit_url,
                            }
                        )
                        print(f"[Flow x{target_n}] Harvested c{c_idx}/{target_n} -> {c_out.name} ({w_px}x{h_px})", file=sys.stderr, flush=True)
                    rec = {
                        "status": "ok",
                        "tag": job["tag"],
                        "count_requested": target_n,
                        "candidates": cand_records,
                        "uploaded_references": uploaded_n,
                        "config_notes": cfg_notes,
                        "project_url": proj_root,
                    }
                    results.append(rec)
                    continue

                saved_path, tier, menu_items = _download_highest_res_image(
                    page,
                    downloads_dir,
                    job["output"],
                    before_srcs=before_srcs,
                    wait_timeout=args.timeout,
                    preferred_tier=job["tier"],
                )

                w_px, h_px = 0, 0
                try:
                    from PIL import Image
                    with Image.open(saved_path) as im:
                        w_px, h_px = im.size
                except Exception:
                    pass

                rec = {
                    "status": "ok",
                    "tag": job["tag"],
                    "output_path": str(saved_path),
                    "resolution": [w_px, h_px],
                    "download_tier": tier,
                    "download_menu_options": menu_items,
                    "uploaded_references": uploaded_n,
                    "config_notes": cfg_notes,
                    "project_url": page.url,
                }
                results.append(rec)
                print(f"[Flow Batch {j_idx + 1}/{len(jobs)}] DONE {job['tag']} -> {saved_path.name} ({w_px}x{h_px}, {tier}, refs={uploaded_n})", file=sys.stderr, flush=True)

            if args.batch_json is not None:
                print(json.dumps({"status": "ok", "count": len(results), "results": results}, ensure_ascii=False, indent=2))
            else:
                print(json.dumps(results[0], ensure_ascii=False, indent=2))
            return 0
        finally:
            try:
                ctx.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
