"""Multi-Backend `1원본들 → 1생성` Image Generation Runtime (`gti` 4K Native CLI, `gemini` REST API, and `manifest` mode)."""

from __future__ import annotations

import asyncio
import base64
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from ..origin_guard import OriginOnlyOneShotGuard


class StudioImageRuntime:
    """Executes single-pass (`1원본들 → 1생성`) multi-reference image generation with cryptographic lineage enforcement."""

    def __init__(
        self,
        gti_bin: Optional[str] = None,
        origin_guard: Optional[OriginOnlyOneShotGuard] = None,
    ) -> None:
        self.gti_bin = gti_bin or shutil.which("gti") or (
            "/opt/homebrew/bin/gti" if Path("/opt/homebrew/bin/gti").exists() else None
        )
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.origin_guard = origin_guard or OriginOnlyOneShotGuard()

    def capabilities(self) -> Dict[str, object]:
        gti_ready = bool(self.gti_bin and Path(self.gti_bin).exists())
        gemini_ready = bool(self.gemini_api_key)
        default_provider = "gti" if gti_ready else ("gemini" if gemini_ready else "manifest")
        return {
            "gti_installed": gti_ready,
            "gti_path": self.gti_bin,
            "gemini_configured": gemini_ready,
            "default_provider": default_provider,
            "supported_providers": ["gti", "gemini", "manifest"],
            "one_origin_one_gen_guard": True,
        }

    async def generate_image(
        self,
        prompt: str,
        reference_image_paths: List[Path],
        output_path: Path,
        size: str = "3840x2160",
        provider: str = "auto",
        job_id: Optional[str] = None,
    ) -> Path:
        # HARD GATE 1: Verify `1원본들 → 1생성` BEFORE calling any image backend!
        self.origin_guard.verify_one_shot_and_inputs(reference_image_paths, job_id=job_id)

        chosen = provider
        if chosen == "auto":
            caps = self.capabilities()
            chosen = str(caps["default_provider"])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if chosen == "gti":
            out = await self._run_gti(prompt, reference_image_paths, output_path, size=size)
        elif chosen == "gemini":
            out = await self._run_gemini(prompt, reference_image_paths, output_path)
        else:
            raise RuntimeError(
                "No live image backend selected ('manifest' mode prepares validated prompts and original reference lists)."
            )

        # HARD GATE 2: Permanently blacklist generated scene output SHA-256 so it can NEVER be re-fed!
        self.origin_guard.register_generated_scene_output(out, job_id=job_id or "")
        return out

    async def _run_gti(
        self,
        prompt: str,
        reference_image_paths: List[Path],
        output_path: Path,
        size: str = "3840x2160",
    ) -> Path:
        if not self.gti_bin or not Path(self.gti_bin).exists():
            raise RuntimeError("GTI CLI ('gti') is not installed on this machine.")

        cmd: List[str] = [
            str(self.gti_bin),
            "--prompt",
            prompt,
            "--output",
            str(output_path),
            "--size",
            size,
        ]
        for ref_p in reference_image_paths:
            if Path(ref_p).exists():
                cmd.extend(["--image", str(ref_p)])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0 or not output_path.exists():
            err_msg = stderr.decode("utf-8", errors="replace").strip() or stdout.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"GTI generation failed (exit {proc.returncode}): {err_msg[:400]}")
        return output_path

    async def _run_gemini(
        self,
        prompt: str,
        reference_image_paths: List[Path],
        output_path: Path,
    ) -> Path:
        if not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

        parts: List[Dict[str, object]] = []
        for idx, ref_p in enumerate(reference_image_paths, 1):
            p = Path(ref_p)
            if p.exists():
                mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
                b64 = base64.b64encode(p.read_bytes()).decode("ascii")
                parts.append({"text": f"Image {idx}:"})
                parts.append({"inlineData": {"mimeType": mime, "data": b64}})
        parts.append({"text": prompt})

        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image-preview:generateContent"
        async with httpx.AsyncClient(timeout=240.0) as client:
            resp = await client.post(
                url,
                params={"key": self.gemini_api_key},
                json={
                    "contents": [{"parts": parts}],
                    "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
                },
            )
            resp.raise_for_status()
            payload = resp.json()

        for cand in payload.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                inline = part.get("inlineData")
                if inline and inline.get("data"):
                    output_path.write_bytes(base64.b64decode(inline["data"]))
                    return output_path
        raise RuntimeError("Gemini response did not contain an image payload.")
