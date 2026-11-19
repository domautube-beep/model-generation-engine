#!/usr/bin/env python3
"""모델생성 스튜디오 v3.0 로컬 웹 서버 런처 (`http://127.0.0.1:18830`)."""

from __future__ import annotations

import argparse
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

from model_generation_engine.studio_app.server import STUDIO_PORT, create_studio_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch 모델생성 스튜디오 v3.0 Web App")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=STUDIO_PORT)
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser tab")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    app = create_studio_app(workspace_dir=args.workspace)

    if not args.no_browser:
        def _open() -> None:
            time.sleep(1.1)
            webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()

    print(f"\n[모델생성 스튜디오 v3.0] 서버 실행 중 -> {url}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
