#!/bin/zsh
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR"

echo "============================================================"
echo "  모델생성 스튜디오 v3.0 (8각도 · 멀티아이템 · 다중장소 · 3D프리비즈)"
echo "  접속 주소: http://127.0.0.1:18830"
echo "============================================================"

exec /Users/doma/.local/bin/uv run --no-project \
  --with fastapi \
  --with uvicorn \
  --with python-multipart \
  --with httpx \
  --with opencv-python-headless \
  --with onnxruntime \
  --with pillow \
  python3 "$SCRIPT_DIR/launch_studio.py"
