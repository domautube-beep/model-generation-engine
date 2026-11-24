#!/bin/zsh
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR"

echo "============================================================"
echo "  릴프로필 (프로필) 스튜디오 — 정본 v1.0.0+"
echo "  (5렌즈x5현장맞춤조명 · 8방향 마스터뱅크 · 디에어브러싱 4K 포어복원)"
echo "  접속 주소: http://127.0.0.1:18830"
echo "============================================================"

# 1) 버전 무결성 자동 체크
if command -v python3 >/dev/null 2>&1; then
  python3 "$SCRIPT_DIR/scripts/version_ctl.py" check || true
fi

# 2) 로컬 .venv 우선 -> uv 자동 탐색 -> 자동 .venv 부트스트랩 (어느 맥/컴퓨터에서나 무의존 실행)
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/launch_studio.py" "$@"
fi

for UV_CAND in "$(command -v uv 2>/dev/null || true)" "$HOME/.local/bin/uv" "/opt/homebrew/bin/uv" "/usr/local/bin/uv"; do
  if [ -n "$UV_CAND" ] && [ -x "$UV_CAND" ]; then
    exec "$UV_CAND" run --no-project \
      --with fastapi \
      --with uvicorn \
      --with python-multipart \
      --with httpx \
      --with opencv-python-headless \
      --with onnxruntime \
      --with pillow \
      python3 "$SCRIPT_DIR/launch_studio.py" "$@"
  fi
done

echo "[릴프로필 초기 환경 자동 구성] 최초 1회 로컬 가상환경(.venv)을 생성합니다..."
python3 -m venv "$SCRIPT_DIR/.venv"
"$SCRIPT_DIR/.venv/bin/pip" install --upgrade pip
"$SCRIPT_DIR/.venv/bin/pip" install -e "$SCRIPT_DIR" fastapi uvicorn python-multipart httpx
exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/launch_studio.py" "$@"
