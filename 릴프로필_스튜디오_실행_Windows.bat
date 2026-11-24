@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONPATH=%~dp0

echo ============================================================
echo   릴프로필 (프로필) 스튜디오 — 정본 v1.0.0+ (Windows)
echo   접속 주소: http://127.0.0.1:18830
echo ============================================================

if not exist ".venv\Scripts\python.exe" (
    echo [초기 1회 설정] 로컬 Python 가상환경(.venv)을 생성하고 필수 패키지를 설치합니다...
    python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -e . fastapi uvicorn python-multipart httpx
)

".venv\Scripts\python.exe" scripts\version_ctl.py check
".venv\Scripts\python.exe" launch_studio.py
pause
