#!/bin/zsh
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

echo "============================================================"
echo "  릴프로필 (프로필) — 정본 v1.0.5+ 원클릭 설치 스크립트"
echo "  설치 경로: $REPO_DIR"
echo "============================================================"

# 1. 로컬 ~/Projects 별칭 심볼릭 링크 등록 ('릴프로필', '프로필')
mkdir -p "$HOME/Projects"
for ALIAS_NAME in "릴프로필" "프로필"; do
  TARGET_LINK="$HOME/Projects/$ALIAS_NAME"
  if [ ! -e "$TARGET_LINK" ] && [ ! -L "$TARGET_LINK" ]; then
    ln -s "$REPO_DIR" "$TARGET_LINK"
    echo "[별칭 등록] $TARGET_LINK -> $REPO_DIR"
  fi
done

# 2. Python 가상환경(.venv) 준비 및 의존성 설치
if [ ! -x "$REPO_DIR/.venv/bin/python" ]; then
  echo "[가상환경 생성] .venv 구성 및 필수 라이브러리 설치 중..."
  python3 -m venv "$REPO_DIR/.venv"
  "$REPO_DIR/.venv/bin/pip" install --upgrade pip
  "$REPO_DIR/.venv/bin/pip" install -e "$REPO_DIR" fastapi uvicorn python-multipart httpx
else
  echo "[가상환경 확인] 기존 .venv 사용 가능"
fi

# 3. Git pre-commit 자동 버전체크 훅 연결 (.git 존재 시)
if [ -d "$REPO_DIR/.git" ]; then
  mkdir -p "$REPO_DIR/.git/hooks"
  cat > "$REPO_DIR/.git/hooks/pre-commit" <<'EOF'
#!/bin/sh
REPO_ROOT="$(git rev-parse --show-toplevel)"
exec python3 "$REPO_ROOT/hooks/pre-commit-version-check.py"
EOF
  chmod +x "$REPO_DIR/.git/hooks/pre-commit" "$REPO_DIR/hooks/pre-commit-version-check.py"
  echo "[Git 훅 등록] pre-commit 자동 버전체크 활성화 완료"
fi

# 4. Antigravity, Claude Code & iCloud Control Tower 경량 스킬 등록 (< 50KB)
SKILL_ROOTS=("$HOME/.gemini/config/skills" "$HOME/.claude/skills")
ICLOUD_SKILLS="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Antigravity_Control_Tower/skills"
if [ -d "$(dirname "$ICLOUD_SKILLS")" ]; then
  SKILL_ROOTS+=("$ICLOUD_SKILLS")
fi

for SKILL_ROOT in "${SKILL_ROOTS[@]}"; do
  for SNAME in "reel-profile" "model-generation-engine"; do
    if [ -f "$REPO_DIR/skills/$SNAME/SKILL.md" ]; then
      mkdir -p "$SKILL_ROOT/$SNAME"
      sed "s|__REELPROFILE_REPO_DIR__|$REPO_DIR|g" "$REPO_DIR/skills/$SNAME/SKILL.md" > "$SKILL_ROOT/$SNAME/SKILL.md"
      echo "[AI 스킬 등록] $SKILL_ROOT/$SNAME/SKILL.md"
    fi
  done
done

# 5. Antigravity 전역 `원본1생성게이트.py` 및 `hooks.json` 동기화
if [ -f "$REPO_DIR/hooks/enforce_1origin_1gen_hook.py" ]; then
  mkdir -p "$HOME/.gemini/config"
  cp "$REPO_DIR/hooks/enforce_1origin_1gen_hook.py" "$HOME/.gemini/config/원본1생성게이트.py"
  chmod +x "$HOME/.gemini/config/원본1생성게이트.py"
  echo "[전역 보안 훅 동기화] $HOME/.gemini/config/원본1생성게이트.py"
fi

# 6. 버전 무결성 최종 체크
python3 "$REPO_DIR/scripts/version_ctl.py" check

echo ""
echo "설치가 완료되었습니다!"
echo "- 스튜디오 실행: '$REPO_DIR/릴프로필 스튜디오 실행.command' 더블클릭 (http://127.0.0.1:18830)"
echo "- AI 에이전트에서 '릴프로필' 또는 '프로필'이라고 부르면 즉시 로드됩니다."
