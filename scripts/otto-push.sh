#!/bin/bash
# otto-push.sh — Auto commit & push otto-ai ke GitHub
# Simpan ke: /data/asd/otto-ai/otto-push.sh
# chmod +x otto-push.sh

OTTO_DIR="/data/asd/otto-ai"
LOG="$OTTO_DIR/otto.log"

cd "$OTTO_DIR" || exit 1

# ── Cari SSH agent yang aktif ──────────────────────────────────────────────
SSH_AGENT_SOCK=$(ls /home/xyz/.ssh/agent/s.*.agent.* 2>/dev/null | head -1)
if [ -n "$SSH_AGENT_SOCK" ]; then
    export SSH_AUTH_SOCK="$SSH_AGENT_SOCK"
fi

# ── Cek syntax semua file Python sebelum commit ───────────────────────────
SYNTAX_ERRORS=""
while IFS= read -r -d '' pyfile; do
    ERR=$(python3 -m py_compile "$pyfile" 2>&1)
    if [ $? -ne 0 ]; then
        SYNTAX_ERRORS+="$pyfile: $ERR\n"
    fi
done < <(find "$OTTO_DIR" -name "*.py" -not -path "*/venv/*" -print0)

if [ -n "$SYNTAX_ERRORS" ]; then
    echo "$(date '+%H:%M') [Syntax] ERROR ditemukan:" >> "$LOG"
    echo -e "$SYNTAX_ERRORS" >> "$LOG"
    # Tampilkan error langsung di terminal
    echo ""
    echo "❌ SyntaxError — push dibatalkan:"
    echo -e "$SYNTAX_ERRORS"
    notify-send "Otto" "❌ SyntaxError! Push dibatalkan — cek terminal" --urgency=critical 2>/dev/null
    exit 1
fi

echo "$(date '+%H:%M') [Syntax] Semua file Python OK." >> "$LOG"

# ── Cek apakah ada perubahan ───────────────────────────────────────────────
if [ -z "$(git status --porcelain)" ]; then
    echo "$(date '+%H:%M') [Git] Tidak ada perubahan, skip." >> "$LOG"
    notify-send "Otto" "[Git] Tidak ada perubahan" --urgency=low 2>/dev/null
    exit 0
fi

# ── Generate folder tree snapshot sebelum commit ──────────────────────────
if [ -f "$OTTO_DIR/generate_tree.sh" ]; then
    bash "$OTTO_DIR/generate_tree.sh" >> "$LOG" 2>&1
    echo "$(date '+%H:%M') [Tree] Snapshot diperbarui." >> "$LOG"
else
    echo "$(date '+%H:%M') [Tree] generate_tree.sh tidak ditemukan, skip." >> "$LOG"
fi

# ── Regenerate otto_for_claude.txt ────────────────────────────────────────
if [ -f "$OTTO_DIR/generate_context.sh" ]; then
    bash "$OTTO_DIR/generate_context.sh" >> "$LOG" 2>&1
    echo "$(date '+%H:%M') [Context] otto_for_claude.txt diperbarui." >> "$LOG"
fi

# ── Commit & push ─────────────────────────────────────────────────────────
git add -A
COMMIT_MSG="auto: $(date '+%Y-%m-%d %H:%M')"
git commit -m "$COMMIT_MSG"

PUSH_OUTPUT=$(git push origin main 2>&1)
PUSH_STATUS=$?
echo "$PUSH_OUTPUT" >> "$LOG"

if [ $PUSH_STATUS -eq 0 ]; then
    echo "$(date '+%H:%M') [Git] Push berhasil: $COMMIT_MSG" >> "$LOG"
    echo "✅ Push berhasil: $COMMIT_MSG"
    notify-send "Otto" "📤 Otto-AI di-push ke GitHub" --urgency=low 2>/dev/null
else
    echo "$(date '+%H:%M') [Git] Push GAGAL:" >> "$LOG"
    echo "$PUSH_OUTPUT" >> "$LOG"
    echo ""
    echo "❌ Push GAGAL:"
    echo "$PUSH_OUTPUT"
    notify-send "Otto" "❌ Push gagal! $PUSH_OUTPUT" --urgency=critical 2>/dev/null
fi
