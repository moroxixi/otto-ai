#!/bin/bash
# otto-push.sh — Auto commit & push otto-ai ke GitHub
# Simpan ke: /data/asd/otto-ai/otto-push.sh
# chmod +x otto-push.sh
# Jalankan manual: bash otto-push.sh
# Atau jadwalkan via cron / systemd timer

OTTO_DIR="/data/asd/otto-ai"
LOG="$OTTO_DIR/otto.log"

cd "$OTTO_DIR" || exit 1

# ── Cari SSH agent yang aktif ──────────────────────────────────────────────
SSH_AGENT_SOCK=$(ls /home/xyz/.ssh/agent/s.*.agent.* 2>/dev/null | head -1)
if [ -n "$SSH_AGENT_SOCK" ]; then
    export SSH_AUTH_SOCK="$SSH_AGENT_SOCK"
fi

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

# ── Regenerate otto_for_claude.txt (context file) ─────────────────────────
# Opsional: jalankan script context generator jika ada
if [ -f "$OTTO_DIR/generate_context.sh" ]; then
    bash "$OTTO_DIR/generate_context.sh" >> "$LOG" 2>&1
    echo "$(date '+%H:%M') [Context] otto_for_claude.txt diperbarui." >> "$LOG"
fi

# ── Commit & push ─────────────────────────────────────────────────────────
git add -A
git commit -m "auto: $(date '+%Y-%m-%d %H:%M')"
git push origin main >> "$LOG" 2>&1

# ── Cek hasil push ────────────────────────────────────────────────────────
if [ $? -eq 0 ]; then
    echo "$(date '+%H:%M') [Git] Push berhasil." >> "$LOG"
    notify-send "Otto" "📤 Otto-AI di-push ke GitHub" --urgency=low 2>/dev/null
else
    echo "$(date '+%H:%M') [Git] Push GAGAL — cek koneksi atau SSH key." >> "$LOG"
    notify-send "Otto" "❌ Push gagal! Cek otto.log" --urgency=critical 2>/dev/null
fi
