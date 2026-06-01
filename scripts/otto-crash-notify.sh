#!/bin/bash
# scripts/otto-crash-notify.sh
# Dipanggil systemd saat Otto mati (ExecStopPost)
# Kirim notif ke HP + ucapkan via TTS

TOPIC="xyz_notif"
PIPER="/usr/local/bin/piper"
PIPER_MODEL="/data/asd/otto-ai/voices/id_ID-news_tts-medium.onnx"
TTS_OUT="/tmp/otto_crash.wav"
LOG_FILE="/data/asd/otto-ai/data/crash.log"
SINK_ID=58

# ── Ambil info crash ──────────────────────────────────────────────────────────
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
EXIT_CODE="${EXIT_CODE:-?}"         # di-set otomatis oleh systemd
REASON="${EXIT_CODE}"

# Tulis ke crash log
mkdir -p "$(dirname "$LOG_FILE")"
echo "[$TIMESTAMP] Otto crash — exit code: $EXIT_CODE" >> "$LOG_FILE"

# ── Notif ke HP via ntfy ──────────────────────────────────────────────────────
curl -s -d "Otto crash jam $TIMESTAMP — exit code: $EXIT_CODE. Sedang restart otomatis..." \
     -H "Title: Otto Crash!" \
     -H "Priority: urgent" \
     -H "Tags: warning,robot" \
     "https://ntfy.sh/$TOPIC" > /dev/null 2>&1

# ── TTS via Piper ─────────────────────────────────────────────────────────────
echo "Otto mengalami error dan sedang restart otomatis." \
    | "$PIPER" \
        --model "$PIPER_MODEL" \
        --output_file "$TTS_OUT" 2>/dev/null

if [ -f "$TTS_OUT" ]; then
    pw-play --target "$SINK_ID" --volume 0.5 "$TTS_OUT" 2>/dev/null
    rm -f "$TTS_OUT"
fi
