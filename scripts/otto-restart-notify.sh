#!/bin/bash
# scripts/otto-restart-notify.sh
# Dipanggil di awal startup Otto (ExecStartPost)
# Hanya notif kalau ini BUKAN pertama kali jalan (ada crash log sebelumnya)

LOG_FILE="/data/asd/otto-ai/data/crash.log"
TOPIC="xyz_notif"
PIPER="/usr/local/bin/piper"
PIPER_MODEL="/data/asd/otto-ai/voices/id_ID-news_tts-medium.onnx"
TTS_OUT="/tmp/otto_restart.wav"
SINK_ID=58

# Kalau tidak ada crash log → ini startup pertama, tidak perlu notif
if [ ! -f "$LOG_FILE" ]; then
    exit 0
fi

# Cek apakah crash log diupdate dalam 60 detik terakhir
LAST_MODIFIED=$(stat -c %Y "$LOG_FILE" 2>/dev/null || echo 0)
NOW=$(date +%s)
DIFF=$(( NOW - LAST_MODIFIED ))

if [ "$DIFF" -gt 60 ]; then
    exit 0   # crash log lama, bukan crash barusan
fi

# ── Ada crash barusan → kirim notif "sudah pulih" ────────────────────────────
curl -s -d "Otto sudah restart dan kembali aktif." \
     -H "Title: Otto Kembali Online" \
     -H "Priority: default" \
     -H "Tags: white_check_mark,robot" \
     "https://ntfy.sh/$TOPIC" > /dev/null 2>&1

echo "Otto sudah kembali aktif." \
    | "$PIPER" \
        --model "$PIPER_MODEL" \
        --output_file "$TTS_OUT" 2>/dev/null

if [ -f "$TTS_OUT" ]; then
    pw-play --target "$SINK_ID" --volume 0.5 "$TTS_OUT" 2>/dev/null
    rm -f "$TTS_OUT"
fi
