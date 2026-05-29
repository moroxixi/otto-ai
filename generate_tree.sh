#!/bin/bash
# generate_tree.sh — Otto AI Project Snapshot Generator
# Jalankan dari root: bash generate_tree.sh
# Output: otto_for_claude.txt (lampirkan ke chat baru)

OUTPUT="otto_for_claude.txt"
BASE="/data/asd/otto-ai"
cd "$BASE" || exit 1

# ─── Header ───────────────────────────────────────────────────────────────────
cat > "$OUTPUT" << 'HEADER'
╔══════════════════════════════════════════════════════════════════╗
║                    OTTO-AI — CONTEXT FILE                        ║
║           Lampirkan file ini di awal setiap chat baru            ║
╚══════════════════════════════════════════════════════════════════╝

━━━ IDENTITAS PROYEK ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Nama         : Otto
Pemilik      : Rofi (pengguna satu-satunya)
Tujuan       : AI asisten proaktif yang belajar mengenal Rofi
               seperti manusia mengenal manusia — bukan dari
               file hardcoded, tapi dari observasi nyata.

━━━ FILOSOFI INTI ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Otto BUKAN asisten reaktif biasa.
Otto mengikuti siklus ini secara alami:

  Mengamati → Bertanya untuk mengenal → Menyimpulkan
      ↑                                      ↓
  Merevisi  ←←←←←←← Bertanya untuk verifikasi

Contoh nyata:
  Libo (versi lama): "Rofi suka kopi oat" → hardcoded di file
  Otto (versi baru): Otto amati Rofi pesan kopi tiap pagi
                     → hipotesis: "Rofi suka kopi"
                     → tanya: "Rofi, kamu tiap pagi minum kopi?"
                     → Rofi jawab → Otto simpan atau revisi

Aturan Otto:
  ✗ Tidak boleh hardcode fakta tentang Rofi
  ✓ Semua pengetahuan tentang Rofi harus dari observasi + konfirmasi
  ✓ Jika hipotesis salah → revisi, jangan defensif
  ✓ Otto boleh diam dan amati lebih lama sebelum menyimpulkan

━━━ STACK TEKNIS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OS       : openSUSE Tumbleweed, Hyprland, Wayland
Audio    : PipeWire — pw-record / pw-play, sink ID 58
STT      : faster-whisper
           - tiny   → perintah pendek (latency rendah)
           - medium → transkripsi panjang (lebih akurat)
LLM      : Groq API
           - llama-3.1-8b-instant   → perintah & aksi cepat
           - llama-3.3-70b-versatile → ngobrol & analisis profil
           - 6 API key dengan round-robin rotation otomatis
TTS      : Piper binary (/usr/local/bin/piper)
           Model: id_ID-news_tts-medium.onnx
Server   : FastAPI + uvicorn, port 8000, HTTPS self-signed SSL
Interface: WebSocket dari iPhone (Chrome)
Path     : /data/asd/otto-ai/

━━━ ARSITEKTUR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

iPhone → WebSocket → server/app.py
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        transcriber   brain.py   executor
        (Whisper)    (Groq LLM)  (dispatch)
              │           │           │
              └───────────┘     skills/*.py
                    │
              memory.py + profiler.py
                    │
              speaker.py (Piper TTS)
                    │
                 iPhone ←

Paralel (diam-diam):
  activity_watcher → profiler → curiosity → tanya Rofi

━━━ STATUS FILE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HEADER

# ─── Status tiap file ─────────────────────────────────────────────────────────
echo "" >> "$OUTPUT"

check_file() {
    local filepath="$1"
    local label="$2"
    if [ -f "$BASE/$filepath" ]; then
        local size=$(wc -l < "$BASE/$filepath" 2>/dev/null)
        local mtime=$(stat -c "%Y" "$BASE/$filepath" 2>/dev/null)
        local age=$(( ($(date +%s) - mtime) / 3600 ))
        printf "  %-40s ✓ %4d baris  (${age}j lalu)\n" "$label" "$size" >> "$OUTPUT"
    else
        printf "  %-40s ✗ belum dibuat\n" "$label" >> "$OUTPUT"
    fi
}

echo "[ CORE ]" >> "$OUTPUT"
check_file "core/config.py"        "core/config.py"
check_file "core/memory.py"        "core/memory.py"
check_file "core/transcriber.py"   "core/transcriber.py"
check_file "core/speaker.py"       "core/speaker.py"
check_file "core/brain.py"         "core/brain.py"
check_file "core/executor.py"      "core/executor.py"

echo "" >> "$OUTPUT"
echo "[ SERVER ]" >> "$OUTPUT"
check_file "server/app.py"              "server/app.py"
check_file "server/websocket.py"        "server/websocket.py"
check_file "server/static/index.html"   "server/static/index.html"

echo "" >> "$OUTPUT"
echo "[ SKILLS ]" >> "$OUTPUT"
check_file "skills/system.py"     "skills/system.py"
check_file "skills/media.py"      "skills/media.py"
check_file "skills/reminder.py"   "skills/reminder.py"

echo "" >> "$OUTPUT"
echo "[ INTELLIGENCE ]" >> "$OUTPUT"
check_file "intelligence/activity_watcher.py"  "intelligence/activity_watcher.py"
check_file "intelligence/profiler.py"           "intelligence/profiler.py"
check_file "intelligence/curiosity.py"          "intelligence/curiosity.py"
check_file "intelligence/scheduler.py"          "intelligence/scheduler.py"

echo "" >> "$OUTPUT"
echo "[ SELF ]" >> "$OUTPUT"
check_file "self/model.py"           "self/model.py"
check_file "self/github_checker.py"  "self/github_checker.py"

echo "" >> "$OUTPUT"
echo "[ DATA ]" >> "$OUTPUT"
check_file "data/memory.json"      "data/memory.json"
check_file "data/profile.json"     "data/profile.json"
check_file "data/hypotheses.json"  "data/hypotheses.json"
check_file "data/activity.log"     "data/activity.log"

# ─── Tree folder ──────────────────────────────────────────────────────────────
cat >> "$OUTPUT" << 'TREE_HEADER'

━━━ STRUKTUR FOLDER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TREE_HEADER

# tree kalau ada, fallback ke find
if command -v tree &>/dev/null; then
    tree "$BASE" \
        --noreport \
        -I "*.pyc|__pycache__|*.onnx|*.wav|*.mp3|cache|recordings|*.json.migrated" \
        -a \
        --dirsfirst \
        -h \
        2>/dev/null >> "$OUTPUT"
else
    find "$BASE" \
        -not -path "*/cache/*" \
        -not -path "*/__pycache__/*" \
        -not -name "*.pyc" \
        -not -name "*.onnx" \
        -not -name "*.wav" \
        | sort \
        | sed "s|$BASE||" \
        | awk '{
            n = split($0, a, "/")
            indent = ""
            for (i=2; i<n; i++) indent = indent "│   "
            if (n > 1) print indent "├── " a[n]
        }' >> "$OUTPUT"
fi

# ─── Footer ───────────────────────────────────────────────────────────────────
cat >> "$OUTPUT" << 'FOOTER'

━━━ CARA PAKAI FILE INI ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Jalankan: bash /data/asd/otto-ai/generate_tree.sh
2. Lampirkan otto_for_claude.txt ke chat baru dengan Claude
3. Claude langsung tau konteks penuh — tidak perlu penjelasan ulang

File ini di-generate otomatis. Jangan edit manual.
FOOTER

# ─── Metadata ─────────────────────────────────────────────────────────────────
GENERATED_AT=$(date "+%Y-%m-%d %H:%M")
TOTAL_PY=$(find "$BASE" -name "*.py" 2>/dev/null | wc -l)
TOTAL_JSON=$(find "$BASE" -name "*.json" 2>/dev/null | wc -l)

sed -i "1a Generated  : $GENERATED_AT  |  .py: $TOTAL_PY  |  .json: $TOTAL_JSON" "$OUTPUT"

echo ""
echo "✓ otto_for_claude.txt berhasil dibuat"
echo "  → $BASE/$OUTPUT"
echo "  → Lampirkan ke chat baru untuk lanjut"
