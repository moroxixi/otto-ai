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

# ─── Status tiap file — DYNAMIC ───────────────────────────────────────────────

scan_folder() {
    local label="$1"
    local folder="$2"
    echo "" >> "$OUTPUT"
    echo "[ $label ]" >> "$OUTPUT"
    
    # Scan semua .py dan .json di folder, sort alfabetis
    find "$BASE/$folder" -maxdepth 1 \( -name "*.py" -o -name "*.json" \) 2>/dev/null | sort | while read -r filepath; do
        local relpath="${filepath#$BASE/}"
        local size=$(wc -l < "$filepath" 2>/dev/null)
        local mtime=$(stat -c "%Y" "$filepath" 2>/dev/null)
        local age=$(( ($(date +%s) - mtime) / 3600 ))
        printf "  %-40s ✓ %4d baris  (${age}j lalu)\n" "$relpath" "$size" >> "$OUTPUT"
    done
    
    # Jika folder kosong / tidak ada file
    local count=$(find "$BASE/$folder" -maxdepth 1 \( -name "*.py" -o -name "*.json" \) 2>/dev/null | wc -l)
    if [ "$count" -eq 0 ]; then
        echo "  (belum ada file)" >> "$OUTPUT"
    fi
}

scan_folder "CORE"         "core"
scan_folder "SERVER"       "server"
scan_folder "SKILLS"       "skills"
scan_folder "INTELLIGENCE" "intelligence"
scan_folder "SELF"         "otto_self"
scan_folder "DATA"         "data"
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
