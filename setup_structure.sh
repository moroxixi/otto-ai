#!/bin/bash
# setup_structure.sh — Buat struktur folder Otto & pindahkan file yang ada
# Jalankan SEKALI dari root: bash setup_structure.sh
# Setelah ini tidak perlu dijalankan lagi

BASE="/data/asd/otto-ai"
cd "$BASE" || exit 1

echo "━━━ Otto AI — Setup Struktur Folder ━━━"
echo ""

# ─── 1. Buat semua folder ─────────────────────────────────────────────────────
echo "▶ Membuat folder..."

mkdir -p core
mkdir -p server/static
mkdir -p skills
mkdir -p intelligence
mkdir -p self
mkdir -p data
mkdir -p ssl
mkdir -p voices    # sudah ada, tidak masalah

echo "  ✓ core/"
echo "  ✓ server/  (+ server/static/)"
echo "  ✓ skills/"
echo "  ✓ intelligence/"
echo "  ✓ self/"
echo "  ✓ data/"
echo "  ✓ ssl/"
echo "  ✓ voices/"

# ─── 2. Pindahkan file yang sudah dibuat ──────────────────────────────────────
echo ""
echo "▶ Memindahkan file yang sudah ada..."

move_if_exists() {
    local src="$BASE/$1"
    local dst="$BASE/$2"
    if [ -f "$src" ]; then
        mv "$src" "$dst"
        echo "  ✓ $1 → $2"
    else
        echo "  ○ $1 (belum ada, skip)"
    fi
}

# File yang sudah kita buat di chat ini
move_if_exists "config.py"       "core/config.py"
move_if_exists "generate_tree.sh" "generate_tree.sh"   # tetap di root

# ─── 3. Buat __init__.py di setiap package Python ────────────────────────────
echo ""
echo "▶ Membuat __init__.py..."

for pkg in core server skills intelligence self; do
    touch "$BASE/$pkg/__init__.py"
    echo "  ✓ $pkg/__init__.py"
done

# ─── 4. Buat README singkat di tiap folder ───────────────────────────────────
echo ""
echo "▶ Membuat README per folder..."

cat > "$BASE/core/README.md" << 'EOF'
# core/
Fondasi Otto — tidak bisa jalan tanpa semua file di sini.

| File | Fungsi |
|---|---|
| config.py | Semua konstanta, path, API key rotation |
| memory.py | Ingatan jangka pendek & panjang |
| transcriber.py | Whisper STT — suara → teks |
| speaker.py | Piper TTS — teks → suara |
| brain.py | LLM engine (Groq) |
| executor.py | Dispatcher aksi dari intent |
EOF

cat > "$BASE/server/README.md" << 'EOF'
# server/
Interface antara iPhone dan Otto.

| File | Fungsi |
|---|---|
| app.py | FastAPI app, HTTPS, routing |
| websocket.py | WebSocket handler real-time |
| static/index.html | UI di iPhone Chrome |
EOF

cat > "$BASE/skills/README.md" << 'EOF'
# skills/
Kemampuan spesifik Otto — tiap file = satu domain aksi.

| File | Fungsi |
|---|---|
| system.py | volume, lock, shutdown, brightness |
| media.py | putar/pause/skip lagu |
| reminder.py | ingatkan_aku, jadwal |
EOF

cat > "$BASE/intelligence/README.md" << 'EOF'
# intelligence/
Yang membuat Otto "hidup" — lapisan proaktif.

| File | Fungsi |
|---|---|
| activity_watcher.py | Amati pola aktivitas Rofi |
| profiler.py | Bangun model Rofi dari observasi |
| curiosity.py | Observe → Ask → Hypothesize → Verify |
| scheduler.py | Jadwal tugas background |
EOF

cat > "$BASE/self/README.md" << 'EOF'
# self/
Kesadaran Otto terhadap dirinya sendiri.

| File | Fungsi |
|---|---|
| model.py | Self-model Otto |
| github_checker.py | Cek update repo Otto sendiri |
EOF

cat > "$BASE/data/README.md" << 'EOF'
# data/
Semua data runtime Otto — jangan edit manual.

| File | Isi |
|---|---|
| memory.json | Short & long term memory |
| profile.json | Model Rofi yang dibangun Otto |
| hypotheses.json | Hipotesis aktif yang belum diverifikasi |
| activity.log | Log aktivitas mentah |
EOF

echo "  ✓ core/README.md"
echo "  ✓ server/README.md"
echo "  ✓ skills/README.md"
echo "  ✓ intelligence/README.md"
echo "  ✓ self/README.md"
echo "  ✓ data/README.md"

# ─── 5. Buat .gitignore ──────────────────────────────────────────────────────
echo ""
echo "▶ Membuat .gitignore..."

cat > "$BASE/.gitignore" << 'EOF'
# Environment
.env

# Data runtime (jangan di-commit)
data/
ssl/

# Audio & model besar
voices/
*.wav
*.mp3
*.onnx

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.venv/

# Output Claude
otto_for_claude.txt
EOF

echo "  ✓ .gitignore"

# ─── 6. Tampilkan hasil ───────────────────────────────────────────────────────
echo ""
echo "━━━ Struktur Akhir ━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
find "$BASE" \
    -not -path "*/__pycache__/*" \
    -not -path "*/voices/*" \
    -not -path "*/data/*" \
    -not -name "*.pyc" \
    -not -name "*.onnx" \
    | sort \
    | sed "s|$BASE/||" \
    | grep -v "^$" \
    | awk -F/ '{
        depth = NF - 1
        indent = ""
        for (i=0; i<depth; i++) indent = indent "    "
        if (NF == 1) print "otto-ai/"
        else if ($NF != "") print indent "├── " $NF
    }'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ Setup selesai."
echo ""
echo "Langkah berikutnya:"
echo "  1. Jalankan: bash generate_tree.sh"
echo "     → Buat otto_for_claude.txt untuk konteks chat baru"
echo "  2. Lanjut buat: core/memory.py"
echo ""
