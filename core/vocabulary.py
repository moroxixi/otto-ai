# core/vocabulary.py
# Kosakata khusus Otto — disuntikkan ke Whisper sebagai initial_prompt
# Tujuan: Whisper lebih akurat kenali kata-kata khas Otto + Rofi
#         tanpa perlu training ulang, hampir 0 overhead komputasi

# ── Nama yang sering salah tangkap Whisper ─────────────────────
# Format: { "salah_tangkap": "yang_benar" }
# Tambah di sini kalau Otto sering salah dengar nama tertentu
NAMA_ALIAS: dict[str, str] = {
    "auto":   "Otto",
    "oto":    "Otto",
    "otot":   "Otto",
    "otak":   "Otto",
    "otto":   "Otto",   # jaga-jaga kapitalisasi
    "oto,":   "Otto",
    "itu,":   "Otto",
}

# ── Istilah khusus yang harus Whisper kenali ───────────────────
ISTILAH_KHUSUS: list[str] = [
    # Nama asisten
    "Otto",


    # Perintah media
    "putar lagu",
    "matikan lagu",
    "lagu jadul",
    "lagu santai",
    "tambah volume",
    "kecilkan volume",
    
    

    # Nama & tempat lokal
    "Rofi",
    "Cirebon",
    "Hyprland",
    "Wayland",
    "PipeWire",
]


def build_initial_prompt() -> str:
    """
    Buat initial_prompt untuk faster-whisper.

    Cara kerja: Whisper lihat kalimat-kalimat ini sebelum transkripsi
    → dia bias ke kata-kata yang sudah pernah "dilihat" ini.
    Tidak ada komputasi ekstra, hanya context window Whisper diisi.

    Tips: gunakan kalimat natural, bukan list kata. Whisper lebih
    responsif ke kalimat daripada kata-kata tersendiri.
    """
    parts = [
        "Otto putar lagu santai, lagu jadul, matikan lagu.",
        "Otto pindah ke ruang kerja satu, dua, tiga." 
        "Otto ingatkan aku, tambah volume, kecilkan volume.",    ]
    return " ".join(parts)


# Cache — dibangun sekali saat import, tidak diulang tiap request
WHISPER_INITIAL_PROMPT: str = build_initial_prompt()
