# core/config.py
# Tulang punggung Otto — semua konstanta & path ada di sini
# Jangan hardcode path/key di file lain, selalu import dari sini

import os
import itertools
from pathlib import Path
from dotenv import load_dotenv

# ─── Load .env ────────────────────────────────────────────────────────────────
BASE_DIR = Path("/data/asd/otto-ai")
load_dotenv(BASE_DIR / ".env")

# ─── PATH ─────────────────────────────────────────────────────────────────────
PATHS = {
    "base":          BASE_DIR,
    "voices":        BASE_DIR / "voices",
    "memory":        BASE_DIR / "data" / "memory.json",
    "profile":       BASE_DIR / "data" / "profile.json",
    "activity_log":  BASE_DIR / "data" / "activity.log",
    "hypotheses":    BASE_DIR / "data" / "hypotheses.json",
    "short_term_cache": BASE_DIR / "data" / "short_term_cache.json",
    "otto_model":    BASE_DIR / "otto_self" / "otto_model.json",
    "ssl_cert":      BASE_DIR / "ssl" / "cert.pem",
    "ssl_key":       BASE_DIR / "ssl" / "key.pem",
}

# Buat folder data & ssl jika belum ada
for key in ["memory", "profile", "activity_log", "hypotheses"]:
    PATHS[key].parent.mkdir(parents=True, exist_ok=True)

# ─── GROQ API KEY ROTATION ────────────────────────────────────────────────────
# Otto pakai round-robin: kalau key 1 kena rate limit → otomatis key 2, dst.
_groq_keys_raw = [
    os.getenv(f"GROQ_API_KEY_{i}") for i in range(1, 7)
]
GROQ_API_KEYS = [k for k in _groq_keys_raw if k]  # buang yang None/kosong

if not GROQ_API_KEYS:
    raise EnvironmentError(
        "Tidak ada Groq API key ditemukan di .env. "
        "Pastikan format: GROQ_API_KEY_1=gsk_xxx"
    )

_groq_cycle = itertools.cycle(GROQ_API_KEYS)

def get_groq_key() -> str:
    """Ambil key berikutnya dari rotation. Panggil tiap request LLM."""
    return next(_groq_cycle)

# ─── MODEL CONFIG ─────────────────────────────────────────────────────────────
MODELS = {
    # Untuk perintah singkat — cepat, hemat quota
    "command": "llama-3.1-8b-instant",
    # Untuk ngobrol panjang / analisis profil — lebih dalam
    "chat":    "llama-3.3-70b-versatile",
}

# ─── WHISPER STT ──────────────────────────────────────────────────────────────
WHISPER = {
    # "tiny"   → perintah pendek, latency rendah
    # "medium" → transkripsi panjang, lebih akurat
    "model_command": "small",
    "model_chat":    "small",
    "language":      "id",        # Bahasa Indonesia
    "device":        "cpu",       # ganti "cuda" kalau ada GPU
    "compute_type":  "int8",     # hemat RAM di CPU
    "stt_timeout": 300,
}

# ─── PIPER TTS ────────────────────────────────────────────────────────────────
PIPER = {
    "binary":  "/usr/local/bin/piper",
    "model":   BASE_DIR / "voices" / "id_ID-news_tts-medium.onnx",
    "config":  BASE_DIR / "voices" / "id_ID-news_tts-medium.onnx.json",
}

# ─── KOKORO TTS ───────────────────────────────────────────────────────────────
KOKORO = {
    "model":  BASE_DIR / "voices" / "kokoro" / "kokoro-v1.0.onnx",
    "voices": BASE_DIR / "voices" / "kokoro" / "voices-v1.0.bin",
    # Ganti voice di sini untuk eksperimen
    "voice":  "am_michael",
    "speed":  1.0,
    "lang":   "id",
}

# ─── OUTPUT ROUTING ───────────────────────────────────────────────────────────
OUTPUT = {
    # "laptop" → pw-play langsung ke speaker laptop
    # "ws"     → kirim base64 WAV ke iPhone via WebSocket
    # "both"   → laptop + ws (untuk debugging)
    "proactive_output": "laptop",   # Otto ngomong sendiri → speaker laptop
    "response_output":  "ws",       # Balas Rofi → ke iPhone
}

# ─── AUDIO (PipeWire) ─────────────────────────────────────────────────────────
AUDIO = {
    "sink_id":     58,           # ID speaker laptop kamu
    "record_cmd":  "pw-record",
    "play_cmd":    "pw-play",
    "sample_rate": 16000,        # Whisper butuh 16kHz
    "channels":    1,
    "format":      "s16",        # signed 16-bit
}

# ─── SERVER ───────────────────────────────────────────────────────────────────
SERVER = {
    "host":    "0.0.0.0",
    "port":    8000,
    "ssl":     True,
    "reload":  False,            # True hanya saat development
}

# ─── INTELLIGENCE ─────────────────────────────────────────────────────────────
INTELLIGENCE = {
    # Berapa observasi terkumpul sebelum Otto mulai buat hipotesis
    "min_observations_to_hypothesize": 5,
    # Berapa kali pola muncul sebelum dianggap kebiasaan
    "pattern_threshold": 3,
    # Jeda minimum antar pertanyaan proaktif Otto (detik)
    "curiosity_cooldown": 3600,   # 1 jam
    # Jam aktif Rofi (Otto lebih waspada di rentang ini)
    "active_hours": (6, 23),      # 06:00–23:00
    "curiosity_boldness": 0.5,
}

# ─── MEMORY ───────────────────────────────────────────────────────────────────
MEMORY = {
    # Berapa pesan terakhir yang dibawa ke konteks LLM
    "short_term_limit": 20,
    # Berapa item max di long-term memory
    "long_term_limit":  200,
}

# ─── DEBUG ────────────────────────────────────────────────────────────────────
DEBUG = os.getenv("OTTO_DEBUG", "false").lower() == "true"
LOG_LEVEL = "DEBUG" if DEBUG else "INFO"
