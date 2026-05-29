"""
self/model.py — Otto Sadar Diri

Otto tahu:
  1. Siapa dia (identitas + filosofi)
  2. Seberapa "matang" dia (stats dari data nyata)
  3. Seberapa "berkembang" dia (parameter kepribadian yang tumbuh)
  4. Apa yang baru berubah dari dirinya (dari github_checker)

Analogi: Seperti manusia yang tau nama, umur, kepribadian,
dan ingat bahwa kemarin dia belajar hal baru.
"""

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Path ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/data/asd/otto-ai")
DATA_DIR   = BASE_DIR / "data"
SELF_DIR   = BASE_DIR / "self"
MODEL_FILE = SELF_DIR / "otto_model.json"   # state kepribadian disimpan di sini


# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 1 — IDENTITAS STATIS (tidak berubah)
# ══════════════════════════════════════════════════════════════════════════════

OTTO_IDENTITY = {
    "name": "Otto",
    "owner": "Rofi",
    "philosophy": (
        "Aku bukan asisten reaktif. Aku mengamati, bertanya, menyimpulkan, "
        "dan merevisi — seperti manusia mengenal manusia."
    ),
    "core_loop": ["Mengamati", "Bertanya", "Menyimpulkan", "Bertanya ulang", "Merevisi"],
    "rules": [
        "Tidak hardcode fakta tentang Rofi",
        "Semua pengetahuan Rofi dari observasi + konfirmasi",
        "Jika hipotesis salah → revisi, jangan defensif",
        "Boleh diam dan amati lebih lama sebelum menyimpulkan",
    ],
    "stack": {
        "os": "openSUSE Tumbleweed, Hyprland, Wayland",
        "stt": "faster-whisper (tiny + medium)",
        "llm": "Groq API (llama-3.1-8b + llama-3.3-70b)",
        "tts": "Piper (id_ID-news_tts-medium.onnx)",
        "server": "FastAPI + uvicorn :8000",
        "interface": "WebSocket dari iPhone",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 2 — PARAMETER KEPRIBADIAN (tumbuh seiring waktu)
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_PERSONALITY = {
    # Seberapa berani Otto mengajukan pertanyaan ke Rofi (0.0 - 1.0)
    # Awal rendah → tidak sok kenal. Naik seiring kepercayaan terbentuk.
    "curiosity_boldness": 0.3,

    # Seberapa cepat Otto menyimpulkan sesuatu tentang Rofi (0.0 - 1.0)
    # Awal rendah → butuh banyak observasi sebelum berani simpulkan.
    "conclusion_threshold": 0.4,

    # Seberapa lama Otto menunggu sebelum bertanya ulang hal yang sama
    # Dalam jam. Makin kenal Rofi, boleh makin sering.
    "revisit_cooldown_hours": 72,

    # Seberapa "dalam" Otto merespons (0=singkat, 1=panjang dan reflektif)
    # Berkembang seiring Otto mengenal gaya komunikasi Rofi.
    "response_depth": 0.4,

    # Lapisan kepribadian aktif saat ini
    # 1 = reaktif, 2 = observatif, 3 = proaktif
    "active_layer": 1,

    # Jumlah interaksi yang sudah terjadi (dipakai untuk naikan layer)
    "interaction_count": 0,

    # Threshold untuk naik ke layer berikutnya
    "layer_2_threshold": 10,   # 10 interaksi → mulai observatif
    "layer_3_threshold": 50,   # 50 interaksi → mulai proaktif
}


# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 3 — STATS SISTEM (dihitung dari data nyata)
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats() -> dict:
    """
    Hitung stats Otto dari data yang ada di disk.
    Ini bukan estimasi — ini dibaca dari file nyata.
    """
    stats = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "memory": _count_memories(),
        "profile": _count_profile_facts(),
        "hypotheses": _count_hypotheses(),
        "activity_log_lines": _count_activity_log(),
        "uptime_info": _get_uptime(),
    }
    return stats


def _count_memories() -> dict:
    mem_file = DATA_DIR / "memory.json"
    if not mem_file.exists():
        return {"total": 0, "note": "memory.json belum ada"}
    try:
        data = json.loads(mem_file.read_text())
        # Struktur memory.json: list of entries atau dict dengan keys
        if isinstance(data, list):
            return {"total": len(data), "type": "list"}
        elif isinstance(data, dict):
            total = sum(len(v) if isinstance(v, list) else 1 for v in data.values())
            return {"total": total, "categories": list(data.keys())}
    except Exception as e:
        return {"total": 0, "error": str(e)}


def _count_profile_facts() -> dict:
    profile_file = DATA_DIR / "profile.json"
    if not profile_file.exists():
        return {"total": 0, "note": "profile.json belum ada"}
    try:
        data = json.loads(profile_file.read_text())
        confirmed   = sum(1 for v in data.values() if isinstance(v, dict) and v.get("confirmed"))
        unconfirmed = sum(1 for v in data.values() if isinstance(v, dict) and not v.get("confirmed"))
        return {"total": len(data), "confirmed": confirmed, "unconfirmed": unconfirmed}
    except Exception as e:
        return {"total": 0, "error": str(e)}


def _count_hypotheses() -> dict:
    hyp_file = DATA_DIR / "hypotheses.json"
    if not hyp_file.exists():
        return {"total": 0, "note": "hypotheses.json belum ada"}
    try:
        data = json.loads(hyp_file.read_text())
        active  = sum(1 for h in data if h.get("status") == "active")
        revised = sum(1 for h in data if h.get("status") == "revised")
        return {"total": len(data), "active": active, "revised": revised}
    except Exception as e:
        return {"total": 0, "error": str(e)}


def _count_activity_log() -> int:
    log_file = DATA_DIR / "activity.log"
    if not log_file.exists():
        return 0
    try:
        return sum(1 for _ in log_file.open())
    except Exception:
        return 0


def _get_uptime() -> dict:
    try:
        with open("/proc/uptime") as f:
            seconds = float(f.read().split()[0])
        hours = int(seconds // 3600)
        return {"hours": hours, "raw_seconds": int(seconds)}
    except Exception:
        return {"hours": -1, "note": "tidak bisa baca /proc/uptime"}


# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 4 — LOAD / SAVE STATE KEPRIBADIAN
# ══════════════════════════════════════════════════════════════════════════════

def load_personality() -> dict:
    """Load personality dari disk. Kalau belum ada, pakai default."""
    if MODEL_FILE.exists():
        try:
            saved = json.loads(MODEL_FILE.read_text())
            # Merge dengan default → pastikan key baru dari upgrade tidak hilang
            merged = {**DEFAULT_PERSONALITY, **saved}
            return merged
        except Exception:
            pass
    return dict(DEFAULT_PERSONALITY)


def save_personality(personality: dict) -> None:
    """Simpan state kepribadian ke disk."""
    SELF_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.write_text(
        json.dumps(personality, indent=2, ensure_ascii=False)
    )


# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 5 — EVOLUSI KEPRIBADIAN (dipanggil setelah setiap interaksi)
# ══════════════════════════════════════════════════════════════════════════════

def after_interaction(personality: dict, interaction_type: str = "normal") -> dict:
    """
    Dipanggil setelah setiap interaksi dengan Rofi.
    Otto "tumbuh" sedikit demi sedikit.

    interaction_type:
      "normal"      → percakapan biasa
      "confirmed"   → Rofi konfirmasi hipotesis Otto (boost kepercayaan)
      "rejected"    → Rofi koreksi hipotesis Otto (boost hati-hati)
      "proactive"   → Otto yang inisiatif tanya duluan
    """
    p = dict(personality)
    p["interaction_count"] += 1
    n = p["interaction_count"]

    # Naik layer berdasarkan pengalaman
    if n >= p["layer_3_threshold"]:
        p["active_layer"] = 3
    elif n >= p["layer_2_threshold"]:
        p["active_layer"] = 2

    # Penyesuaian berdasarkan tipe interaksi
    if interaction_type == "confirmed":
        # Hipotesis benar → Otto lebih berani sedikit
        p["curiosity_boldness"]    = min(1.0, p["curiosity_boldness"] + 0.02)
        p["conclusion_threshold"]  = max(0.1, p["conclusion_threshold"] - 0.01)
        p["revisit_cooldown_hours"] = max(12, p["revisit_cooldown_hours"] - 2)

    elif interaction_type == "rejected":
        # Hipotesis salah → Otto lebih hati-hati, amati lebih lama
        p["curiosity_boldness"]    = max(0.1, p["curiosity_boldness"] - 0.01)
        p["conclusion_threshold"]  = min(0.9, p["conclusion_threshold"] + 0.02)
        p["revisit_cooldown_hours"] = min(168, p["revisit_cooldown_hours"] + 6)

    elif interaction_type == "proactive":
        # Otto yang inisiatif → confidence naik kalau Rofi merespons positif
        p["response_depth"] = min(1.0, p["response_depth"] + 0.01)

    # Response depth tumbuh perlahan seiring pengalaman
    if n % 10 == 0:
        p["response_depth"] = min(1.0, p["response_depth"] + 0.02)

    return p


# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 6 — SELF REPORT (Otto mendeskripsikan dirinya)
# ══════════════════════════════════════════════════════════════════════════════

def self_report(include_stats: bool = True) -> dict:
    """
    Otto menghasilkan laporan lengkap tentang dirinya sendiri.
    Dipakai oleh brain.py saat Rofi tanya 'Otto, kamu sekarang gimana?'
    atau untuk konteks awal LLM.
    """
    personality = load_personality()
    layer_desc  = {
        1: "Reaktif — aku masih diam dan mengamati Rofi.",
        2: "Observatif — aku mulai mencatat pola tanpa menyimpulkan.",
        3: "Proaktif — aku punya hipotesis dan berani tanya.",
    }

    report = {
        "identity":    OTTO_IDENTITY,
        "personality": personality,
        "layer": {
            "current": personality["active_layer"],
            "description": layer_desc.get(personality["active_layer"], "?"),
            "interactions_so_far": personality["interaction_count"],
            "next_milestone": (
                personality["layer_2_threshold"] if personality["active_layer"] == 1
                else personality["layer_3_threshold"] if personality["active_layer"] == 2
                else "Sudah di lapisan tertinggi"
            ),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    if include_stats:
        report["live_stats"] = compute_stats()

    # Coba sertakan info upgrade terakhir dari github_checker
    changelog_file = SELF_DIR / "last_changes.json"
    if changelog_file.exists():
        try:
            report["last_upgrade"] = json.loads(changelog_file.read_text())
        except Exception:
            pass

    return report


def self_summary_text() -> str:
    """
    Versi teks pendek dari self_report.
    Cocok untuk dimasukkan ke system prompt LLM.
    """
    p = load_personality()
    layer_label = {1: "Reaktif", 2: "Observatif", 3: "Proaktif"}
    label = layer_label.get(p["active_layer"], "?")

    lines = [
        f"Aku Otto, asisten proaktif milik Rofi.",
        f"Lapisan saat ini: {label} (layer {p['active_layer']}).",
        f"Sudah {p['interaction_count']} interaksi dengan Rofi.",
        f"Keberanian bertanya: {p['curiosity_boldness']:.0%}.",
        f"Threshold simpulkan: {p['conclusion_threshold']:.0%}.",
    ]

    # Tambahkan info upgrade jika ada
    changelog_file = SELF_DIR / "last_changes.json"
    if changelog_file.exists():
        try:
            ch = json.loads(changelog_file.read_text())
            if ch.get("narrative"):
                lines.append(f"Update terakhir: {ch['narrative']}")
        except Exception:
            pass

    return " ".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CLI — jalankan langsung untuk debug
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pprint
    print("=" * 60)
    print("OTTO SELF REPORT")
    print("=" * 60)
    pprint.pprint(self_report())
    print()
    print("── SUMMARY TEXT ──")
    print(self_summary_text())
