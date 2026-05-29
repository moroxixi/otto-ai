"""
skills/media.py — Skill Media Otto
=====================================
Kontrol musik via mpv dengan IPC socket.

Lagu yang tersedia:
  - santai  → /data/asd/musik/santai.mp3
  - jadul   → /data/asd/musik/jadul.mp3

Cara daftar ke executor (dari app.py):
    from skills.media import register_media_skills
    register_media_skills(executor)
"""

import asyncio
import json
import logging
import os

logger = logging.getLogger("otto.skills.media")

# ─────────────────────────── Konfigurasi ────────────────────────────────────

MPV_SOCKET  = "/tmp/otto-mpv.sock"
AUDIO_SINK  = 58   # PipeWire sink ID (dari config)

SONGS = {
    "santai": "/data/asd/musik/santai.mp3",
    "jadul":  "/data/asd/musik/jadul.mp3",
}


# ─────────────────────────── MPV Helpers ────────────────────────────────────

async def _mpv_running() -> bool:
    """Cek apakah mpv socket aktif."""
    return os.path.exists(MPV_SOCKET)


async def _mpv_send(command: list) -> bool:
    """Kirim perintah JSON ke mpv IPC socket. Return True jika berhasil."""
    if not await _mpv_running():
        return False
    try:
        payload = json.dumps({"command": command}) + "\n"
        reader, writer = await asyncio.open_unix_connection(MPV_SOCKET)
        writer.write(payload.encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        return True
    except Exception as e:
        logger.warning("[media] mpv IPC error: %s", e)
        return False


async def _stop_current() -> None:
    """Hentikan mpv yang sedang berjalan (jika ada)."""
    if await _mpv_running():
        await _mpv_send(["quit"])
        await asyncio.sleep(0.3)   # tunggu proses benar-benar berhenti


async def _play_file(path: str) -> tuple[bool, str]:
    """
    Jalankan mpv dengan file path.
    Pakai --audio-device untuk arahkan ke sink PipeWire yang benar.
    """
    if not os.path.exists(path):
        return False, f"File tidak ditemukan: {path}"

    # Hentikan yang sedang main dulu
    await _stop_current()

    cmd = (
        f"mpv --no-video "
        f"--input-ipc-server={MPV_SOCKET} "
        f"--audio-device=pipewire/sink-{AUDIO_SINK} "
        f"'{path}' "
        f"--really-quiet &"
    )
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await asyncio.sleep(0.5)   # beri waktu mpv mulai dan buat socket
    logger.info("[media] mpv started → %s", path)
    return True, ""


# ─────────────────────────── Handlers ────────────────────────────────────────

async def handle_play_santai(text: str, match, brain, **_):
    ok, err = await _play_file(SONGS["santai"])
    if ok:
        return "Oke, memutar lagu santai."
    return f"Gagal putar lagu santai. {err}"


async def handle_play_jadul(text: str, match, brain, **_):
    ok, err = await _play_file(SONGS["jadul"])
    if ok:
        return "Oke, memutar lagu jadul."
    return f"Gagal putar lagu jadul. {err}"


async def handle_stop_music(text: str, match, brain, **_):
    if not await _mpv_running():
        return "Tidak ada musik yang sedang diputar."
    await _stop_current()
    return "Musik dihentikan."


async def handle_pause_music(text: str, match, brain, **_):
    """Toggle pause/resume."""
    ok = await _mpv_send(["cycle", "pause"])
    if ok:
        return "Oke."
    return "Tidak ada musik yang sedang diputar."


# ─────────────────────────── Register ────────────────────────────────────────

def register_media_skills(executor) -> None:
    """Daftarkan semua skill media ke executor. Panggil sekali dari app.py."""

    executor.register(
        name     = "play_santai",
        pattern  = r"\b(putar|play|nyalain|mainkan)\s*(lagu\s*)?(santai|relax|relaxing)\b",
        handler  = handle_play_santai,
        examples = ["putar lagu santai", "play santai"],
    )

    executor.register(
        name     = "play_jadul",
        pattern  = r"\b(putar|play|nyalain|mainkan)\s*(lagu\s*)?(jadul|lawas|old|nostalgia)\b",
        handler  = handle_play_jadul,
        examples = ["putar lagu jadul", "play jadul", "nyalain lagu lawas"],
    )

    executor.register(
        name     = "stop_music",
        pattern  = r"\b(stop|hentikan|matiin|pause|berhenti)\s*(musik|lagu|mpv|music)\b"
                   r"|\b(stop|hentikan)\s*musik\b",
        handler  = handle_stop_music,
        examples = ["stop musik", "hentikan lagu", "matiin musik"],
    )

    executor.register(
        name     = "pause_music",
        pattern  = r"\b(pause|resume|lanjutin|lanjutkan)\s*(musik|lagu)?\b",
        handler  = handle_pause_music,
        examples = ["pause", "resume musik", "lanjutin lagu"],
    )

    logger.info("[media] 4 skill media terdaftar: play_santai, play_jadul, stop, pause")
