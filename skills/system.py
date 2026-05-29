"""
skills/system.py — Skill Sistem Otto
======================================
Perintah yang didukung:
  - Matikan / shutdown laptop
  - Volume naik / turun (wpctl via PipeWire)

Cara daftar ke executor (dari app.py):
    from skills.system import register_system_skills
    register_system_skills(executor)
"""

import asyncio
import logging

logger = logging.getLogger("otto.skills.system")


# ─────────────────────────── Helpers ────────────────────────────────────────

async def _run(cmd: str) -> tuple[int, str]:
    """Jalankan perintah shell, kembalikan (returncode, output)."""
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    return proc.returncode, out.decode().strip()


# ─────────────────────────── Handlers ────────────────────────────────────────

async def handle_shutdown(text: str, match, brain, **_):
    """Matikan laptop setelah 3 detik (beri waktu TTS selesai bicara)."""
    logger.info("[system] Shutdown diminta.")

    async def _do_shutdown():
        await asyncio.sleep(3)
        await _run("systemctl poweroff")

    asyncio.create_task(_do_shutdown())
    return "Oke Rofi, laptop akan mati dalam 3 detik."


async def handle_volume_up(text: str, match, brain, **_):
    """Naikkan volume 10% via wpctl (PipeWire)."""
    code, out = await _run("wpctl set-volume @DEFAULT_AUDIO_SINK@ 10%+")
    if code == 0:
        # Ambil volume sekarang untuk laporan
        _, vol_out = await _run(
            "wpctl get-volume @DEFAULT_AUDIO_SINK@ | awk '{printf \"%d\", $2 * 100}'"
        )
        logger.info("[system] Volume naik → %s%%", vol_out)
        return f"Volume naik. Sekarang sekitar {vol_out} persen."
    logger.error("[system] volume_up gagal: %s", out)
    return "Gagal menaikkan volume."


async def handle_volume_down(text: str, match, brain, **_):
    """Turunkan volume 10% via wpctl (PipeWire)."""
    code, out = await _run("wpctl set-volume @DEFAULT_AUDIO_SINK@ 10%-")
    if code == 0:
        _, vol_out = await _run(
            "wpctl get-volume @DEFAULT_AUDIO_SINK@ | awk '{printf \"%d\", $2 * 100}'"
        )
        logger.info("[system] Volume turun → %s%%", vol_out)
        return f"Volume turun. Sekarang sekitar {vol_out} persen."
    logger.error("[system] volume_down gagal: %s", out)
    return "Gagal menurunkan volume."


# ─────────────────────────── Register ────────────────────────────────────────

def register_system_skills(executor) -> None:
    """Daftarkan semua skill system ke executor. Panggil sekali dari app.py."""

    executor.register(
        name     = "shutdown",
        pattern  = r"\b(matiin|matikan|shutdown|turn off|power off)\s*(laptop|komputer|pc|mesin)?\b",
        handler  = handle_shutdown,
        examples = ["matiin laptop", "shutdown", "matikan komputer"],
    )

    executor.register(
        name     = "volume_up",
        pattern  = r"\b(volume|suara|keras)\s*(naik|up|tambah|kerasin|besarin|lebih keras)\b"
                   r"|\b(naikkan|kerasin|besarin)\s*(volume|suara)\b",
        handler  = handle_volume_up,
        examples = ["volume naik", "kerasin suara", "naikkan volume"],
    )

    executor.register(
        name     = "volume_down",
        pattern  = r"\b(volume|suara)\s*(turun|down|kurangi|kecilin|lebih kecil)\b"
                   r"|\b(turunkan|kecilin|kurangi)\s*(volume|suara)\b",
        handler  = handle_volume_down,
        examples = ["volume turun", "kecilin suara", "turunkan volume"],
    )

    logger.info("[system] 3 skill system terdaftar: shutdown, volume_up, volume_down")
