"""
skills/reminder.py — Skill Pengingat & Tracker Otto
======================================================
Dua fungsi utama:

1. PENGINGAT (one-shot)
   Kirim notifikasi + TTS pada waktu tertentu.
   Format: "30m", "2h", "10:30"
   Contoh ucapan:
     "ingatkan aku minum obat 30 menit lagi"
     "kasih tau aku meeting jam 14:30"

2. TRACKER DATA PRIBADI
   Simpan data series (berat badan, tekanan darah, dll) ke memory Otto.
   Contoh ucapan:
     "catat berat badanku 68 kg"
     "berat badanku sekarang 70"
     "lihat catatan berat badan"
     "catat tensi 120 per 80"

Notifikasi dikirim via:
  - notify-send (desktop)
  - ntfy-bridge.sh (iPhone push notification)
  - Piper TTS → pw-play ke sink 58

Cara daftar ke executor (dari app.py):
    from skills.reminder import register_reminder_skills
    register_reminder_skills(executor)
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger("otto.skills.reminder")

# ─────────────────────────── Konfigurasi ────────────────────────────────────

PIPER_BIN   = "/usr/local/bin/piper"
VOICE_MODEL = "/data/asd/libo-ai/voices/id_ID-news_tts-medium.onnx"
AUDIO_SINK  = 58
NTFY_BRIDGE = "/root/.local/bin/ntfy-bridge.sh"  # sesuaikan jika perlu

# File JSON untuk menyimpan data tracker (berat badan, dsb)
TRACKER_FILE = Path("/data/asd/otto-ai/data/tracker.json")


# ─────────────────────────── Notifikasi ──────────────────────────────────────

async def _run(cmd: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    return proc.returncode, out.decode().strip()


async def _notify(judul: str, pesan: str) -> None:
    """Kirim notif desktop + push iPhone + TTS."""
    # 1. Desktop notif
    await _run(f'notify-send "Pengingat!" "{pesan}" -a "Otto" -u critical -i "appointment-soon"')

    # 2. iPhone push
    await _run(f'{NTFY_BRIDGE} "Pengingat!" "{pesan}" 2>/dev/null || true')

    # 3. TTS
    tts_text = f"Bos Rofi, ini pengingat untuk: {pesan}."
    tts_cmd  = (
        f'echo "{tts_text}" | {PIPER_BIN} '
        f'--model {VOICE_MODEL} '
        f'--output_file /tmp/otto_reminder.wav && '
        f'pw-play --target {AUDIO_SINK} /tmp/otto_reminder.wav ; '
        f'rm -f /tmp/otto_reminder.wav'
    )
    await _run(tts_cmd)
    logger.info("[reminder] Notif terkirim: %s", pesan)


# ─────────────────────────── Parser Waktu ────────────────────────────────────

def _parse_delay(waktu_str: str) -> int | None:
    """
    Parse string waktu → detik.
    Format yang diterima: "30m", "2h", "30 menit", "2 jam", "10:30"
    Return None jika tidak valid.
    """
    s = waktu_str.strip().lower()

    # "30m" atau "30 menit"
    m = re.search(r'(\d+)\s*m(?:enit)?', s)
    if m:
        return int(m.group(1)) * 60

    # "2h" atau "2 jam"
    m = re.search(r'(\d+)\s*(?:h|jam)', s)
    if m:
        return int(m.group(1)) * 3600

    # "10:30" — jam spesifik hari ini / besok
    m = re.match(r'^(\d{1,2}):(\d{2})$', s)
    if m:
        now    = datetime.now()
        target = now.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return int((target - now).total_seconds())

    return None


# ─────────────────────────── Tracker (Data Seri) ─────────────────────────────

def _load_tracker() -> dict:
    TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    if TRACKER_FILE.exists():
        try:
            return json.loads(TRACKER_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_tracker(data: dict) -> None:
    TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRACKER_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _tracker_add(kategori: str, nilai: str, satuan: str = "") -> str:
    """Tambah satu entri baru ke kategori tracker."""
    data = _load_tracker()
    if kategori not in data:
        data[kategori] = []

    entri = {
        "waktu": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "nilai": nilai,
        "satuan": satuan,
    }
    data[kategori].append(entri)
    _save_tracker(data)

    label = f"{nilai} {satuan}".strip()
    logger.info("[tracker] +%s → %s", kategori, label)
    return label


def _tracker_get(kategori: str, n: int = 5) -> list[dict]:
    """Ambil n entri terakhir dari kategori."""
    data = _load_tracker()
    return data.get(kategori, [])[-n:]


# ─────────────────────────── Handlers ────────────────────────────────────────

async def handle_reminder(text: str, match, brain, **_):
    """
    Contoh:
      "ingatkan aku minum obat 30 menit lagi"
      "kasih tau aku meeting jam 14:00"
      "ingatkan beli susu 2 jam lagi"
    """
    # Group 1 = isi pengingat, Group 2 = waktu
    pesan     = match.group(1).strip() if match.lastindex >= 1 else text
    waktu_str = match.group(2).strip() if match.lastindex >= 2 else ""

    detik = _parse_delay(waktu_str) if waktu_str else None

    if detik is None:
        return (
            "Format waktu tidak aku kenali. "
            "Coba: '30 menit lagi', '2 jam lagi', atau 'jam 14:30'."
        )

    menit = detik // 60
    jam   = menit // 60

    if jam > 0:
        label_waktu = f"{jam} jam lagi" if menit % 60 == 0 else f"{jam} jam {menit % 60} menit lagi"
    else:
        label_waktu = f"{menit} menit lagi"

    async def _delayed():
        await asyncio.sleep(detik)
        await _notify("Pengingat", pesan)

    asyncio.create_task(_delayed())
    logger.info("[reminder] Terjadwal '%s' dalam %ds", pesan, detik)
    return f"Oke, aku ingatkan '{pesan}' {label_waktu}."


async def handle_track_weight(text: str, match, brain, **_):
    """Catat berat badan. Contoh: 'catat berat badanku 68 kg'"""
    nilai = (match.group(1) or match.group(2)).strip()
    label = _tracker_add("berat_badan", nilai, "kg")

    # Simpan ke memory Otto juga
    brain.memory.remember(
        f"rofi.berat_badan.{datetime.now().strftime('%Y%m%d')}",
        f"{label} pada {datetime.now().strftime('%d %b %Y')}",
        source="tracker",
    )

    # Cek tren (jika ada data sebelumnya)
    riwayat = _tracker_get("berat_badan", n=5)
    tren    = ""
    if len(riwayat) >= 2:
        try:
            sebelumnya = float(riwayat[-2]["nilai"])
            sekarang   = float(nilai)
            selisih    = sekarang - sebelumnya
            if selisih > 0:
                tren = f" Naik {selisih:.1f} kg dari catatan sebelumnya."
            elif selisih < 0:
                tren = f" Turun {abs(selisih):.1f} kg dari catatan sebelumnya."
            else:
                tren = " Sama seperti catatan sebelumnya."
        except ValueError:
            pass

    return f"Berat badan {label} sudah aku catat.{tren}"


async def handle_view_weight(text: str, match, brain, **_):
    """Tampilkan riwayat berat badan."""
    riwayat = _tracker_get("berat_badan", n=7)
    if not riwayat:
        return "Belum ada catatan berat badan."

    lines = ["Riwayat berat badan kamu:"]
    for e in riwayat:
        lines.append(f"  {e['waktu']}  →  {e['nilai']} {e['satuan']}".strip())
    return "\n".join(lines)


async def handle_track_custom(text: str, match, brain, **_):
    """
    Tracker generik untuk data lain.
    Contoh: 'catat tensi 120/80', 'catat gula darah 95'
    """
    kategori = match.group(1).strip().lower().replace(" ", "_")
    nilai    = match.group(2).strip()
    label    = _tracker_add(kategori, nilai)

    brain.memory.remember(
        f"rofi.{kategori}.{datetime.now().strftime('%Y%m%d')}",
        f"{label} pada {datetime.now().strftime('%d %b %Y')}",
        source="tracker",
    )
    return f"Oke, {kategori.replace('_', ' ')} {label} sudah aku catat."


# ─────────────────────────── Register ────────────────────────────────────────

def register_reminder_skills(executor) -> None:
    """Daftarkan semua skill reminder ke executor. Panggil sekali dari app.py."""

    executor.register(
        name     = "reminder",
        pattern  = r"\b(?:ingatkan|kasih\s+tau|remind|alert)\s+(?:aku\s+)?(.+?)\s+"
                   r"(?:dalam\s+)?(\d+\s*(?:m|menit|h|jam)|jam\s+\d{1,2}:\d{2}|\d{1,2}:\d{2})"
                   r"(?:\s+lagi)?\b",
        handler  = handle_reminder,
        examples = [
            "ingatkan aku minum obat 30 menit lagi",
            "kasih tau aku meeting jam 14:00",
            "ingatkan beli susu 2 jam lagi",
        ],
    )

    executor.register(
        name     = "track_weight",
        pattern  = r"\b(?:catat|simpan|tulis)\s+berat\s+(?:badan)?(?:ku|saya)?\s*(\d+(?:[.,]\d+)?)\s*(?:kg)?\b"
                   r"|\bberat\s+(?:badan)?(?:ku|saya)?\s+(?:sekarang\s+)?(\d+(?:[.,]\d+)?)\s*(?:kg)?\b",
        handler  = handle_track_weight,
        examples = [
            "catat berat badanku 68 kg",
            "berat badanku sekarang 70",
            "simpan berat badan 65.5",
        ],
    )

    executor.register(
        name     = "view_weight",
        pattern  = r"\b(?:lihat|tampilkan|cek|show)\s+(?:catatan\s+)?berat\s+badan\b",
        handler  = handle_view_weight,
        examples = ["lihat catatan berat badan", "cek berat badan"],
    )

    executor.register(
        name     = "track_custom",
        pattern  = r"\b(?:catat|simpan|tulis)\s+(tensi|tekanan\s+darah|gula\s+darah|kolesterol|detak\s+jantung)\s+"
                   r"(\d[\d/.,\s]*(?:mmhg|mg|bpm)?)\b",
        handler  = handle_track_custom,
        examples = [
            "catat tensi 120/80",
            "catat gula darah 95",
            "simpan kolesterol 180",
        ],
    )

    logger.info("[reminder] 4 skill reminder terdaftar: reminder, track_weight, view_weight, track_custom")
