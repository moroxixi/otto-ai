# core/vocabulary.py
"""
Kamus kosakata Otto — sepenuhnya dinamis.
Tidak ada satu pun kata yang hardcoded di sini.

Semua entri disimpan di data/vocabulary.json.
Otto bisa tambah sendiri, Rofi bisa koreksi kapan saja.

Export yang dipakai transcriber.py (backward-compatible):
  - NAMA_ALIAS          → dict[str, str]  untuk _normalize_nama()
  - WHISPER_INITIAL_PROMPT → str          untuk initial_prompt Whisper

Fungsi untuk Otto (tulis):
  - tambah_alias(salah, benar, sumber)
  - tambah_istilah(kata, sumber)

Fungsi untuk Rofi (koreksi):
  - koreksi_alias(salah, benar_baru)
  - hapus_alias(salah)
  - get_pending_review()
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("otto.vocabulary")

# ── Path ke file kamus ──────────────────────────────────────────────────────
VOCAB_PATH = Path(__file__).parent.parent / "data" / "vocabulary.json"

_EMPTY_STORE: dict = {"version": 1, "alias": {}, "istilah": []}


# ── I/O ─────────────────────────────────────────────────────────────────────

def _load() -> dict:
    if not VOCAB_PATH.exists():
        _save(_EMPTY_STORE.copy())
        logger.info("[vocab] vocabulary.json dibuat baru di %s", VOCAB_PATH)
        return _EMPTY_STORE.copy()
    try:
        return json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("[vocab] Gagal baca vocabulary.json: %s — pakai store kosong", e)
        return _EMPTY_STORE.copy()


def _save(data: dict) -> None:
    try:
        VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
        VOCAB_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception as e:
        logger.error("[vocab] Gagal simpan vocabulary.json: %s", e)


# ── Fungsi tulis (dipakai Otto) ─────────────────────────────────────────────

def tambah_alias(salah: str, benar: str, sumber: str = "otto") -> bool:
    """
    Tambah alias koreksi typo Whisper.
    Return True kalau berhasil, False kalau duplikat.

    sumber: "otto" = Otto deteksi sendiri (konfirmasi=False, perlu review Rofi)
            "rofi" = Rofi input langsung    (konfirmasi=True, langsung aktif)
    """
    salah = salah.lower().strip()
    benar = benar.strip()
    if not salah or not benar:
        return False

    data = _load()
    if salah in data["alias"]:
        logger.debug("[vocab] Alias '%s' sudah ada, skip.", salah)
        return False  # duplikat — tidak overwrite

    data["alias"][salah] = {
        "benar":      benar,
        "sumber":     sumber,
        "konfirmasi": sumber == "rofi",   # otto → False, rofi → True
        "ditambah":   datetime.now().isoformat(timespec="minutes"),
    }
    _save(data)
    logger.info("[vocab] Alias baru: '%s' → '%s' (sumber=%s)", salah, benar, sumber)
    return True


def tambah_istilah(kata: str, sumber: str = "otto") -> bool:
    """
    Tambah kata/frasa ke vocab hint Whisper (initial_prompt).
    Return True kalau berhasil, False kalau duplikat.
    """
    kata = kata.strip()
    if not kata:
        return False

    data = _load()
    existing = {e["kata"].lower() for e in data["istilah"]}
    if kata.lower() in existing:
        logger.debug("[vocab] Istilah '%s' sudah ada, skip.", kata)
        return False

    data["istilah"].append({
        "kata":       kata,
        "sumber":     sumber,
        "konfirmasi": sumber == "rofi",
        "ditambah":   datetime.now().isoformat(timespec="minutes"),
    })
    _save(data)
    logger.info("[vocab] Istilah baru: '%s' (sumber=%s)", kata, sumber)
    return True


# ── Fungsi koreksi (dipakai Rofi) ───────────────────────────────────────────

def koreksi_alias(salah: str, benar_baru: str) -> bool:
    """
    Rofi koreksi nilai alias yang Otto simpan salah.
    Otomatis set konfirmasi=True setelah dikoreksi.
    """
    data = _load()
    if salah not in data["alias"]:
        logger.warning("[vocab] koreksi_alias: '%s' tidak ditemukan", salah)
        return False

    data["alias"][salah]["benar"]      = benar_baru.strip()
    data["alias"][salah]["konfirmasi"] = True
    data["alias"][salah]["dikoreksi"]  = datetime.now().isoformat(timespec="minutes")
    _save(data)
    logger.info("[vocab] Alias dikoreksi: '%s' → '%s'", salah, benar_baru)
    return True


def konfirmasi_alias(salah: str) -> bool:
    """Rofi konfirmasi alias Otto tanpa perlu ubah nilainya."""
    data = _load()
    if salah not in data["alias"]:
        return False
    data["alias"][salah]["konfirmasi"] = True
    _save(data)
    return True


def hapus_alias(salah: str) -> bool:
    """Hapus alias (kalau Otto salah input atau tidak relevan)."""
    data = _load()
    if salah not in data["alias"]:
        return False
    del data["alias"][salah]
    _save(data)
    logger.info("[vocab] Alias dihapus: '%s'", salah)
    return True


def hapus_istilah(kata: str) -> bool:
    """Hapus istilah dari vocab hint."""
    data = _load()
    sebelum = len(data["istilah"])
    data["istilah"] = [e for e in data["istilah"] if e["kata"].lower() != kata.lower()]
    if len(data["istilah"]) < sebelum:
        _save(data)
        logger.info("[vocab] Istilah dihapus: '%s'", kata)
        return True
    return False


# ── Query ────────────────────────────────────────────────────────────────────

def get_pending_review() -> list[dict]:
    """
    Kembalikan semua entri yang Otto tambah tapi belum Rofi konfirmasi.
    Dipakai Otto untuk lapor: "Rofi, aku simpan beberapa kata baru, cek ya."
    """
    data = _load()
    pending = []

    for salah, info in data["alias"].items():
        if not info.get("konfirmasi", True):
            pending.append({
                "tipe":     "alias",
                "salah":    salah,
                "benar":    info["benar"],
                "ditambah": info.get("ditambah", ""),
            })

    for item in data["istilah"]:
        if not item.get("konfirmasi", True):
            pending.append({
                "tipe":     "istilah",
                "kata":     item["kata"],
                "ditambah": item.get("ditambah", ""),
            })

    return pending


# ── Build produk untuk transcriber.py ───────────────────────────────────────

def get_alias_map() -> dict[str, str]:
    """
    Kembalikan alias yang sudah dikonfirmasi saja.
    Ini yang aktif dipakai _normalize_nama() di transcriber.py.
    """
    data = _load()
    return {
        salah: info["benar"]
        for salah, info in data["alias"].items()
        if info.get("konfirmasi", False)
    }


def build_initial_prompt() -> str:
    """
    Buat initial_prompt untuk faster-whisper dari istilah yang dikonfirmasi.
    Kalau belum ada istilah sama sekali, kembalikan string kosong
    (Whisper tetap jalan normal, tanpa hint).
    """
    data = _load()
    istilah_aktif = [
        e["kata"] for e in data["istilah"]
        if e.get("konfirmasi", False)
    ]
    if not istilah_aktif:
        return ""
    # Whisper lebih responsif ke kalimat natural daripada list kata mentah
    return "Nama: " + ", ".join(istilah_aktif[:30]) + "."


# ── Export backward-compatible untuk transcriber.py ─────────────────────────
# transcriber.py import langsung:
#   from core.vocabulary import WHISPER_INITIAL_PROMPT, NAMA_ALIAS
#
# Keduanya di-build sekali saat import. Kalau vocabulary.json diupdate
# saat runtime, transcriber perlu reload — atau pakai get_alias_map()
# secara langsung (lihat catatan implementasi di bawah).

NAMA_ALIAS: dict[str, str]   = get_alias_map()
WHISPER_INITIAL_PROMPT: str  = build_initial_prompt()
