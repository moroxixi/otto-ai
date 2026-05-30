# core/shortcut.py
# Shortcut untuk perintah yang tidak punya skill (Tipe B)
# Cara kerja: LLM jawab sekali → disimpan → request berikutnya bypass LLM

import json
import time
import logging
from pathlib import Path
from core.config import PATHS

logger = logging.getLogger("otto.shortcut")

SHORTCUT_FILE = PATHS["base"] / "data" / "shortcuts.json"

# Satu-satunya pengecualian: reminder selalu punya parameter unik
# "ingatkan 10 menit" vs "ingatkan besok jam 8" → tidak bisa di-cache
_SKIP_KEYWORDS = {"ingatkan", "remind", "pengingat"}


def _should_skip(text: str) -> bool:
    """Return True jika teks ini tidak boleh di-shortcut."""
    low = text.lower()
    return any(kw in low for kw in _SKIP_KEYWORDS)


def _normalize(text: str) -> str:
    return text.strip().lower()


def _load() -> dict:
    if not SHORTCUT_FILE.exists():
        return {}
    try:
        with open(SHORTCUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(data: dict) -> None:
    SHORTCUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SHORTCUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def check(text: str) -> dict | None:
    """
    Cek shortcut. Return cached result jika ada, None jika tidak.
    Dipanggil di executor SETELAH skill check gagal.
    """
    if _should_skip(text):
        return None

    key = _normalize(text)
    data = _load()
    entry = data.get(key)

    if entry:
        logger.info("[shortcut] HIT '%s' (sudah %dx)", key, entry.get("count", 1))
        return entry["result"]
    return None


def record(text: str, result: dict) -> None:
    """
    Simpan hasil LLM ke shortcut store.
    Dipanggil di executor setelah brain.think() berhasil.
    """
    if _should_skip(text):
        logger.debug("[shortcut] SKIP record (reminder/pengingat): '%s'", text[:40])
        return

    key = _normalize(text)
    data = _load()

    prev_count = data.get(key, {}).get("count", 0)
    data[key] = {
        "count":     prev_count + 1,
        "result":    result,
        "saved_at":  time.time(),
    }

    _save(data)
    logger.info("[shortcut] SAVED '%s' (total %dx)", key, data[key]["count"])
