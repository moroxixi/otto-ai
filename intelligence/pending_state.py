"""
intelligence/pending_state.py — Persistent Curiosity Pending State
==================================================================
Ganti memory.set_temp("curiosity_pending") yang hilang saat restart.

Masalah lama:
  memory.set_temp() → simpan di RAM (_temp dict)
  Server restart → dict hilang → curiosity_pending = None
  Rofi jawab pertanyaan setelah restart → verdict tidak diproses
  Hipotesis stuck di "pending" selamanya

Solusi:
  PendingState simpan hyp_id ke file JSON kecil di disk.
  Survive restart, atomic write, thread-safe.

Penggunaan (di app.py — ganti memory.set_temp / get_temp / delete_temp):
  from intelligence.pending_state import pending_state

  # Simpan (sebelum tanya Rofi):
  pending_state.set(hyp_id)

  # Cek (saat Rofi jawab):
  hyp_id = pending_state.get()

  # Hapus (setelah diproses):
  pending_state.clear()

  # Cek apakah expired (Rofi tidak jawab > 30 menit):
  if pending_state.is_expired():
      pending_state.clear()
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("otto.intelligence.pending_state")

# File kecil — hanya satu field hyp_id + timestamp
STATE_FILE = Path("/data/asd/otto-ai/data/curiosity_pending.json")

# Berapa lama pending dianggap valid (detik)
# Jika Rofi tidak jawab dalam 30 menit → expired → auto-clear
PENDING_TTL = 30 * 60  # 30 menit


class PendingState:
    """
    Simpan satu pending hypothesis ID ke disk.
    Survive restart. Atomic write. Simple.
    """

    def __init__(self, path: Path = STATE_FILE, ttl: int = PENDING_TTL) -> None:
        self._path = path
        self._ttl  = ttl
        # Cache in-memory untuk avoid disk read berulang
        self._cached_id:  Optional[str]   = None
        self._cached_at:  Optional[float] = None
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def set(self, hyp_id: str) -> None:
        """Simpan hypothesis ID yang sedang menunggu jawaban Rofi."""
        self._cached_id = hyp_id
        self._cached_at = time.time()
        self._save()
        logger.info("[pending] Set pending: %s", hyp_id)

    def get(self) -> Optional[str]:
        """
        Ambil pending hypothesis ID.
        Return None jika tidak ada atau sudah expired.
        """
        if self._cached_id is None:
            return None

        if self.is_expired():
            logger.info("[pending] Pending %s expired, auto-clear.", self._cached_id)
            self.clear()
            return None

        return self._cached_id

    def clear(self) -> None:
        """Hapus pending state — dipanggil setelah jawaban diproses."""
        old = self._cached_id
        self._cached_id = None
        self._cached_at = None
        self._save()
        if old:
            logger.info("[pending] Clear pending: %s", old)

    def is_expired(self) -> bool:
        """True jika pending sudah lebih dari TTL detik."""
        if self._cached_at is None:
            return False
        return (time.time() - self._cached_at) > self._ttl

    def has_pending(self) -> bool:
        """True jika ada pending yang masih valid."""
        return self.get() is not None

    # ── Persistensi ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        """Atomic write ke disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "hyp_id":    self._cached_id,
            "set_at":    self._cached_at,
            "expires_at": (self._cached_at + self._ttl) if self._cached_at else None,
        }
        # Write ke temp file dulu, lalu rename (atomic)
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            tmp.replace(self._path)
        except OSError as e:
            logger.error("[pending] Gagal simpan: %s", e)

    def _load(self) -> None:
        """Load dari disk saat startup."""
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._cached_id = data.get("hyp_id")
            self._cached_at = data.get("set_at")
            if self._cached_id:
                if self.is_expired():
                    logger.info(
                        "[pending] Startup: pending %s ditemukan tapi sudah expired — clear.",
                        self._cached_id,
                    )
                    self.clear()
                else:
                    remaining = int((self._cached_at + self._ttl) - time.time()) // 60
                    logger.info(
                        "[pending] Startup: pending %s di-recover, sisa ~%d menit.",
                        self._cached_id, remaining,
                    )
        except Exception as e:
            logger.warning("[pending] Gagal load state: %s", e)
            self._cached_id = None
            self._cached_at = None

    def __repr__(self) -> str:
        if self._cached_id is None:
            return "<PendingState: kosong>"
        expired = "EXPIRED" if self.is_expired() else "valid"
        return f"<PendingState: {self._cached_id} ({expired})>"


# ─────────────────────────── Singleton ───────────────────────────────────────

pending_state = PendingState()
