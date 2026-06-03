# core/memory.py
# Sistem ingatan Otto — dua lapis: pendek (RAM) & panjang (disk)
#
# Short-term : percakapan terakhir, dibawa ke konteks LLM tiap request
# Long-term  : fakta penting yang sudah diverifikasi, disimpan ke JSON

import json
import time
import logging
from pathlib import Path
from typing import Optional
from collections import deque

from core.config import PATHS, MEMORY

logger = logging.getLogger("otto.core.memory")

SHORT_TERM_PERSIST_PATH = PATHS["short_term_cache"]

class MemoryManager:
    """
    Kelola dua lapis ingatan Otto:
      - short_term : deque of {role, content, timestamp}
      - long_term  : dict tersimpan di disk, key = topik/label
    """

    def __init__(self):
        self.memory_path: Path = PATHS["memory"]
        self._short_term: deque = deque(maxlen=MEMORY["short_term_limit"])
        self._load_short_term()
        self._long_term: dict = {}
        self._load_long_term()
        self._temp: dict[str, str] = {}

        # Cache fingerprint — untuk deteksi perubahan long-term
        # Brain pakai ini untuk tahu kapan harus rebuild system prompt
        self._long_term_version: int = 0

    # ─── SHORT TERM ───────────────────────────────────────────────────────────

    def add_message(self, role: str, content: str) -> None:
        """
        Tambah pesan ke short-term memory.
        role: "user" | "assistant" | "system"
        """
        self._short_term.append({
            "role":      role,
            "content":   content,
            "timestamp": time.time(),
        })

    def get_short_term(self) -> list[dict]:
        """
        Kembalikan pesan untuk dikirim ke LLM.
        Hanya field role + content (timestamp tidak perlu dikirim ke Groq).
        """
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self._short_term
        ]

    def get_recent_messages(self, limit: int = 20) -> list[dict]:
        """
        Kembalikan N pesan terakhir untuk konteks LLM.
        Alias bersih dari get_short_term() dengan batas jumlah.

        Dipanggil dari app.py sebelum brain.think().
        """
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in self._short_term
        ]
        return messages[-limit:]

    def clear_short_term(self) -> None:
        """Reset percakapan — misal saat sesi baru dimulai."""
        self._short_term.clear()

    def persist_short_term(self, max_messages: int = 20) -> None:
        """
        Simpan N pesan terakhir ke disk.
        Dipanggil saat shutdown agar Otto tidak amnesia total setelah restart.
        """
        messages = list(self._short_term)[-max_messages:]
        if not messages:
            logger.info("[memory] Short-term kosong, skip persist.")
            return
        try:
            SHORT_TERM_PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
            SHORT_TERM_PERSIST_PATH.write_text(
                json.dumps(messages, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            logger.info(
                "[memory] Short-term cache disimpan: %d pesan.", len(messages)
            )
        except OSError as e:
            logger.error("[memory] Gagal simpan short-term cache: %s", e)
    
    def _load_short_term(self) -> None:
        """
        Muat kembali pesan terakhir dari disk saat startup.
        Sehingga Otto ingat konteks percakapan sebelum restart.
        """
        if not SHORT_TERM_PERSIST_PATH.exists():
            return
        try:
            messages = json.loads(
                SHORT_TERM_PERSIST_PATH.read_text(encoding="utf-8")
            )
            if not isinstance(messages, list):
                return
            for msg in messages:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    # Pastikan ada timestamp, fallback ke 0 jika tidak ada
                    if "timestamp" not in msg:
                        msg["timestamp"] = 0.0
                    self._short_term.append(msg)
            logger.info(
                "[memory] Short-term cache dimuat: %d pesan dari sesi sebelumnya.",
                len(messages)
            )
        except Exception as e:
            logger.warning("[memory] Gagal load short-term cache: %s", e)

    def short_term_count(self) -> int:
        return len(self._short_term)

    # ─── LONG TERM ────────────────────────────────────────────────────────────

    def remember(self, key: str, value, source: str = "manual") -> None:
        """
        Simpan fakta ke long-term memory.

        key    : label unik, misal "rofi.kebiasaan.pagi" atau "otto.versi"
        value  : string, angka, list, dict — apapun yang JSON-serializable
        source : dari mana info ini ("observasi", "konfirmasi_rofi", "manual")

        Contoh:
            memory.remember("rofi.minuman.favorit", "kopi oat", "konfirmasi_rofi")
        """
        self._long_term[key] = {
            "value":      value,
            "source":     source,
            "updated_at": time.time(),
            "confirmed":  source == "konfirmasi_rofi",
        }
        self._long_term_version += 1   # ← tandai ada perubahan
        self._save_long_term()

    def forget(self, key: str) -> bool:
        """
        Hapus satu fakta dari long-term memory.
        Return True jika berhasil, False jika key tidak ditemukan.
        """
        if key in self._long_term:
            del self._long_term[key]
            self._long_term_version += 1   # ← tandai ada perubahan
            self._save_long_term()
            return True
        return False

    def recall(self, key: str, default=None):
        """Ambil nilai dari long-term memory."""
        entry = self._long_term.get(key)
        if entry is None:
            return default
        return entry["value"]

    def recall_entry(self, key: str) -> Optional[dict]:
        """Ambil entry lengkap (termasuk source, timestamp, confirmed)."""
        return self._long_term.get(key)

    def search(self, keyword: str) -> dict:
        """Cari semua key yang mengandung keyword."""
        keyword = keyword.lower()
        return {
            k: v for k, v in self._long_term.items()
            if keyword in k.lower() or (
                isinstance(v.get("value"), str) and keyword in v["value"].lower()
            )
        }

    def all_confirmed(self) -> dict:
        """Kembalikan hanya fakta yang sudah dikonfirmasi Rofi."""
        return {
            k: v for k, v in self._long_term.items()
            if v.get("confirmed", False)
        }

    def long_term_count(self) -> int:
        return len(self._long_term)

    def long_term_version(self) -> int:
        """
        Versi counter long-term memory.
        Naik setiap kali ada remember() atau forget().
        Brain pakai ini untuk cache invalidation system prompt.
        """
        return self._long_term_version

    def summary_for_llm(self, max_items: int = 15) -> str:
        """
        Buat ringkasan long-term memory untuk dimasukkan ke system prompt LLM.
        Prioritaskan fakta yang sudah dikonfirmasi, terbaru duluan.
        """
        if not self._long_term:
            return ""

        sorted_entries = sorted(
            self._long_term.items(),
            key=lambda x: (
                -int(x[1].get("confirmed", False)),
                -x[1].get("updated_at", 0)
            )
        )[:max_items]

        lines = ["[Yang Otto tahu]"]
        for key, entry in sorted_entries:
            val = entry["value"]
            src = entry["source"]
            lines.append(f"- {key}: {val} ({src})")

        return "\n".join(lines)

    # ─── PERSIST ──────────────────────────────────────────────────────────────

    def _load_long_term(self) -> None:
        if self.memory_path.exists():
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._long_term = data
            except (json.JSONDecodeError, OSError):
                self._long_term = {}
        else:
            self._long_term = {}

    def _save_long_term(self) -> None:
        if len(self._long_term) > MEMORY["long_term_limit"]:
            sorted_keys = sorted(
                self._long_term.items(),
                key=lambda x: (
                    int(x[1].get("confirmed", False)),
                    x[1].get("updated_at", 0)
                )
            )
            to_remove = [k for k, _ in sorted_keys[:len(self._long_term) - MEMORY["long_term_limit"]]]
            for k in to_remove:
                del self._long_term[k]

        try:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(self._long_term, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"[memory] Gagal simpan: {e}")

    # ─── TEMP ─────────────────────────────────────────────────────────────────

    def get_temp(self, key: str) -> str | None:
        return self._temp.get(key)

    def set_temp(self, key: str, value: str) -> None:
        self._temp[key] = value

    def delete_temp(self, key: str) -> None:
        self._temp.pop(key, None)

    # ─── DEBUG ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"<MemoryManager "
            f"short={self.short_term_count()}/{MEMORY['short_term_limit']} "
            f"long={self.long_term_count()}/{MEMORY['long_term_limit']}>"
        )


# Singleton
memory = MemoryManager()
