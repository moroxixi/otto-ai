# core/memory.py
# Sistem ingatan Otto — dua lapis: pendek (RAM) & panjang (disk)
#
# Short-term : percakapan terakhir, dibawa ke konteks LLM tiap request
# Long-term  : fakta penting yang sudah diverifikasi, disimpan ke JSON

import json
import time
from pathlib import Path
from typing import Optional
from collections import deque

from core.config import PATHS, MEMORY


class MemoryManager:
    """
    Kelola dua lapis ingatan Otto:
      - short_term : deque of {role, content, timestamp}
      - long_term  : dict tersimpan di disk, key = topik/label
    """

    def __init__(self):
        self.memory_path: Path = PATHS["memory"]
        self._short_term: deque = deque(maxlen=MEMORY["short_term_limit"])
        self._long_term: dict = {}
        self._load_long_term()
        self._temp: dict[str, str] = {}

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

    def clear_short_term(self) -> None:
        """Reset percakapan — misal saat sesi baru dimulai."""
        self._short_term.clear()

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
        self._save_long_term()

    def recall(self, key: str, default=None):
        """
        Ambil nilai dari long-term memory.
        Return default jika key tidak ada.
        """
        entry = self._long_term.get(key)
        if entry is None:
            return default
        return entry["value"]

    def recall_entry(self, key: str) -> Optional[dict]:
        """Ambil entry lengkap (termasuk source, timestamp, confirmed)."""
        return self._long_term.get(key)

    def forget(self, key: str) -> bool:
        """
        Hapus satu fakta dari long-term memory.
        Return True jika berhasil, False jika key tidak ditemukan.
        """
        if key in self._long_term:
            del self._long_term[key]
            self._save_long_term()
            return True
        return False

    def search(self, keyword: str) -> dict:
        """
        Cari semua key yang mengandung keyword.
        Berguna saat brain.py perlu cari konteks relevan.

        Contoh: memory.search("rofi") → semua yang diketahui tentang Rofi
        """
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

    def summary_for_llm(self, max_items: int = 15) -> str:
        """
        Buat ringkasan long-term memory untuk dimasukkan ke system prompt LLM.
        Prioritaskan fakta yang sudah dikonfirmasi, terbaru duluan.

        Contoh output:
          [Yang Otto tahu tentang Rofi]
          - rofi.minuman.favorit: kopi oat (konfirmasi_rofi)
          - rofi.kebiasaan.pagi: aktif jam 7 (observasi)
        """
        if not self._long_term:
            return ""

        # Urutkan: confirmed dulu, lalu terbaru
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
        """Load dari disk saat startup."""
        if self.memory_path.exists():
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Validasi: pastikan format benar
                    if isinstance(data, dict):
                        self._long_term = data
            except (json.JSONDecodeError, OSError):
                # File rusak → mulai kosong, tidak crash
                self._long_term = {}
        else:
            self._long_term = {}

    def _save_long_term(self) -> None:
        """Simpan ke disk. Dipanggil tiap kali ada perubahan."""
        # Trim jika melebihi batas
        if len(self._long_term) > MEMORY["long_term_limit"]:
            # Buang yang paling lama & tidak confirmed
            sorted_keys = sorted(
                self._long_term.items(),
                key=lambda x: (
                    int(x[1].get("confirmed", False)),
                    x[1].get("updated_at", 0)
                )
            )
            keys_to_remove = [k for k, _ in sorted_keys[:len(self._long_term) - MEMORY["long_term_limit"]]]
            for k in keys_to_remove:
                del self._long_term[k]

        try:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(self._long_term, f, ensure_ascii=False, indent=2)
        except OSError as e:
            # Log tapi tidak crash — Otto tetap berjalan walau gagal simpan
            print(f"[memory] Gagal simpan: {e}")









    def get_temp(self, key: str) -> str | None:
        """Ambil nilai sementara (tidak persisten ke disk)."""
        return self._temp.get(key)
    
    def set_temp(self, key: str, value: str) -> None:
        """Simpan nilai sementara di memory (hilang saat restart)."""
        self._temp[key] = value
    
    def delete_temp(self, key: str) -> None:
        """Hapus nilai sementara."""
        self._temp.pop(key, None)

    # ─── DEBUG ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"<MemoryManager "
            f"short={self.short_term_count()}/{MEMORY['short_term_limit']} "
            f"long={self.long_term_count()}/{MEMORY['long_term_limit']}>"
        )


# Singleton — satu instance dipakai di seluruh Otto
memory = MemoryManager()
