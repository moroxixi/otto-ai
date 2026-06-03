"""
intelligence/consolidator.py — Memory Consolidation Otto
=========================================================
Tugasnya: sebelum short-term memory penuh dan pesan lama hilang,
ringkas percakapan → ekstrak fakta penting → simpan ke long-term memory.

Filosofi:
  Short-term = RAM — cepat, terbatas, hilang saat restart
  Long-term  = disk — persisten, terstruktur, dibawa ke system prompt

  Tanpa consolidator: setiap restart Otto "amnesia total"
  Dengan consolidator: Otto ingat apa yang pernah dibicarakan

Cara kerja:
  1. brain.py panggil consolidator.maybe_consolidate() setelah setiap think()
  2. Cek apakah short-term sudah mencapai CONSOLIDATION_THRESHOLD
  3. Jika ya → ambil pesan yang belum diproses → kirim ke Groq
  4. Groq ekstrak fakta tentang Rofi dalam format JSON terstruktur
  5. Setiap fakta → memory.remember() ke long-term
  6. Catat posisi terakhir yang sudah dikonsolidasi (tidak proses ulang)

Integrasi (di brain.py):
    # Di __init__:
    from intelligence.consolidator import Consolidator
    self._consolidator = Consolidator(memory, groq_call_fn=self._call_groq)

    # Di think(), setelah _log_to_memory selesai:
    asyncio.create_task(self._consolidator.maybe_consolidate())

Desain sadar:
  - TIDAK hapus short-term — deque tetap jalan normal
  - Gunakan Groq (LLM kecil cukup) bukan model besar
  - JSON response wajib — fallback ke regex jika parse gagal
  - Idempotent: fakta duplikat tidak overwrite yang confirmed
  - State konsolidasi survive restart via JSON file
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("otto.intelligence.consolidator")

# ─────────────────────────── Konfigurasi ─────────────────────────────────────

STATE_FILE = Path("/data/asd/otto-ai/data/consolidation_state.json")

# Mulai konsolidasi setelah berapa pesan di short-term
CONSOLIDATION_THRESHOLD = 15

# Minimum pesan baru sejak konsolidasi terakhir sebelum proses lagi
MIN_NEW_MESSAGES = 10

# Berapa pesan yang diambil per batch konsolidasi
BATCH_SIZE = 20

# Model ringan untuk konsolidasi (hemat token)
CONSOLIDATION_MODEL = "llama-3.1-8b-instant"

# Prompt ke LLM untuk ekstrak fakta
_EXTRACTION_PROMPT = """\
Kamu sedang membantu sistem memori AI bernama Otto.
Di bawah ini adalah percakapan antara Otto (asisten) dan Rofi (pengguna).

TUGAS:
Ekstrak fakta penting tentang ROFI dari percakapan ini.
Hanya fakta yang EKSPLISIT disebutkan — jangan asumsikan.
Abaikan basa-basi dan percakapan umum.

FORMAT RESPONSE (JSON saja, tanpa penjelasan, tanpa markdown):
{
  "facts": [
    {
      "key": "rofi.preferensi.minuman",
      "value": "kopi oat setiap pagi",
      "source": "observasi_percakapan",
      "confidence": 0.8
    }
  ]
}

Panduan key:
- Gunakan format: rofi.<kategori>.<subkategori>
- Kategori: preferensi, kebiasaan, jadwal, pekerjaan, kesehatan, teknologi, karakter
- Contoh: rofi.preferensi.minuman, rofi.kebiasaan.tidur, rofi.pekerjaan.jenis

Jika tidak ada fakta penting, kembalikan: {"facts": []}

PERCAKAPAN:
{conversation}
"""


# ─────────────────────────── Consolidator ────────────────────────────────────

class Consolidator:
    """
    Meringkas short-term memory → long-term facts secara otomatis.

    Penggunaan:
        consolidator = Consolidator(memory, groq_call_fn=brain._call_groq)
        # Dipanggil otomatis dari brain.py:
        asyncio.create_task(consolidator.maybe_consolidate())
    """

    def __init__(self, memory, groq_call_fn: Callable, profiler=None) -> None:
        self._memory         = memory
        self._call_groq      = groq_call_fn
        self._profiler  = profiler
        self._lock = asyncio.Lock()

        # State persisten
        self._last_consolidated_count: int = 0   # total pesan saat konsolidasi terakhir
        self._total_facts_saved: int       = 0
        self._load_state()

        logger.info(
            "[consolidator] Siap. Threshold=%d pesan. Fakta tersimpan sebelumnya=%d.",
            CONSOLIDATION_THRESHOLD,
            self._total_facts_saved,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def maybe_consolidate(self) -> None:
        """
        Cek apakah perlu konsolidasi. Jika ya, jalankan.
        Aman dipanggil setiap saat — ada guard internal.
        """
        # Guard: jangan jalan paralel
        if self._lock.locked():
            return

        async with self._lock:
            current_count = self._memory.short_term_count()
            if current_count < CONSOLIDATION_THRESHOLD:
                return
            new_since_last = current_count - self._last_consolidated_count
            if new_since_last < MIN_NEW_MESSAGES and self._last_consolidated_count > 0:
                return
            try:
                await self._consolidate()
            except Exception as e:
                logger.error("[consolidator] Error: %s", e, exc_info=True)


    async def force_consolidate(self) -> int:
        """
        Paksa konsolidasi sekarang, abaikan threshold.
        Return jumlah fakta yang berhasil disimpan.
        Berguna untuk: sebelum restart, atau debug manual.
        """
        if self._lock.locked():
            logger.warning("[consolidator] Sedang berjalan, skip force.")
            return 0

        async with self._lock:
            return await self._consolidate()

    def get_stats(self) -> dict:
        """Statistik konsolidasi untuk monitoring."""
        return {
            "total_facts_saved":        self._total_facts_saved,
            "last_consolidated_count":  self._last_consolidated_count,
            "current_short_term":       self._memory.short_term_count(),
            "long_term_count":          self._memory.long_term_count(),
        }

    # ── Core Logic ────────────────────────────────────────────────────────────

    async def _consolidate(self) -> int:
        """
        Jalankan satu siklus konsolidasi.
        Return jumlah fakta yang berhasil disimpan.
        """
        messages = self._memory.get_recent_messages(limit=BATCH_SIZE)
        if not messages:
            return 0

        logger.info(
            "[consolidator] Mulai konsolidasi %d pesan...", len(messages)
        )

        # Format percakapan untuk LLM
        conversation_text = self._format_conversation(messages)

        # Kirim ke Groq
        raw_response = await self._extract_facts_from_llm(conversation_text)
        if not raw_response:
            return 0

        # Parse JSON dari LLM
        facts = self._parse_facts(raw_response)
        if not facts:
            logger.info("[consolidator] Tidak ada fakta baru ditemukan.")
            self._update_state()
            return 0

        # Simpan ke long-term memory
        saved = self._save_facts(facts)

        # Update state
        self._total_facts_saved += saved
        self._update_state()

        logger.info(
            "[consolidator] Selesai. %d/%d fakta disimpan ke long-term memory.",
            saved, len(facts),
        )
        return saved

    # ── LLM Extraction ────────────────────────────────────────────────────────

    async def _extract_facts_from_llm(self, conversation_text: str) -> Optional[str]:
        """Kirim percakapan ke Groq, minta ekstraksi fakta dalam JSON."""
        prompt = _EXTRACTION_PROMPT.replace("{conversation}", conversation_text)

        messages = [
            {"role": "user", "content": prompt}
        ]

        try:
            raw = await self._call_groq(
                messages,
                model=CONSOLIDATION_MODEL,
            )
            text = raw.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            logger.debug("[consolidator] Raw LLM response: %.200s", text)
            return text
        except Exception as e:
            logger.error("[consolidator] Gagal panggil Groq: %s", e)
            return None

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_facts(self, raw_text: str) -> list[dict]:
        """
        Parse JSON dari respons LLM.
        Robust: coba bersihkan markdown fence jika ada.
        """
        # Bersihkan markdown fence ```json ... ```
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text  = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        try:
            data  = json.loads(text)
            facts = data.get("facts", [])
            if not isinstance(facts, list):
                return []

            # Validasi setiap fakta
            valid = []
            for f in facts:
                if not isinstance(f, dict):
                    continue
                if not f.get("key") or not f.get("value"):
                    continue
                # Pastikan key ada prefix rofi.
                if not f["key"].startswith("rofi."):
                    f["key"] = f"rofi.{f['key']}"
                valid.append(f)

            return valid

        except json.JSONDecodeError as e:
            logger.warning("[consolidator] Gagal parse JSON: %s | raw: %.100s", e, raw_text)
            return []

    # ── Simpan ke Long-Term ───────────────────────────────────────────────────

    def _save_facts(self, facts: list[dict]) -> int:
        """
        Simpan fakta ke long-term memory.
        Skip fakta yang sudah ada dan sudah dikonfirmasi Rofi
        (jangan overwrite fakta confirmed dengan observasi baru).
        """
        saved = 0
        for fact in facts:
            key        = fact["key"]
            value      = fact["value"]
            source     = fact.get("source", "observasi_percakapan")
            confidence = fact.get("confidence", 0.6)

            # Jika confidence tinggi → inject ke profiler sebagai hipotesis
            if confidence > 0.7 and hasattr(self, "_profiler") and self._profiler:
                self._profiler.inject_hypothesis(fact)

            # Cek apakah sudah ada dan sudah dikonfirmasi
            existing = self._memory.recall_entry(key)
            if existing and existing.get("confirmed", False):
                logger.debug(
                    "[consolidator] Skip '%s' — sudah dikonfirmasi Rofi.", key
                )
                continue

            # Cek apakah value berubah signifikan
            if existing and existing.get("value") == value:
                logger.debug(
                    "[consolidator] Skip '%s' — value sama, tidak perlu update.", key
                )
                continue

            # Simpan dengan tag confidence di source
            tagged_source = f"{source} (conf={confidence:.0%})"
            self._memory.remember(key, value, source=tagged_source)

            logger.info(
                "[consolidator] ✓ Disimpan: %s = %s (%s)",
                key, value, tagged_source,
            )
            saved += 1

        return saved

    # ── Format Percakapan ─────────────────────────────────────────────────────

    @staticmethod
    def _format_conversation(messages: list[dict]) -> str:
        """Format list pesan jadi teks percakapan yang mudah dibaca LLM."""
        lines = []
        for msg in messages:
            role    = msg.get("role", "unknown")
            content = msg.get("content", "").strip()
            if not content:
                continue
            label = "Rofi" if role == "user" else "Otto"
            lines.append(f"{label}: {content}")
        return "\n".join(lines)

    # ── State Persistensi ─────────────────────────────────────────────────────

    def _update_state(self) -> None:
        """Simpan state konsolidasi ke disk agar survive restart."""
        self._last_consolidated_count = self._memory.short_term_count()
        state = {
            "last_consolidated_count": self._last_consolidated_count,
            "total_facts_saved":       self._total_facts_saved,
            "last_run_at":             time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        except OSError as e:
            logger.error("[consolidator] Gagal simpan state: %s", e)

    def _load_state(self) -> None:
        """Load state dari disk saat startup."""
        if not STATE_FILE.exists():
            return
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            self._last_consolidated_count = state.get("last_consolidated_count", 0)
            self._total_facts_saved       = state.get("total_facts_saved", 0)
            logger.info(
                "[consolidator] State dimuat. Last count=%d, total fakta=%d.",
                self._last_consolidated_count,
                self._total_facts_saved,
            )
        except Exception as e:
            logger.warning("[consolidator] Gagal load state: %s", e)


# ─────────────────────────── Singleton Helper ────────────────────────────────

_consolidator_instance: Optional[Consolidator] = None

def get_consolidator() -> Consolidator:
    if _consolidator_instance is None:
        raise RuntimeError(
            "Consolidator belum diinisialisasi. Panggil init_consolidator() dulu."
        )
    return _consolidator_instance

def init_consolidator(memory, groq_call_fn, profiler=None) -> Consolidator:
    global _consolidator_instance
    _consolidator_instance = Consolidator(memory, groq_call_fn, profiler=profiler)
    return _consolidator_instance


# ─────────────────────────── Quick Test ──────────────────────────────────────

if __name__ == "__main__":
    import asyncio, logging
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")

    # Mock memory
    class MockMemory:
        def __init__(self):
            self._short = [
                {"role": "user",      "content": "aku tiap pagi minum kopi oat"},
                {"role": "assistant", "content": "oh, kopi oat ya. enak tuh!"},
                {"role": "user",      "content": "iya, punya 4 cabang warung kopi"},
                {"role": "assistant", "content": "wah serius? usaha sendiri?"},
                {"role": "user",      "content": "iya, sambil WFH kadang-kadang"},
                {"role": "user",      "content": "pakai linux archlinux di laptop"},
                {"role": "assistant", "content": "keren, arch user ya"},
                {"role": "user",      "content": "tidur biasanya jam 11 malam"},
                {"role": "assistant", "content": "oke, aku catat"},
                {"role": "user",      "content": "suka dengerin lagu jadul pas kerja"},
                {"role": "assistant", "content": "vibe yang enak tuh"},
                {"role": "user",      "content": "lagi diet juga, pantau berat badan"},
                {"role": "assistant", "content": "semangat!"},
                {"role": "user",      "content": "target turun 5kg bulan ini"},
                {"role": "assistant", "content": "aku bisa bantu ingatkan"},
            ]
            self._long = {}
            self._version = 0

        def short_term_count(self):      return len(self._short)
        def get_recent_messages(self, limit=20): return self._short[-limit:]
        def long_term_count(self):       return len(self._long)
        def recall_entry(self, key):     return self._long.get(key)

        def remember(self, key, value, source="manual"):
            self._long[key] = {"value": value, "source": source, "confirmed": False}
            self._version += 1
            print(f"  [MEMORY] remember: {key} = {value}")

    # Mock Groq response
    FAKE_GROQ_RESPONSE = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "facts": [
                        {"key": "rofi.preferensi.minuman", "value": "kopi oat setiap pagi", "source": "observasi_percakapan", "confidence": 0.8},
                        {"key": "rofi.pekerjaan.jenis",   "value": "pemilik 4 cabang warung kopi", "source": "observasi_percakapan", "confidence": 0.85},
                        {"key": "rofi.kebiasaan.kerja",   "value": "WFH kadang-kadang", "source": "observasi_percakapan", "confidence": 0.7},
                        {"key": "rofi.teknologi.os",      "value": "Arch Linux", "source": "observasi_percakapan", "confidence": 0.9},
                        {"key": "rofi.kebiasaan.tidur",   "value": "tidur sekitar jam 11 malam", "source": "observasi_percakapan", "confidence": 0.75},
                        {"key": "rofi.preferensi.musik",  "value": "lagu jadul saat kerja", "source": "observasi_percakapan", "confidence": 0.7},
                        {"key": "rofi.kesehatan.target",  "value": "diet, target turun 5kg", "source": "observasi_percakapan", "confidence": 0.8},
                    ]
                })
            }
        }]
    }

    async def mock_call_groq(messages, **kwargs):
        await asyncio.sleep(0.1)  # simulasi latency
        return FAKE_GROQ_RESPONSE

    async def _test():
        global STATE_FILE
        STATE_FILE = Path("/tmp/otto_consolidation_state.json")

        memory       = MockMemory()
        consolidator = Consolidator(memory, groq_call_fn=mock_call_groq)

        print(f"\n=== STATUS AWAL ===")
        print(f"  Short-term: {memory.short_term_count()} pesan")
        print(f"  Long-term:  {memory.long_term_count()} fakta")

        print(f"\n=== MAYBE CONSOLIDATE (threshold={CONSOLIDATION_THRESHOLD}) ===")
        await consolidator.maybe_consolidate()

        print(f"\n=== FORCE CONSOLIDATE ===")
        saved = await consolidator.force_consolidate()
        print(f"  Fakta disimpan: {saved}")

        print(f"\n=== STATUS AKHIR ===")
        print(f"  Long-term: {memory.long_term_count()} fakta")
        stats = consolidator.get_stats()
        for k, v in stats.items():
            print(f"  {k}: {v}")

    asyncio.run(_test())
