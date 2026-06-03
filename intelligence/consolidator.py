"""
intelligence/consolidator.py — Memory Consolidation Otto
=========================================================
Tugasnya: sebelum short-term memory penuh dan pesan lama hilang,
ringkas percakapan → ekstrak fakta penting → simpan ke long-term memory
→ inferensi relasi antar fakta → simpan implikasi perilaku.

Filosofi:
  Short-term = RAM — cepat, terbatas, hilang saat restart
  Long-term  = disk — persisten, terstruktur, dibawa ke system prompt
  Relations  = kesimpulan dari hubungan antar fakta → implikasi nyata

  Tanpa consolidator: Otto amnesia total
  Dengan consolidator: Otto ingat fakta
  Dengan relational inference: Otto MENGERTI Rofi, bukan hanya mencatat

Cara kerja:
  1. brain.py panggil consolidator.maybe_consolidate() setelah setiap think()
  2. Cek apakah short-term sudah mencapai CONSOLIDATION_THRESHOLD
  3. Jika ya → ambil pesan yang belum diproses → kirim ke Groq
  4. Groq ekstrak fakta tentang Rofi dalam format JSON terstruktur
  5. Setiap fakta → memory.remember() ke long-term
  6. [BARU] Groq inferensi relasi antar fakta baru + fakta lama
  7. [BARU] Setiap relasi → memory.add_relation() ke relational layer
  8. Catat posisi terakhir yang sudah dikonsolidasi (tidak proses ulang)
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

CONSOLIDATION_THRESHOLD = 15
MIN_NEW_MESSAGES        = 10
BATCH_SIZE              = 20
CONSOLIDATION_MODEL     = "llama-3.1-8b-instant"

# ─────────────────────────── Prompt: Ekstraksi Fakta ─────────────────────────

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

# ─────────────────────────── Prompt: Inferensi Relasi ────────────────────────

_RELATION_PROMPT = """\
Kamu membantu sistem memori AI bernama Otto yang belajar mengenal Rofi.

FAKTA BARU yang baru saja dikonfirmasi tentang Rofi:
{new_facts}

FAKTA LAMA yang sudah tersimpan tentang Rofi:
{existing_facts}

TUGAS:
Dari kombinasi fakta-fakta di atas, temukan HUBUNGAN yang bermakna.
Hubungan yang berguna = hubungan yang menghasilkan implikasi perilaku konkret untuk Otto.

Contoh hubungan yang berguna:
- "Rofi tidur jam 23 + Rofi aktif nanya hal teknis malam hari" 
  → implikasi: "Jangan kirim reminder penting sebelum jam 10 pagi"
- "Rofi punya 4 cabang warung kopi + Rofi sering bahas operasional"
  → implikasi: "Ketika Rofi stress, tanya dulu soal warung atau hal lain?"

Contoh hubungan yang TIDAK berguna (jangan masukkan):
- "Rofi suka kopi + Rofi minum kopi pagi" → terlalu trivial
- "Rofi pakai Linux" → fakta tunggal, bukan relasi

FORMAT RESPONSE (JSON saja, tanpa penjelasan, tanpa markdown):
{
  "relations": [
    {
      "id": "tidur_larut_produktif_malam",
      "from_facts": ["rofi.kebiasaan.tidur", "rofi.produktif.waktu"],
      "description": "Rofi tidur larut, kemungkinan produktif di malam hari",
      "implication": "Kirim reminder penting antara jam 20-22, bukan pagi hari",
      "confidence": 0.75
    }
  ]
}

Aturan:
- id: snake_case, singkat, deskriptif
- from_facts: gunakan key yang persis sama dari daftar fakta
- description: kalimat pendek menjelaskan hubungan (bukan hanya restating fakta)
- implication: tindakan KONKRET yang Otto harus ambil
- confidence: 0.6–0.9 (jangan 1.0 kecuali sangat jelas)
- Maksimal 3 relasi per panggilan
- Jika tidak ada relasi bermakna: {"relations": []}
"""


# ─────────────────────────── Consolidator ────────────────────────────────────

class Consolidator:
    """
    Meringkas short-term memory → long-term facts → relational inference.

    Penggunaan:
        consolidator = Consolidator(memory, groq_call_fn=brain._call_groq)
        asyncio.create_task(consolidator.maybe_consolidate())
    """

    def __init__(self, memory, groq_call_fn: Callable, profiler=None) -> None:
        self._memory    = memory
        self._call_groq = groq_call_fn
        self._profiler  = profiler
        self._lock      = asyncio.Lock()

        self._last_consolidated_count: int = 0
        self._total_facts_saved: int       = 0
        self._total_relations_inferred: int = 0
        self._load_state()

        logger.info(
            "[consolidator] Siap. Threshold=%d pesan. "
            "Fakta=%d, Relasi=%d.",
            CONSOLIDATION_THRESHOLD,
            self._total_facts_saved,
            self._total_relations_inferred,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def maybe_consolidate(self) -> None:
        """Cek dan jalankan konsolidasi jika perlu. Aman dipanggil kapan saja."""
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
        """Paksa konsolidasi sekarang, abaikan threshold."""
        if self._lock.locked():
            logger.warning("[consolidator] Sedang berjalan, skip force.")
            return 0

        async with self._lock:
            return await self._consolidate()

    def get_stats(self) -> dict:
        return {
            "total_facts_saved":          self._total_facts_saved,
            "total_relations_inferred":   self._total_relations_inferred,
            "last_consolidated_count":    self._last_consolidated_count,
            "current_short_term":         self._memory.short_term_count(),
            "long_term_count":            self._memory.long_term_count(),
            "relations_count":            self._memory.relations_count(),
        }

    # ── Core Logic ────────────────────────────────────────────────────────────

    async def _consolidate(self) -> int:
        """Satu siklus konsolidasi lengkap: ekstrak fakta → inferensi relasi."""
        messages = self._memory.get_recent_messages(limit=BATCH_SIZE)
        if not messages:
            return 0

        logger.info("[consolidator] Mulai konsolidasi %d pesan...", len(messages))

        # 1. Ekstrak fakta dari percakapan
        conversation_text = self._format_conversation(messages)
        raw_response      = await self._extract_facts_from_llm(conversation_text)
        if not raw_response:
            return 0

        facts = self._parse_facts(raw_response)
        if not facts:
            logger.info("[consolidator] Tidak ada fakta baru.")
            self._update_state()
            return 0

        # 2. Simpan fakta ke long-term
        saved_facts = self._save_facts(facts)

        # 3. Inferensi relasi dari fakta baru + fakta lama
        #    Jalankan hanya jika ada fakta baru yang berhasil disimpan
        saved_relations = 0
        if saved_facts > 0:
            saved_relations = await self._infer_relations(facts)

        # 4. Update state
        self._total_facts_saved       += saved_facts
        self._total_relations_inferred += saved_relations
        self._update_state()

        logger.info(
            "[consolidator] Selesai. Fakta=%d, Relasi=%d.",
            saved_facts, saved_relations,
        )
        return saved_facts

    # ── LLM: Ekstraksi Fakta ──────────────────────────────────────────────────

    async def _extract_facts_from_llm(self, conversation_text: str) -> Optional[str]:
        prompt = _EXTRACTION_PROMPT.replace("{conversation}", conversation_text)
        try:
            raw = await self._call_groq(
                [{"role": "user", "content": prompt}],
                model=CONSOLIDATION_MODEL,
            )
            text = raw.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            logger.debug("[consolidator] Fact extraction response: %.200s", text)
            return text
        except Exception as e:
            logger.error("[consolidator] Gagal ekstrak fakta: %s", e)
            return None

    # ── LLM: Inferensi Relasi ─────────────────────────────────────────────────

    async def _infer_relations(self, new_facts: list[dict]) -> int:
        """
        Minta LLM temukan hubungan bermakna antara fakta baru dan fakta lama.
        Kembalikan jumlah relasi yang berhasil disimpan.
        """
        # Format fakta baru
        new_facts_text = "\n".join(
            f"- {f['key']}: {f['value']}"
            for f in new_facts
            if f.get("key") and f.get("value")
        )
        if not new_facts_text:
            return 0

        # Ambil fakta lama yang relevan (maks 10 untuk hemat token)
        existing_items = sorted(
            self._memory._long_term.items(),
            key=lambda x: -x[1].get("updated_at", 0)
        )[:10]
        existing_facts_text = "\n".join(
            f"- {k}: {v['value']}"
            for k, v in existing_items
        )
        if not existing_facts_text:
            existing_facts_text = "(belum ada fakta lama)"

        prompt = (
            _RELATION_PROMPT
            .replace("{new_facts}", new_facts_text)
            .replace("{existing_facts}", existing_facts_text)
        )

        try:
            raw = await self._call_groq(
                [{"role": "user", "content": prompt}],
                model=CONSOLIDATION_MODEL,
            )
            text = raw.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            logger.debug("[consolidator] Relation inference response: %.200s", text)
        except Exception as e:
            logger.error("[consolidator] Gagal inferensi relasi: %s", e)
            return 0

        # Parse dan simpan relasi
        relations = self._parse_relations(text)
        if not relations:
            logger.info("[consolidator] Tidak ada relasi baru ditemukan.")
            return 0

        saved = 0
        for r in relations:
            self._memory.add_relation(
                relation_id   = r["id"],
                from_facts    = r.get("from_facts", []),
                description   = r["description"],
                implication   = r["implication"],
                confidence    = r.get("confidence", 0.65),
                relation_type = "inferred",
            )
            saved += 1
            logger.info(
                "[consolidator] ✓ Relasi: '%s' (impl: %s)",
                r["id"], r["implication"][:60]
            )

        return saved

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_facts(self, raw_text: str) -> list[dict]:
        """Parse JSON fakta dari respons LLM."""
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text  = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        try:
            data  = json.loads(text)
            facts = data.get("facts", [])
            if not isinstance(facts, list):
                return []

            valid = []
            for f in facts:
                if not isinstance(f, dict):
                    continue
                if not f.get("key") or not f.get("value"):
                    continue
                if not f["key"].startswith("rofi."):
                    f["key"] = f"rofi.{f['key']}"
                valid.append(f)
            return valid

        except json.JSONDecodeError as e:
            logger.warning("[consolidator] Gagal parse fakta JSON: %s", e)
            return []

    def _parse_relations(self, raw_text: str) -> list[dict]:
        """Parse JSON relasi dari respons LLM."""
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text  = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        try:
            data      = json.loads(text)
            relations = data.get("relations", [])
            if not isinstance(relations, list):
                return []

            valid = []
            for r in relations:
                if not isinstance(r, dict):
                    continue
                # Field wajib
                if not r.get("id") or not r.get("description") or not r.get("implication"):
                    continue
                # Sanitasi id: snake_case tanpa spasi
                r["id"] = r["id"].replace(" ", "_").replace("-", "_").lower()
                # Confidence dalam range yang wajar
                conf = float(r.get("confidence", 0.65))
                r["confidence"] = max(0.5, min(0.95, conf))
                valid.append(r)
            return valid

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("[consolidator] Gagal parse relasi JSON: %s", e)
            return []

    # ── Simpan Fakta ──────────────────────────────────────────────────────────

    def _save_facts(self, facts: list[dict]) -> int:
        """Simpan fakta ke long-term memory."""
        saved = 0
        for fact in facts:
            key        = fact["key"]
            value      = fact["value"]
            source     = fact.get("source", "observasi_percakapan")
            confidence = fact.get("confidence", 0.6)

            # Inject ke profiler sebagai hipotesis jika confidence tinggi
            if confidence > 0.7 and hasattr(self, "_profiler") and self._profiler:
                self._profiler.inject_hypothesis(fact)

            # Skip jika sudah dikonfirmasi Rofi
            existing = self._memory.recall_entry(key)
            if existing and existing.get("confirmed", False):
                logger.debug("[consolidator] Skip '%s' — sudah dikonfirmasi.", key)
                continue

            # Skip jika value sama
            if existing and existing.get("value") == value:
                logger.debug("[consolidator] Skip '%s' — value sama.", key)
                continue

            tagged_source = f"{source} (conf={confidence:.0%})"
            self._memory.remember(key, value, source=tagged_source)
            logger.info("[consolidator] ✓ Fakta: %s = %s", key, value)
            saved += 1

        return saved

    # ── Format Percakapan ─────────────────────────────────────────────────────

    @staticmethod
    def _format_conversation(messages: list[dict]) -> str:
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
        self._last_consolidated_count = self._memory.short_term_count()
        state = {
            "last_consolidated_count":  self._last_consolidated_count,
            "total_facts_saved":        self._total_facts_saved,
            "total_relations_inferred": self._total_relations_inferred,
            "last_run_at":              time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        except OSError as e:
            logger.error("[consolidator] Gagal simpan state: %s", e)

    def _load_state(self) -> None:
        if not STATE_FILE.exists():
            return
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            self._last_consolidated_count  = state.get("last_consolidated_count", 0)
            self._total_facts_saved        = state.get("total_facts_saved", 0)
            self._total_relations_inferred = state.get("total_relations_inferred", 0)
            logger.info(
                "[consolidator] State dimuat. last=%d, fakta=%d, relasi=%d.",
                self._last_consolidated_count,
                self._total_facts_saved,
                self._total_relations_inferred,
            )
        except Exception as e:
            logger.warning("[consolidator] Gagal load state: %s", e)


# ─────────────────────────── Singleton Helper ────────────────────────────────

_consolidator_instance: Optional[Consolidator] = None

def get_consolidator() -> Consolidator:
    if _consolidator_instance is None:
        raise RuntimeError("Consolidator belum diinisialisasi.")
    return _consolidator_instance

def init_consolidator(memory, groq_call_fn, profiler=None) -> Consolidator:
    global _consolidator_instance
    _consolidator_instance = Consolidator(memory, groq_call_fn, profiler=profiler)
    return _consolidator_instance


# ─────────────────────────── Quick Test ──────────────────────────────────────

if __name__ == "__main__":
    import asyncio, logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    class MockMemory:
        def __init__(self):
            self._short = [
                {"role": "user",      "content": "aku tiap pagi minum kopi oat"},
                {"role": "assistant", "content": "oh, kopi oat ya. enak tuh!"},
                {"role": "user",      "content": "iya, punya 4 cabang warung kopi"},
                {"role": "assistant", "content": "wah serius? usaha sendiri?"},
                {"role": "user",      "content": "iya, sambil WFH kadang-kadang"},
                {"role": "user",      "content": "tidur biasanya jam 11 malam"},
                {"role": "assistant", "content": "oke, aku catat"},
                {"role": "user",      "content": "paling produktif setelah maghrib"},
                {"role": "assistant", "content": "noted, malam hari ya"},
                {"role": "user",      "content": "target turun 5kg bulan ini"},
                {"role": "assistant", "content": "aku bisa bantu ingatkan"},
                {"role": "user",      "content": "biasanya olahraga pagi kalau sempat"},
                {"role": "assistant", "content": "wah keren, disiplin"},
                {"role": "user",      "content": "tapi susah bangun pagi"},
                {"role": "assistant", "content": "haha, aku maklum"},
                {"role": "user",      "content": "makanya kalau olahraga sering skip"},
            ]
            self._long = {
                "rofi.kebiasaan.tidur": {
                    "value": "tidur sekitar jam 11 malam",
                    "source": "observasi_percakapan",
                    "confirmed": False,
                    "updated_at": time.time() - 3600,
                }
            }
            self._relations = {}
            self._version   = 0

        def short_term_count(self):              return len(self._short)
        def get_recent_messages(self, limit=20): return self._short[-limit:]
        def long_term_count(self):               return len(self._long)
        def relations_count(self):               return len(self._relations)
        def recall_entry(self, key):             return self._long.get(key)

        def remember(self, key, value, source="manual"):
            self._long[key] = {"value": value, "source": source, "confirmed": False, "updated_at": time.time()}
            print(f"  [FAKTA] {key} = {value}")

        def add_relation(self, relation_id, from_facts, description, implication, confidence, relation_type):
            self._relations[relation_id] = {
                "id": relation_id, "from_facts": from_facts,
                "description": description, "implication": implication,
                "confidence": confidence, "relation_type": relation_type,
            }
            print(f"  [RELASI] {relation_id}: {implication}")

    FAKE_FACTS_RESPONSE = json.dumps({
        "facts": [
            {"key": "rofi.preferensi.minuman", "value": "kopi oat setiap pagi", "source": "observasi_percakapan", "confidence": 0.8},
            {"key": "rofi.produktif.waktu",    "value": "setelah maghrib / malam hari", "source": "observasi_percakapan", "confidence": 0.85},
            {"key": "rofi.kesehatan.target",   "value": "diet, target turun 5kg", "source": "observasi_percakapan", "confidence": 0.8},
            {"key": "rofi.kebiasaan.olahraga", "value": "rencana olahraga pagi tapi sering skip karena susah bangun", "source": "observasi_percakapan", "confidence": 0.75},
        ]
    })

    FAKE_RELATION_RESPONSE = json.dumps({
        "relations": [
            {
                "id": "tidur_larut_produktif_malam",
                "from_facts": ["rofi.kebiasaan.tidur", "rofi.produktif.waktu"],
                "description": "Rofi tidur larut dan paling produktif malam hari — pola night owl",
                "implication": "Kirim reminder penting antara jam 20-22, hindari notifikasi sebelum jam 10 pagi",
                "confidence": 0.8,
            },
            {
                "id": "olahraga_pagi_konflik_tidur_larut",
                "from_facts": ["rofi.kebiasaan.olahraga", "rofi.kebiasaan.tidur"],
                "description": "Rencana olahraga pagi Rofi sering gagal karena tidur larut — ada konflik jadwal",
                "implication": "Jika Rofi minta diingatkan olahraga, sarankan sesi malam atau siang bukan pagi",
                "confidence": 0.75,
            },
        ]
    })

    call_count = 0
    async def mock_call_groq(messages, **kwargs):
        global call_count
        await asyncio.sleep(0.05)
        call_count += 1
        # Panggilan pertama = ekstrak fakta, kedua = inferensi relasi
        return {"choices": [{"message": {"content": FAKE_FACTS_RESPONSE if call_count == 1 else FAKE_RELATION_RESPONSE}}]}

    async def _test():
        global STATE_FILE
        STATE_FILE = Path("/tmp/otto_consolidation_state.json")

        mem = MockMemory()
        c   = Consolidator(mem, groq_call_fn=mock_call_groq)

        print("\n=== FORCE CONSOLIDATE ===")
        saved = await c.force_consolidate()

        print(f"\n=== HASIL ===")
        print(f"  Fakta disimpan : {mem.long_term_count()}")
        print(f"  Relasi dibuat  : {mem.relations_count()}")
        stats = c.get_stats()
        for k, v in stats.items():
            print(f"  {k}: {v}")

    asyncio.run(_test())
