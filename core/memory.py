# core/memory.py
# Sistem ingatan Otto — dua lapis: pendek (RAM) & panjang (disk)
# + lapisan ketiga: relational memory (hubungan antar fakta)
#
# Short-term  : percakapan terakhir, dibawa ke konteks LLM tiap request
# Long-term   : fakta penting yang sudah diverifikasi, disimpan ke JSON
# Relations   : inferensi hubungan antar fakta → implikasi perilaku Otto

import json
import time
import logging
from pathlib import Path
from typing import Optional
from collections import deque

from core.config import PATHS, MEMORY

logger = logging.getLogger("otto.core.memory")

SHORT_TERM_PERSIST_PATH = PATHS["short_term_cache"]

# Path relasi — di sebelah memory.json
# Contoh: /data/asd/otto-ai/data/memory_relations.json
_RELATIONS_PATH = Path("/data/asd/otto-ai/data/memory_relations.json")


class MemoryManager:
    """
    Kelola tiga lapis ingatan Otto:
      - short_term  : deque of {role, content, timestamp}
      - long_term   : dict tersimpan di disk, key = topik/label
      - relations   : graph sederhana hubungan antar fakta + implikasinya
    """

    def __init__(self):
        self.memory_path: Path = PATHS["memory"]
        self._short_term: deque = deque(maxlen=MEMORY["short_term_limit"])
        self._load_short_term()
        self._long_term: dict = {}
        self._load_long_term()
        self._temp: dict[str, str] = {}

        # ── Relational memory ──────────────────────────────────────────────
        # Struktur setiap entry:
        # {
        #   "id": "tidur_larut→produktif_malam",
        #   "from_facts": ["rofi.kebiasaan.tidur", "rofi.produktif.waktu"],
        #   "relation_type": "inferred",      # "inferred" | "confirmed" | "rejected"
        #   "description": "Rofi tidur larut → kemungkinan produktif malam",
        #   "implication": "Jangan kirim reminder penting sebelum jam 10 pagi",
        #   "confidence": 0.7,
        #   "created_at": 1234567890.0,
        #   "confirmed_at": null,
        # }
        self._relations: dict[str, dict] = {}
        self._load_relations()

        # Cache fingerprint — untuk deteksi perubahan long-term
        self._long_term_version: int = 0

    # ─── SHORT TERM ───────────────────────────────────────────────────────────

    def add_message(self, role: str, content: str) -> None:
        self._short_term.append({
            "role":      role,
            "content":   content,
            "timestamp": time.time(),
        })

    def get_short_term(self) -> list[dict]:
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self._short_term
        ]

    def get_recent_messages(self, limit: int = 20) -> list[dict]:
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in self._short_term
        ]
        return messages[-limit:]

    def clear_short_term(self) -> None:
        self._short_term.clear()

    def persist_short_term(self, max_messages: int = 20) -> None:
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
            logger.info("[memory] Short-term cache disimpan: %d pesan.", len(messages))
        except OSError as e:
            logger.error("[memory] Gagal simpan short-term cache: %s", e)

    def _load_short_term(self) -> None:
        if not SHORT_TERM_PERSIST_PATH.exists():
            return
        try:
            messages = json.loads(SHORT_TERM_PERSIST_PATH.read_text(encoding="utf-8"))
            if not isinstance(messages, list):
                return
            for msg in messages:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
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

    def remember(self, key: str, value, source: str = "manual") -> bool:
        """
        Simpan fakta ke long-term memory.
 
        Return:
            True  — fakta berhasil disimpan
            False — fakta DIBLOK karena mencoba overwrite confirmed fact
                    dengan value berbeda (proteksi dari LLM hallucination)
 
        Kasus yang ditangani:
          1. Fakta baru, belum ada → simpan normal
          2. Fakta ada, belum confirmed → update value + source
          3. Fakta ada, confirmed, value SAMA → update timestamp, pertahankan confirmed
          4. Fakta ada, confirmed, value BEDA → BLOK, log warning, return False
          5. source == "konfirmasi_rofi" → selalu simpan, set confirmed=True
             (konfirmasi eksplisit dari Rofi selalu menang)
        """
        existing = self._long_term.get(key)
 
        # Kasus 5: konfirmasi eksplisit dari Rofi — selalu izinkan
        if source == "konfirmasi_rofi":
            self._long_term[key] = {
                "value":      value,
                "source":     source,
                "updated_at": time.time(),
                "confirmed":  True,
            }
            self._long_term_version += 1
            self._save_long_term()
            return True
 
        # Kasus 4: fakta sudah confirmed, value berbeda → BLOK
        if existing and existing.get("confirmed", False):
            if existing.get("value") != value:
                logger.warning(
                    "[memory] BLOCKED overwrite fakta confirmed '%s': "
                    "tersimpan='%s' vs baru='%s' (source=%s). "
                    "Gunakan source='konfirmasi_rofi' untuk override eksplisit.",
                    key, existing["value"], value, source,
                )
                return False
 
        # Kasus 3: fakta ada, confirmed, value SAMA → pertahankan confirmed
        is_confirmed = existing.get("confirmed", False) if existing else False
 
        self._long_term[key] = {
            "value":      value,
            "source":     source,
            "updated_at": time.time(),
            "confirmed":  is_confirmed,
        }
        self._long_term_version += 1
        self._save_long_term()
        return True

    def forget(self, key: str) -> bool:
        if key in self._long_term:
            del self._long_term[key]
            # Hapus juga relasi yang melibatkan key ini
            self._remove_relations_for_key(key)
            self._long_term_version += 1
            self._save_long_term()
            return True
        return False

    def recall(self, key: str, default=None):
        entry = self._long_term.get(key)
        if entry is None:
            return default
        return entry["value"]

    def recall_entry(self, key: str) -> Optional[dict]:
        return self._long_term.get(key)

    def search(self, keyword: str) -> dict:
        keyword = keyword.lower()
        return {
            k: v for k, v in self._long_term.items()
            if keyword in k.lower() or (
                isinstance(v.get("value"), str) and keyword in v["value"].lower()
            )
        }

    def all_confirmed(self) -> dict:
        return {
            k: v for k, v in self._long_term.items()
            if v.get("confirmed", False)
        }

    def long_term_count(self) -> int:
        return len(self._long_term)

    def long_term_version(self) -> int:
        return self._long_term_version

    def summary_for_llm(self, max_items: int = 15) -> str:
        """
        Buat ringkasan long-term memory untuk system prompt LLM.
        Sekarang juga menyertakan implikasi dari relasi yang ada.
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

        lines = ["[Yang Otto tahu tentang Rofi]"]
        for key, entry in sorted_entries:
            val = entry["value"]
            src = entry["source"]
            lines.append(f"- {key}: {val} ({src})")

        # Tambahkan implikasi dari relasi yang ter-confirm atau confidence tinggi
        implications = self.get_implications(min_confidence=0.65)
        if implications:
            lines.append("\n[Kesimpulan perilaku Otto]")
            for imp in implications[:5]:  # maks 5 implikasi di system prompt
                lines.append(f"- {imp}")

        return "\n".join(lines)

    # ─── RELATIONAL MEMORY ────────────────────────────────────────────────────

    def add_relation(
        self,
        relation_id: str,
        from_facts: list[str],
        description: str,
        implication: str,
        confidence: float = 0.6,
        relation_type: str = "inferred",
    ) -> None:
        """
        Simpan hubungan antar fakta ke relational memory.

        Args:
            relation_id  : ID unik, contoh "tidur_larut→produktif_malam"
            from_facts   : list key fakta yang membentuk relasi ini
            description  : kalimat pendek menjelaskan hubungannya
            implication  : aksi konkret yang Otto ambil dari relasi ini
            confidence   : 0.0–1.0
            relation_type: "inferred" | "confirmed" | "rejected"

        Contoh:
            memory.add_relation(
                relation_id  = "tidur_larut→reminder_malam",
                from_facts   = ["rofi.kebiasaan.tidur", "rofi.produktif.waktu"],
                description  = "Rofi tidur larut, kemungkinan aktif malam hari",
                implication  = "Kirim reminder penting antara jam 20-22, bukan pagi",
                confidence   = 0.75,
            )
        """
        # Jangan overwrite relasi yang sudah dikonfirmasi dengan inferred baru
        existing = self._relations.get(relation_id)
        if existing and existing.get("relation_type") == "confirmed":
            logger.debug(
                "[memory] Skip add_relation '%s' — sudah dikonfirmasi.", relation_id
            )
            return

        self._relations[relation_id] = {
            "id":            relation_id,
            "from_facts":    from_facts,
            "relation_type": relation_type,
            "description":   description,
            "implication":   implication,
            "confidence":    confidence,
            "created_at":    time.time(),
            "confirmed_at":  None,
        }
        self._save_relations()
        logger.info(
            "[memory] Relasi baru: '%s' (conf=%.0f%%)", relation_id, confidence * 100
        )

    def confirm_relation(self, relation_id: str) -> bool:
        """
        Rofi mengkonfirmasi relasi ini benar.
        Naikkan confidence ke 1.0 dan tandai confirmed.
        """
        if relation_id not in self._relations:
            return False
        self._relations[relation_id]["relation_type"] = "confirmed"
        self._relations[relation_id]["confidence"]    = 1.0
        self._relations[relation_id]["confirmed_at"]  = time.time()
        self._save_relations()
        logger.info("[memory] Relasi dikonfirmasi: '%s'", relation_id)
        return True

    def reject_relation(self, relation_id: str) -> bool:
        """
        Rofi menolak relasi ini (salah inferensi).
        Tandai rejected — tidak dihapus, tapi tidak dipakai untuk implikasi.
        """
        if relation_id not in self._relations:
            return False
        self._relations[relation_id]["relation_type"] = "rejected"
        self._relations[relation_id]["confidence"]    = 0.0
        self._save_relations()
        logger.info("[memory] Relasi ditolak: '%s'", relation_id)
        return True

    def get_relations(self, fact_key: str = None) -> list[dict]:
        """
        Ambil semua relasi, atau filter yang melibatkan fact_key tertentu.
        Hanya kembalikan yang belum rejected.
        """
        relations = [
            r for r in self._relations.values()
            if r.get("relation_type") != "rejected"
        ]
        if fact_key:
            relations = [
                r for r in relations
                if fact_key in r.get("from_facts", [])
            ]
        return sorted(relations, key=lambda r: -r.get("confidence", 0))

    def get_implications(self, min_confidence: float = 0.6) -> list[str]:
        """
        Kembalikan list implikasi perilaku dari semua relasi aktif.
        Digunakan untuk system prompt Otto.

        Contoh output:
            ["Kirim reminder penting antara jam 20-22, bukan pagi",
             "Rofi aktif di pesan singkat — jangan respons panjang"]
        """
        implications = []
        for r in self._relations.values():
            if r.get("relation_type") == "rejected":
                continue
            if r.get("confidence", 0) >= min_confidence:
                impl = r.get("implication", "").strip()
                if impl:
                    implications.append(impl)
        return implications

    def get_relation(self, relation_id: str) -> Optional[dict]:
        """Ambil satu relasi berdasarkan ID."""
        return self._relations.get(relation_id)

    def relations_summary_for_llm(self) -> str:
        """
        Format ringkas semua relasi aktif untuk dimasukkan ke system prompt.
        Lebih detail dari get_implications() — cocok untuk brain.py.
        """
        active = [
            r for r in self._relations.values()
            if r.get("relation_type") != "rejected"
            and r.get("confidence", 0) >= 0.6
        ]
        if not active:
            return ""

        lines = ["[Pola yang Otto pelajari tentang Rofi]"]
        for r in sorted(active, key=lambda x: -x.get("confidence", 0))[:8]:
            status = "✓" if r["relation_type"] == "confirmed" else "~"
            lines.append(
                f"{status} {r['description']} → {r['implication']} "
                f"(conf={r['confidence']:.0%})"
            )
        return "\n".join(lines)

    def relations_count(self) -> int:
        return len(self._relations)

    def _remove_relations_for_key(self, fact_key: str) -> None:
        """Hapus semua relasi yang melibatkan fact_key (dipanggil saat forget())."""
        to_delete = [
            rid for rid, r in self._relations.items()
            if fact_key in r.get("from_facts", [])
        ]
        for rid in to_delete:
            del self._relations[rid]
        if to_delete:
            self._save_relations()
            logger.debug(
                "[memory] %d relasi dihapus karena '%s' di-forget.", len(to_delete), fact_key
            )

    # ─── PERSIST (Relations) ──────────────────────────────────────────────────

    def _load_relations(self) -> None:
        if not _RELATIONS_PATH.exists():
            return
        try:
            data = json.loads(_RELATIONS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._relations = data
            logger.info(
                "[memory] Relasi dimuat: %d entri.", len(self._relations)
            )
        except Exception as e:
            logger.warning("[memory] Gagal load relations: %s", e)

    def _save_relations(self) -> None:
        try:
            _RELATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
            _RELATIONS_PATH.write_text(
                json.dumps(self._relations, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except OSError as e:
            logger.error("[memory] Gagal simpan relations: %s", e)

    # ─── PERSIST (Long-term) ──────────────────────────────────────────────────

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
            to_remove = [
                k for k, _ in sorted_keys[:len(self._long_term) - MEMORY["long_term_limit"]]
            ]
            for k in to_remove:
                del self._long_term[k]

        try:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(self._long_term, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.error("[memory] Gagal simpan: %s", e)

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
            f"long={self.long_term_count()}/{MEMORY['long_term_limit']} "
            f"relations={self.relations_count()}>"
        )


# Singleton
memory = MemoryManager()
