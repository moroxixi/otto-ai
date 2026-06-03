"""
intelligence/profiler.py — Mesin Hipotesis Otto
=================================================
Profiler membaca data mentah dari activity_watcher dan
membentuk HIPOTESIS tentang Rofi — belum fakta, belum dikonfirmasi.

Filosofi:
  Lapisan 2→3 — jembatan antara OBSERVASI dan PROAKTIF
  "Aku lihat pola. Aku punya dugaan. Belum tentu benar."

Alur kerja:
  activity_watcher.get_summary()
        ↓
    profiler.analyze()         ← kamu di sini
        ↓
  Hypothesis[] disimpan ke hypotheses.json
        ↓
    curiosity.py mengambil hipotesis matang → tanya Rofi

Status hipotesis:
  "pending"    → baru dibentuk, belum dikonfirmasi
  "confirmed"  → Rofi sudah jawab "iya"
  "rejected"   → Rofi sudah jawab "tidak"
  "stale"      → data sudah lama, perlu diperbarui

Cara integrasi (dari app.py atau scheduler.py):
    from intelligence.profiler import Profiler
    profiler = Profiler(watcher)
    await profiler.analyze()
    hypotheses = profiler.get_pending()
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from uuid import uuid4

logger = logging.getLogger("otto.intelligence.profiler")

# ─────────────────────────── Konfigurasi ────────────────────────────────────

HYPOTHESES_FILE  = Path("/data/asd/otto-ai/data/hypotheses.json")
PROFILE_FILE     = Path("/data/asd/otto-ai/data/profile.json")

# Minimum data sebelum mulai buat hipotesis
MIN_INTERACTIONS = 5

# Seberapa kuat sinyal sebelum dianggap pola
KEYWORD_THRESHOLD = 3    # kata harus muncul ≥3x
HOUR_THRESHOLD    = 3    # jam harus aktif ≥3x

# Hipotesis "stale" setelah berapa hari tidak diverifikasi
STALE_DAYS = 7


# ─────────────────────────── Model Hipotesis ─────────────────────────────────

@dataclass
class Hypothesis:
    """Satu hipotesis tentang Rofi."""
    id:          str   = field(default_factory=lambda: uuid4().hex[:8])
    category:    str   = ""       # "habit", "preference", "schedule", "topic"
    claim:       str   = ""       # kalimat hipotesis: "Rofi suka ngobrol malam"
    evidence:    str   = ""       # alasan singkat: "aktif 21x antara jam 20-23"
    confidence:  float = 0.0      # 0.0–1.0 (berapa kuat sinyal)
    status:      str   = "pending"  # pending | confirmed | rejected | stale
    created_at:  str   = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at:  str   = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    asked_count: int   = 0        # berapa kali curiosity sudah tanya ini

    def confirm(self) -> None:
        self.status     = "confirmed"
        self.updated_at = datetime.now().isoformat(timespec="seconds")

    def reject(self) -> None:
        self.status     = "rejected"
        self.updated_at = datetime.now().isoformat(timespec="seconds")

    def mark_stale(self) -> None:
        self.status     = "stale"
        self.updated_at = datetime.now().isoformat(timespec="seconds")

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Hypothesis":
        return Hypothesis(**{k: v for k, v in d.items() if k in Hypothesis.__dataclass_fields__})


# ─────────────────────────── Profiler ────────────────────────────────────────

class Profiler:
    """
    Membaca summary dari ActivityWatcher dan membentuk hipotesis.

    Penggunaan:
        profiler = Profiler(watcher)
        await profiler.analyze()
        pending = profiler.get_pending()  # → list[Hypothesis]
    """

    def __init__(self, watcher) -> None:
        self._watcher     = watcher
        self._hypotheses: list[Hypothesis] = []
        self._load()
        logger.info(
            "[profiler] Siap. %d hipotesis dimuat (%d pending).",
            len(self._hypotheses),
            len(self.get_pending()),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def analyze(self) -> list[Hypothesis]:
        """
        Jalankan analisis. Bentuk hipotesis baru dari data watcher.
        Duplikat klaim yang sudah ada diabaikan.
        Return: list hipotesis BARU yang dibentuk sesi ini.
        """
        summary = self._watcher.get_summary()
        total   = summary.get("total", 0)

        if total < MIN_INTERACTIONS:
            logger.info(
                "[profiler] Data belum cukup (%d/%d interaksi).",
                total, MIN_INTERACTIONS,
            )
            return []

        new_hypotheses: list[Hypothesis] = []

        new_hypotheses += self._analyze_schedule(summary)
        new_hypotheses += self._analyze_topics(summary)
        new_hypotheses += self._analyze_activity_level(summary)

        # Tandai hipotesis lama yang sudah stale
        self._mark_stale_hypotheses()

        # Simpan yang baru (filter duplikat claim)
        existing_claims = {h.claim for h in self._hypotheses}
        added = []
        for h in new_hypotheses:
            if h.claim not in existing_claims:
                self._hypotheses.append(h)
                existing_claims.add(h.claim)
                added.append(h)
                logger.info(
                    "[profiler] +hipotesis [%s] %.0f%% — %s",
                    h.category, h.confidence * 100, h.claim,
                )

        if added:
            self._save()

        return added

    def get_pending(self) -> list[Hypothesis]:
        """Hipotesis yang belum ditanyakan ke Rofi, diurutkan confidence turun."""
        pending = [h for h in self._hypotheses if h.status == "pending"]
        return sorted(pending, key=lambda h: h.confidence, reverse=True)

    def get_confirmed(self) -> list[Hypothesis]:
        """Fakta yang sudah dikonfirmasi Rofi."""
        return [h for h in self._hypotheses if h.status == "confirmed"]

    def get_all(self) -> list[Hypothesis]:
        return list(self._hypotheses)

    def confirm(self, hypothesis_id: str) -> bool:
        """Tandai hipotesis sebagai benar. Return True jika ditemukan."""
        h = self._find(hypothesis_id)
        if h:
            h.confirm()
            self._save()
            logger.info("[profiler] Hipotesis %s dikonfirmasi: %s", h.id, h.claim)
            return True
        return False

    def reject(self, hypothesis_id: str) -> bool:
        """Tandai hipotesis sebagai salah. Return True jika ditemukan."""
        h = self._find(hypothesis_id)
        if h:
            h.reject()
            self._save()
            logger.info("[profiler] Hipotesis %s ditolak: %s", h.id, h.claim)
            return True
        return False

    def increment_asked(self, hypothesis_id: str) -> None:
        """Catat bahwa curiosity sudah tanya hipotesis ini."""
        h = self._find(hypothesis_id)
        if h:
            h.asked_count += 1
            h.updated_at  = datetime.now().isoformat(timespec="seconds")
            self._save()

    def build_profile_summary(self) -> str:
        """
        Buat ringkasan profil Rofi dari hipotesis yang sudah dikonfirmasi.
        Untuk dimasukkan ke system prompt LLM.
        """
        confirmed = self.get_confirmed()
        if not confirmed:
            return ""

        lines = ["Yang Otto ketahui tentang Rofi (sudah dikonfirmasi):"]
        for h in confirmed:
            lines.append(f"- {h.claim}")
        return "\n".join(lines)

    # ── Analyzer: Jadwal / Waktu ──────────────────────────────────────────────

    def _analyze_schedule(self, summary: dict) -> list[Hypothesis]:
        """Buat hipotesis tentang kapan Rofi aktif."""
        results = []
        active_hours = summary.get("active_hours", [])  # [(hour, count), ...]
        total        = summary.get("total", 1)

        SESSIONS = {
            "pagi":  (range(5, 10),  "pagi hari (jam 5–10)"),
            "siang": (range(10, 14), "siang hari (jam 10–14)"),
            "sore":  (range(14, 18), "sore hari (jam 14–18)"),
            "malam": (range(18, 23), "malam hari (jam 18–23)"),
            "larut": (range(23, 24), "larut malam"),
        }

        for session, (hour_range, label) in SESSIONS.items():
            count = sum(c for h, c in active_hours if h in hour_range)
            if count >= HOUR_THRESHOLD:
                conf = min(count / max(total * 0.3, 1), 0.9)
                claim = f"Rofi sering aktif di {label}"
                results.append(Hypothesis(
                    category   = "schedule",
                    claim      = claim,
                    evidence   = f"aktif {count}x di jam {label}",
                    confidence = round(conf, 2),
                ))

        return results

    # ── Analyzer: Topik / Minat ───────────────────────────────────────────────

    def _analyze_topics(self, summary: dict) -> list[Hypothesis]:
        """Buat hipotesis tentang topik yang sering dibicarakan Rofi."""
        results  = []
        keywords = dict(summary.get("top_keywords", []))
        total    = summary.get("total", 1)

        # Kelompok topik sederhana
        TOPIC_GROUPS = {
            "musik":    {"musik", "lagu", "putar", "play", "santai", "jadul"},
            "kerja":    {"kerja", "kerjaan", "meeting", "proposal", "deadline", "kantor", "rapat", "project"},
            "kesehatan":{"obat", "berat", "badan", "tensi", "darah", "tidur", "sakit", "vitamin"},
            "kopi":     {"kopi", "coffee", "espresso", "oat", "susu"},
            "teknologi":{"laptop", "komputer", "coding", "code", "program", "aplikasi"},
        }

        for topic, vocab in TOPIC_GROUPS.items():
            count = sum(keywords.get(w, 0) for w in vocab)
            if count >= KEYWORD_THRESHOLD:
                conf  = min(count / (total * 0.3), 0.9)
                claim = f"Rofi sering membicarakan topik {topic}"
                results.append(Hypothesis(
                    category   = "topic",
                    claim      = claim,
                    evidence   = f"kata terkait '{topic}' muncul {count}x",
                    confidence = round(conf, 2),
                ))

        return results

    
    # ── Analyzer: Tingkat Aktivitas ───────────────────────────────────────────

    def _analyze_activity_level(self, summary: dict) -> list[Hypothesis]:
        """Buat hipotesis dari total dan pola aktivitas."""
        results = []
        total   = summary.get("total", 0)
        days    = summary.get("active_days", [])

        if total < 10:
            return results

        # Berapa hari berbeda Rofi aktif?
        active_day_count = sum(1 for _, c in days if c > 0)

        if active_day_count >= 5:
            results.append(Hypothesis(
                category   = "habit",
                claim      = "Rofi menggunakan Otto hampir setiap hari",
                evidence   = f"aktif {active_day_count} hari berbeda",
                confidence = 0.75,
            ))
        elif active_day_count >= 3:
            results.append(Hypothesis(
                category   = "habit",
                claim      = "Rofi menggunakan Otto beberapa kali seminggu",
                evidence   = f"aktif {active_day_count} hari berbeda",
                confidence = 0.6,
            ))

        return results

    # ── Stale Check ───────────────────────────────────────────────────────────

    def _mark_stale_hypotheses(self) -> None:
        """Tandai hipotesis pending lama sebagai stale."""
        cutoff = datetime.now() - timedelta(days=STALE_DAYS)
        for h in self._hypotheses:
            if h.status != "pending":
                continue
            try:
                created = datetime.fromisoformat(h.created_at)
                if created < cutoff:
                    h.mark_stale()
                    logger.info("[profiler] Hipotesis %s jadi stale: %s", h.id, h.claim)
            except ValueError:
                pass

    # ── Utils ─────────────────────────────────────────────────────────────────

    def _find(self, hypothesis_id: str) -> Optional[Hypothesis]:
        for h in self._hypotheses:
            if h.id == hypothesis_id:
                return h
        return None

    # ── Persistensi ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        HYPOTHESES_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            HYPOTHESES_FILE.write_text(
                json.dumps(
                    [h.to_dict() for h in self._hypotheses],
                    ensure_ascii=False,
                    indent=2,
                )
            )
        except OSError as e:
            logger.error("[profiler] Gagal simpan hypotheses.json: %s", e)

    def _load(self) -> None:
        if not HYPOTHESES_FILE.exists():
            return
        try:
            data = json.loads(HYPOTHESES_FILE.read_text(encoding="utf-8"))
            self._hypotheses = [Hypothesis.from_dict(d) for d in data]
        except Exception as e:
            logger.warning("[profiler] Gagal load hypotheses.json: %s", e)


# ─────────────────────────── Singleton Helper ────────────────────────────────

_profiler_instance: Optional[Profiler] = None

def get_profiler() -> Profiler:
    if _profiler_instance is None:
        raise RuntimeError("Profiler belum diinisialisasi. Panggil init_profiler(watcher) dulu.")
    return _profiler_instance

def init_profiler(watcher) -> Profiler:
    global _profiler_instance
    _profiler_instance = Profiler(watcher)
    return _profiler_instance


def inject_hypothesis(self, fact: dict) -> Optional[Hypothesis]:
    """
    Terima fakta eksternal dari consolidator → buat hipotesis baru.
    fact = {"key": "rofi.preferensi.minuman", "value": "kopi oat", "confidence": 0.8}
    """
    key        = fact.get("key", "")
    value      = fact.get("value", "")
    confidence = fact.get("confidence", 0.6)

    if not key or not value:
        return None

    # Derive category dari key: rofi.<category>.<sub>
    parts    = key.split(".")
    category = parts[1] if len(parts) >= 2 else "preference"

    claim = f"Rofi {value}"

    # Cek duplikat
    existing_claims = {h.claim for h in self._hypotheses}
    if claim in existing_claims:
        return None

    h = Hypothesis(
        category   = category,
        claim      = claim,
        evidence   = f"dari konsolidasi LLM (key={key})",
        confidence = round(confidence, 2),
        status     = "pending",
    )
    self._hypotheses.append(h)
    self._save()
    logger.info("[profiler] inject_hypothesis: %s (%.0f%%)", claim, confidence * 100)
    return h


# ─────────────────────────── Quick Test ─────────────────────────────────────

if __name__ == "__main__":
    import asyncio, logging
    logging.basicConfig(level=logging.DEBUG)

    # Mock watcher sederhana
    class MockWatcher:
        def get_summary(self):
            return {
                "total": 30,
                "top_keywords": [
                    ("kopi", 8), ("meeting", 5), ("musik", 4),
                    ("lagu", 4), ("obat", 3), ("kerja", 6),
                ],
                "top_skills": [
                    ("reminder", 6), ("play_santai", 4), ("track_weight", 3),
                ],
                "active_hours": [(21, 12), (22, 8), (20, 5), (8, 4)],
                "active_days":  [("Senin", 5), ("Selasa", 4), ("Rabu", 4),
                                  ("Kamis", 3), ("Jumat", 3)],
                "patterns": {"peak_hour": 21},
            }

    async def _test():
        # Override path
        global HYPOTHESES_FILE, PROFILE_FILE
        HYPOTHESES_FILE = Path("/tmp/otto_hypotheses.json")
        PROFILE_FILE    = Path("/tmp/otto_profile.json")

        profiler = Profiler(MockWatcher())
        new = await profiler.analyze()

        print(f"\n=== {len(new)} HIPOTESIS BARU ===")
        for h in new:
            print(f"  [{h.category}] {h.confidence:.0%} — {h.claim}")
            print(f"         bukti: {h.evidence}")

        print(f"\n=== PENDING ({len(profiler.get_pending())}) ===")
        for h in profiler.get_pending():
            print(f"  {h.id} | {h.claim}")

        # Simulasi konfirmasi
        if profiler.get_pending():
            first = profiler.get_pending()[0]
            profiler.confirm(first.id)
            print(f"\nDikonfirmasi: {first.claim}")
            print(f"Profile: {profiler.build_profile_summary()}")

    asyncio.run(_test())
