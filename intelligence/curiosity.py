"""
intelligence/curiosity.py — Sistem Tanya Otto
==============================================
Curiosity mengambil hipotesis matang dari Profiler dan memutuskan:
  1. KAPAN waktu yang tepat untuk bertanya ke Rofi
  2. BAGAIMANA cara bertanya (natural, tidak terasa seperti survei)
  3. BERAPA KALI boleh tanya sebelum dianggap mengganggu

Filosofi:
  Otto tidak tanya semua hipotesis sekaligus.
  Otto pilih SATU hipotesis per sesi → tanya dengan cara yang terasa natural
  → tunggu jawaban Rofi → simpan hasilnya ke Profiler.

  Manusia yang baru kenal tidak langsung wawancara.
  Dia selipkan satu pertanyaan di tengah obrolan.

Alur kerja:
  profiler.get_pending()
        ↓
  curiosity.pick_hypothesis()    ← pilih 1 yang paling siap ditanya
        ↓
  curiosity.generate_question()  ← buat kalimat tanya yang natural
        ↓
  Dikirim ke Rofi (via brain.py / app.py)
        ↓
  Rofi jawab → parse_response() → profiler.confirm() / profiler.reject()

Cara integrasi (dari app.py):
    from intelligence.curiosity import Curiosity
    curiosity = Curiosity(profiler)
    question = await curiosity.try_ask()   # None jika belum waktunya
    if question:
        # kirim question ke Rofi lewat TTS / chat
        ...
    # Setelah Rofi jawab:
    result = await curiosity.handle_response(hypothesis_id, rofi_text)
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("otto.intelligence.curiosity")

# ──────────────────────────── Konfigurasi ────────────────────────────────────

STATE_FILE = Path("/data/asd/otto-ai/data/curiosity_state.json")

# Jeda minimum antar pertanyaan (dalam menit) — agar tidak terasa menginterogasi
MIN_GAP_MINUTES = 90

# Maksimal berapa kali satu hipotesis boleh ditanyakan sebelum diabaikan
MAX_ASKED_COUNT = 2

# Waktu "aman" untuk bertanya (jam lokal) — Otto tidak tanya pas Rofi sibuk
SAFE_HOURS = list(range(8, 12)) + list(range(14, 17)) + list(range(19, 22))


# ──────────────────────────── Template Pertanyaan ────────────────────────────

# Template per kategori — dipilih acak agar tidak monoton
QUESTION_TEMPLATES: dict[str, list[str]] = {
    "schedule": [
        "Rofi, aku lihat kamu sering aktif {evidence_hint}. Emang itu waktu favoritmu ya?",
        "Eh Rofi, kayaknya kamu lebih sering ngobrol sama aku {evidence_hint}. Bener nggak?",
        "{evidence_hint} kamu emang biasanya lagi santai, atau lagi sibuk juga?",
    ],
    "topic": [
        "Rofi, aku perhatiin kamu lumayan sering bahas soal {topic_hint}. Emang suka ya?",
        "Boleh aku tanya — {topic_hint} itu kebutuhan sehari-hari kamu atau cuma kadang-kadang?",
        "Sepertinya {topic_hint} cukup penting buat kamu. Ada yang bisa aku bantu lebih di situ?",
    ],
    "habit": [
        "Aku penasaran — {claim_short}. Itu kebiasaan kamu ya?",
        "Rofi, {claim_short}? Aku perhatiin dari beberapa waktu ini.",
        "Boleh aku tanya sesuatu? {claim_short} — bener nggak sih?",
    ],
    "preference": [
        "Rofi, {claim_short} — aku tangkap itu dari obrolannya. Tepat nggak?",
        "Aku mau mastiin — {claim_short}. Itu bener kan?",
        "Kayaknya {claim_short}. Aku boleh simpen itu sebagai catatan tentang kamu?",
    ],
}

# Fallback jika kategori tidak dikenal
DEFAULT_TEMPLATES = [
    "Rofi, boleh aku tanya satu hal? {claim_short} — bener nggak sih?",
    "Aku penasaran soal sesuatu. {claim_short}. Itu tepat nggak?",
]



def _get_min_confidence() -> float:
    """
    Hitung MIN_CONFIDENCE dari personality boldness.
    boldness 0.0 → min_confidence 0.2 (berani tanya meski belum yakin)
    boldness 1.0 → min_confidence 0.7 (hanya tanya kalau sangat yakin)
    """
    from core.config import INTELLIGENCE
    boldness = INTELLIGENCE.get("curiosity_boldness", 0.5)
    boldness = max(0.0, min(1.0, boldness))  # clamp 0–1
    return 0.2 + (boldness * 0.5)  # range: 0.2–0.7

# ──────────────────────────── Curiosity ──────────────────────────────────────

class Curiosity:
    """
    Memilih hipotesis yang tepat dan menghasilkan pertanyaan natural untuk Rofi.

    Penggunaan:
        curiosity = Curiosity(profiler)
        question, hyp_id = await curiosity.try_ask()
        if question:
            # kirim ke Rofi
            await brain.speak(question)
        # Setelah Rofi jawab:
        await curiosity.handle_response(hyp_id, "iya bener banget")
    """

    def __init__(self, profiler) -> None:
        self._profiler = profiler
        self._last_asked_at: Optional[datetime] = None
        self._pending_hypothesis_id: Optional[str] = None
        self._load_state()
        logger.info(
            "[curiosity] Siap. Last ask: %s",
            self._last_asked_at or "belum pernah",
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def try_ask(self) -> tuple[Optional[str], Optional[str]]:
        """
        Coba hasilkan pertanyaan untuk Rofi.

        Return: (question_text, hypothesis_id)
          - Keduanya None jika belum waktunya / tidak ada hipotesis layak.
          - Jika ada pertanyaan: langsung kirim ke Rofi.
        """
        if not self._is_good_time():
            return None, None

        hypothesis = self._pick_hypothesis()
        if hypothesis is None:
            return None, None

        question = self._generate_question(hypothesis)
        if not question:
            return None, None

        # Update state
        self._last_asked_at = datetime.now()
        self._pending_hypothesis_id = hypothesis.id
        self._profiler.increment_asked(hypothesis.id)
        self._save_state()

        logger.info(
            "[curiosity] Tanya hipotesis %s: %s | Q: %s",
            hypothesis.id, hypothesis.claim, question,
        )
        
        from intelligence.growth_tracker import get_tracker
        try:
            get_tracker().record_event("proactive_question", {"question": question[:60]})
        except Exception:
            pass
        return question, hypothesis.id
     
        
    async def handle_response(self, hypothesis_id: str, rofi_text: str) -> str:
       verdict = self._parse_response(rofi_text)

       # Update profiler dulu (WAJIB — ini yang ubah status hipotesis)
       if verdict == "confirmed":
           self._profiler.confirm(hypothesis_id)
           logger.info("[curiosity] Hipotesis %s → CONFIRMED", hypothesis_id)
       elif verdict == "rejected":
           self._profiler.reject(hypothesis_id)
           logger.info("[curiosity] Hipotesis %s → REJECTED", hypothesis_id)

       # Record ke growth tracker
       from intelligence.growth_tracker import get_tracker
       try:
           if verdict == "confirmed":
               get_tracker().record_event("trust_response")
           elif verdict == "rejected":
               get_tracker().record_event("correction_accepted")
       except Exception:
           pass

       # Reset pending
       self._pending_hypothesis_id = None
       self._save_state()
       return verdict




        

            
    def get_pending_hypothesis_id(self) -> Optional[str]:
        """Hipotesis yang sedang menunggu jawaban Rofi."""
        return self._pending_hypothesis_id

    def reset_pending(self) -> None:
        """Reset jika Rofi tidak menjawab (timeout / ganti topik)."""
        if self._pending_hypothesis_id:
            logger.info(
                "[curiosity] Reset pending hipotesis %s (tidak dijawab).",
                self._pending_hypothesis_id,
            )
        self._pending_hypothesis_id = None
        self._save_state()

    # ── Pemilihan Hipotesis ───────────────────────────────────────────────────

    def _pick_hypothesis(self):
        """
        Pilih SATU hipotesis yang paling layak ditanyakan.

        Prioritas:
          1. Confidence tinggi
          2. Belum pernah ditanya (asked_count == 0)
          3. Tidak melewati batas tanya
        """
        candidates = [
            h for h in self._profiler.get_pending()
            if h.confidence >= _get_min_confidence()
            and h.asked_count < MAX_ASKED_COUNT
        ]

        if not candidates:
            return None

        # Prioritaskan yang belum pernah ditanya, lalu urut confidence
        untouched = [h for h in candidates if h.asked_count == 0]
        pool = untouched if untouched else candidates

        # Ambil 3 teratas, pilih acak (agar tidak selalu pertanyaan sama)
        top = pool[:3]
        return random.choice(top)

    # ── Pembuatan Pertanyaan ──────────────────────────────────────────────────

    def _generate_question(self, hypothesis) -> Optional[str]:
        """
        Hasilkan kalimat tanya natural dari hipotesis.
        """
        category = hypothesis.category
        claim    = hypothesis.claim
        evidence = hypothesis.evidence

        templates = QUESTION_TEMPLATES.get(category, DEFAULT_TEMPLATES)
        template  = random.choice(templates)

        # Ekstrak hint dari evidence dan claim
        evidence_hint = self._extract_evidence_hint(evidence)
        topic_hint    = self._extract_topic_hint(claim)
        claim_short   = self._shorten_claim(claim)

        try:
            question = template.format(
                evidence_hint=evidence_hint,
                topic_hint=topic_hint,
                claim_short=claim_short,
            )
        except KeyError:
            # Fallback sederhana jika template punya key yang tidak tersedia
            question = f"Rofi, {claim_short} — bener nggak?"

        return question.strip()

    def _extract_evidence_hint(self, evidence: str) -> str:
        """Ubah evidence teknis jadi bahasa natural."""
        # Contoh: "aktif 12x di jam malam hari (20-23)" → "malam hari"
        if "malam" in evidence:
            return "malam hari"
        if "pagi" in evidence:
            return "pagi hari"
        if "siang" in evidence:
            return "siang hari"
        if "sore" in evidence:
            return "sore hari"
        # Cari jam dari pola "jam X"
        import re
        match = re.search(r"jam (\S+)", evidence)
        if match:
            return f"sekitar jam {match.group(1)}"
        return "waktu tertentu"

    def _extract_topic_hint(self, claim: str) -> str:
        """Ekstrak nama topik dari klaim."""
        # Contoh: "Rofi sering membicarakan topik kopi" → "kopi"
        import re
        match = re.search(r"topik (\w+)", claim)
        if match:
            return match.group(1)
        # Coba kata setelah "tentang"
        match = re.search(r"tentang (\w+)", claim)
        if match:
            return match.group(1)
        return claim.replace("Rofi ", "").replace("sering ", "")

    def _shorten_claim(self, claim: str) -> str:
        """Sederhanakan klaim jadi kalimat tanya pendek."""
        # Hilangkan awalan "Rofi " agar tidak redundan dalam template
        short = claim
        if short.startswith("Rofi "):
            short = short[5:]  # "sering aktif di malam hari"
        # Maksimal 60 karakter
        if len(short) > 60:
            short = short[:57] + "..."
        return short

    # ── Parsing Jawaban Rofi ──────────────────────────────────────────────────

    def _parse_response(self, text: str) -> str:
        """
        Parse jawaban Rofi jadi verdict.

        Return: "confirmed" | "rejected" | "unclear"

        Rofi bicara bahasa Indonesia/Sunda — tangkap kata kunci umum.
        """
        t = text.lower().strip()

        # Sinyal setuju
        YES_SIGNALS = [
            "iya", "ya", "yep", "yap", "bener", "benar", "betul",
            "tepat", "oke", "ok", "yoi", "tentu", "emang", "memang",
            "beneran", "pastinya", "pasti", "sip", "setuju", "confirm",
        ]
        # Sinyal tidak setuju
        NO_SIGNALS = [
            "tidak", "nggak", "ngga", "ga", "gak", "bukan", "salah",
            "keliru", "nope", "nah", "enggak", "engg", "ndak",
        ]

        # Hitung kemunculan sinyal
        yes_score = sum(1 for s in YES_SIGNALS if s in t.split())
        no_score  = sum(1 for s in NO_SIGNALS  if s in t.split())

        # Periksa juga frasa negatif umum ("bukan gitu", "ya nggak")
        if "bukan" in t or "nggak" in t or "tidak" in t:
            no_score += 1

        if yes_score > no_score and yes_score > 0:
            return "confirmed"
        if no_score > yes_score and no_score > 0:
            return "rejected"
        return "unclear"

    # ── Timing ───────────────────────────────────────────────────────────────

    def _is_good_time(self) -> bool:
        """
        Periksa apakah ini waktu yang tepat untuk bertanya.

        Kriteria:
          - Jam lokal ada di SAFE_HOURS
          - Sudah lewat MIN_GAP_MINUTES sejak pertanyaan terakhir
          - Tidak ada hipotesis yang sedang menunggu jawaban
        """
        now = datetime.now()

        # Jangan tanya jika masih ada pertanyaan yang menggantung
        if self._pending_hypothesis_id is not None:
            logger.debug("[curiosity] Ada pending hipotesis, skip tanya.")
            return False

        # Jam tidak aman
        if now.hour not in SAFE_HOURS:
            logger.debug("[curiosity] Jam %d di luar SAFE_HOURS, skip.", now.hour)
            return False

        # Terlalu cepat dari pertanyaan terakhir
        if self._last_asked_at is not None:
            gap = now - self._last_asked_at
            if gap < timedelta(minutes=MIN_GAP_MINUTES):
                remaining = int((timedelta(minutes=MIN_GAP_MINUTES) - gap).total_seconds() / 60)
                logger.debug(
                    "[curiosity] Terlalu cepat. Tunggu %d menit lagi.", remaining
                )
                return False

        return True

    # ── Persistensi State ─────────────────────────────────────────────────────

    def _save_state(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "last_asked_at": self._last_asked_at.isoformat() if self._last_asked_at else None,
            "pending_hypothesis_id": self._pending_hypothesis_id,
        }
        try:
            STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        except OSError as e:
            logger.error("[curiosity] Gagal simpan state: %s", e)

    def _load_state(self) -> None:
        if not STATE_FILE.exists():
            return
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            raw   = state.get("last_asked_at")
            self._last_asked_at = datetime.fromisoformat(raw) if raw else None
            self._pending_hypothesis_id = state.get("pending_hypothesis_id")
        except Exception as e:
            logger.warning("[curiosity] Gagal load state: %s", e)


# ──────────────────────────── Singleton Helper ───────────────────────────────

_curiosity_instance: Optional[Curiosity] = None

def get_curiosity() -> Curiosity:
    if _curiosity_instance is None:
        raise RuntimeError("Curiosity belum diinisialisasi. Panggil init_curiosity(profiler) dulu.")
    return _curiosity_instance

def init_curiosity(profiler) -> Curiosity:
    global _curiosity_instance
    _curiosity_instance = Curiosity(profiler)
    return _curiosity_instance


# ──────────────────────────── Quick Test ─────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    # Mock Hypothesis
    from dataclasses import dataclass, field
    from uuid import uuid4

    @dataclass
    class MockHypothesis:
        id:          str   = field(default_factory=lambda: uuid4().hex[:8])
        category:    str   = "schedule"
        claim:       str   = "Rofi sering aktif di malam hari"
        evidence:    str   = "aktif 12x di jam malam hari"
        confidence:  float = 0.8
        status:      str   = "pending"
        asked_count: int   = 0

    # Mock Profiler
    class MockProfiler:
        def __init__(self):
            self._hyp = [
                MockHypothesis(category="schedule", claim="Rofi sering aktif di malam hari",
                               evidence="aktif 12x di jam malam hari", confidence=0.8),
                MockHypothesis(category="topic", claim="Rofi sering membicarakan topik kopi",
                               evidence="kata terkait 'kopi' muncul 8x", confidence=0.7),
                MockHypothesis(category="habit", claim="Rofi orang yang terjadwal dan sering mengatur pengingat",
                               evidence="skill 'reminder' dipanggil 6x", confidence=0.65),
            ]

        def get_pending(self):
            return [h for h in self._hyp if h.status == "pending"]

        def increment_asked(self, hid):
            for h in self._hyp:
                if h.id == hid:
                    h.asked_count += 1

        def confirm(self, hid):
            for h in self._hyp:
                if h.id == hid:
                    h.status = "confirmed"
                    print(f"  ✓ CONFIRMED: {h.claim}")

        def reject(self, hid):
            for h in self._hyp:
                if h.id == hid:
                    h.status = "rejected"
                    print(f"  ✗ REJECTED: {h.claim}")

    async def _test():
        # Override path untuk test
        global STATE_FILE
        STATE_FILE = Path("/tmp/otto_curiosity_state.json")

        profiler  = MockProfiler()
        curiosity = Curiosity(profiler)

        print("\n=== TRY ASK ===")
        # Force waktu aman untuk test
        curiosity._last_asked_at = None

        question, hyp_id = await curiosity.try_ask()
        if question:
            print(f"  Q: {question}")
            print(f"  ID: {hyp_id}")
        else:
            print("  (tidak ada pertanyaan — mungkin jam tidak aman)")
            # Force untuk demo
            h = profiler.get_pending()[0]
            question = curiosity._generate_question(h)
            hyp_id = h.id
            print(f"  [FORCE] Q: {question}")

        print("\n=== HANDLE RESPONSE ===")
        answers = ["iya bener banget", "nggak sih", "hmm mungkin"]
        for ans in answers:
            verdict = await curiosity.handle_response(hyp_id, ans)
            print(f"  Jawaban: '{ans}' → {verdict}")
            curiosity._pending_hypothesis_id = hyp_id  # reset untuk demo ulang

        print("\n=== TIMING CHECK ===")
        print(f"  SAFE_HOURS: {SAFE_HOURS[0]}–{SAFE_HOURS[-1]}")
        print(f"  MIN_GAP: {MIN_GAP_MINUTES} menit")
        print(f"  is_good_time (setelah baru tanya): {curiosity._is_good_time()}")
        curiosity._last_asked_at = None
        curiosity._pending_hypothesis_id = None
        print(f"  is_good_time (reset): {curiosity._is_good_time()}")

    asyncio.run(_test())
