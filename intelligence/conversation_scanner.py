"""
intelligence/conversation_scanner.py — Real-Time Signal Detector
================================================================
Membaca teks percakapan LANGSUNG saat terjadi — bukan batch 15 menit.
Tugasnya satu: deteksi sinyal penting → inject hipotesis ke Profiler.

Filosofi:
  Profiler.analyze() = batch, berbasis activity log (tiap 15 menit)
  ConversationScanner = real-time, berbasis isi kalimat (tiap pesan)

  Keduanya tidak saling menggantikan — mereka saling melengkapi.
  Scanner tangkap sinyal eksplisit ("aku suka kopi").
  Profiler tangkap pola implisit ("Rofi tanya kopi 8x minggu ini").

Cara kerja:
  1. Terima user_text + otto_text dari brain.py
  2. Jalankan SIGNAL_RULES — list aturan berbasis keyword
  3. Setiap rule yang match → buat Hypothesis langsung
  4. Inject ke Profiler (lewat inject_hypothesis)
  5. Selesai — tidak ada return value, tidak ada blocking

Integrasi (di brain.py):
    # Di __init__:
    from intelligence.conversation_scanner import ConversationScanner
    self._scanner = ConversationScanner(memory, profiler)

    # Di think() dan think_stream(), setelah _log_to_memory:
    asyncio.create_task(self._scan_conversation(user_text, otto_text))

    # Method baru di Brain:
    async def _scan_conversation(self, user_text, otto_text):
        await self._scanner.scan(user_text, source="user")
        await self._scanner.scan(otto_text, source="otto")

Desain sadar:
  - TIDAK pakai LLM untuk scan → latensi 0, tidak buang token
  - TIDAK block think() → create_task, jalan di background
  - Duplikat ditangani Profiler (claim yang sama diabaikan)
  - Confidence scanner sengaja lebih rendah (0.5–0.75) karena
    satu kalimat bukan bukti kuat — Profiler yang naikkan lewat pola
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger("otto.intelligence.scanner")






# ─────────────────────────── Negation Detection ──────────────────────────────

# Kata yang mengindikasikan kalimat berbicara tentang masa lalu / negasi
_NEGATION_MARKERS = [
    "dulu", "bukan", "tidak lagi", "sudah tidak", "udah tidak",
    "pernah", "tapi sekarang", "tapi udah", "tapi sudah",
    "nggak lagi", "ngga lagi", "ga lagi",
]

# Kata yang boost confidence — menunjukkan kebiasaan aktif
_HABIT_BOOSTERS = [
    "biasanya", "masih", "selalu", "rutin", "tiap", "setiap", "sering",
]

# Kata yang turunkan confidence — menunjukkan kejadian sesaat
_TEMPORAL_DAMPENERS = [
    "kemarin", "tadi", "sekali", "pernah", "waktu itu", "dulu",
    "satu kali", "kebetulan",
]


# Kata yang menunjukkan kalimat tentang orang LAIN (bukan Rofi)
_THIRD_PERSON_MARKERS = [
    "temenku", "temanku", "sohibku", "kawanku",
    "kakakku", "adikku", "adek", "kakak",
    "istri", "suami", "pacar", "bokap", "nyokap",
    "orang tua", "ortuku", "dia ", "dia,", "dia.",
    "mereka", "anakku", "ponakan",
    "temen gue", "temen gw",
]

# Kata yang mengindikasikan kalimat MEMANG tentang diri sendiri
_SELF_MARKERS = [
    "aku", "gue", "gw", "saya", "w ", "w,",
    "ane", "ana",
]


def _apply_confidence_modifier(text: str, base_confidence: float) -> float:
    """
    Modifikasi confidence berdasarkan konteks temporal dan negasi.

    - Negasi / masa lalu  → kalikan 0.2 (hampir hapus)
    - Habit booster       → tambah 0.1 (max 0.9)
    - Temporal dampener   → kurangi 0.15

    Return confidence final (float 0.0–1.0).
    """
    t = text.lower()
    conf = base_confidence

    # Cek negasi dulu — kalau ada, langsung drastis turun
    for marker in _NEGATION_MARKERS:
        if marker in t:
            conf *= 0.2
            return round(max(0.0, conf), 3)

    # Boost jika kata kebiasaan aktif
    for booster in _HABIT_BOOSTERS:
        if booster in t:
            conf = min(0.9, conf + 0.1)
            break  # cukup satu boost

    # Dampen jika kata konteks sesaat
    for dampener in _TEMPORAL_DAMPENERS:
        if dampener in t:
            conf = max(0.0, conf - 0.15)
            break  # cukup satu dampener

    return round(conf, 3)


# ─────────────────────────── Model Signal ────────────────────────────────────

@dataclass
class SignalHit:
    """Satu sinyal yang terdeteksi dari satu kalimat."""
    rule_id:    str    # ID rule yang match, untuk debug
    category:   str    # "preference", "habit", "schedule", "topic"
    claim:      str    # kalimat hipotesis yang akan dibuat
    evidence:   str    # bukti: potongan teks asli
    confidence: float  # 0.0–1.0, sengaja rendah (single utterance)


# ─────────────────────────── Signal Rules ─────────────────────────────────────
#
# Setiap rule punya:
#   id          : nama unik untuk debug
#   patterns    : list regex (OR) — salah satu match = trigger
#   source      : "user" (hanya dari Rofi) | "both" (Rofi + Otto)
#   category    : kategori hipotesis
#   claim_fn    : callable(match) → str kalimat hipotesis
#   confidence  : float, berapa yakin dari satu kalimat
#
# Tips menambah rule:
#   - Jaga pattern tetap spesifik — jangan tangkap false positive
#   - confidence max 0.75 (satu kalimat bukan bukti kuat)
#   - claim_fn harus deterministik (kalimat sama → claim sama)
#     agar Profiler bisa deteksi duplikat

@dataclass
class SignalRule:
    id:         str
    patterns:   list[str]       # regex patterns (OR logic)
    source:     str             # "user" | "both"
    category:   str
    claim_fn:   object          # callable(re.Match, text) → str
    confidence: float
    _compiled:  list = None     # internal, diisi di __post_init__

    def __post_init__(self):
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    def match(self, text: str) -> Optional[re.Match]:
        """Return first match atau None."""
        for pat in self._compiled:
            m = pat.search(text)
            if m:
                return m
        return None


# ── Definisi Rules ─────────────────────────────────────────────────────────────

def _build_rules() -> list[SignalRule]:
    return [

        # ── PREFERENSI MINUMAN ──────────────────────────────────────────────
        SignalRule(
            id         = "suka_kopi",
            patterns   = [
                r"\b(suka|doyan|prefer|favorit)\b.{0,20}\bkopi\b",
                r"\bkopi\b.{0,20}\b(enak|suka|doyan|favorit)\b",
                r"\btiap (pagi|hari|malam)\b.{0,20}\bkopi\b",
                r"\bkopi\b.{0,20}\btiap (pagi|hari|malam)\b",
            ],
            source     = "user",
            category   = "preference",
            claim_fn   = lambda m, t: "Rofi suka minum kopi",
            confidence = 0.7,
        ),

        SignalRule(
            id         = "kopi_oat",
            patterns   = [
                r"\bkopi\b.{0,15}\boat\b",
                r"\boat\b.{0,15}\bkopi\b",
            ],
            source     = "user",
            category   = "preference",
            claim_fn   = lambda m, t: "Rofi suka kopi dengan susu oat",
            confidence = 0.75,
        ),

        SignalRule(
            id         = "suka_teh",
            patterns   = [
                r"\b(suka|doyan|favorit)\b.{0,20}\bteh\b",
                r"\bteh\b.{0,15}\b(enak|suka|doyan)\b",
            ],
            source     = "user",
            category   = "preference",
            claim_fn   = lambda m, t: "Rofi suka minum teh",
            confidence = 0.65,
        ),

        # ── PREFERENSI MUSIK ────────────────────────────────────────────────
        SignalRule(
            id         = "suka_musik_santai",
            patterns   = [
                r"\b(suka|doyan|prefer)\b.{0,25}\b(musik|lagu)\b.{0,20}\b(santai|tenang|slow)\b",
                r"\b(santai|tenang|slow)\b.{0,20}\b(musik|lagu)\b",
            ],
            source     = "user",
            category   = "preference",
            claim_fn   = lambda m, t: "Rofi suka musik atau lagu yang santai",
            confidence = 0.65,
        ),

        SignalRule(
            id         = "suka_musik_jadul",
            patterns   = [
                r"\b(suka|doyan|favorit)\b.{0,25}\b(jadul|lawas|lama|klasik|oldies)\b",
                r"\blagu\b.{0,20}\b(jadul|lawas|lama|klasik)\b",
            ],
            source     = "user",
            category   = "preference",
            claim_fn   = lambda m, t: "Rofi suka lagu-lagu jadul atau klasik",
            confidence = 0.65,
        ),

        # ── JADWAL / KEBIASAAN WAKTU ────────────────────────────────────────
        SignalRule(
            id         = "aktif_pagi",
            patterns   = [
                r"\b(biasa|sering|selalu|rutin)\b.{0,20}\bpagi\b",
                r"\bpagi\b.{0,20}\b(biasa|sering|selalu|rutin|bangun|mulai)\b",
                r"\bbangun\b.{0,20}\bjam\s*[4-8]\b",
            ],
            source     = "user",
            category   = "schedule",
            claim_fn   = lambda m, t: "Rofi cenderung aktif atau mulai hari di pagi hari",
            confidence = 0.6,
        ),

        SignalRule(
            id         = "aktif_malam",
            patterns   = [
                r"\b(biasa|sering|selalu)\b.{0,20}\bmalam\b",
                r"\bmalam\b.{0,20}\b(kerja|ngoding|baca|aktif)\b",
                r"\btidur\b.{0,20}\bjam\s*(2[0-9]|[01][0-9])\b",
            ],
            source     = "user",
            category   = "schedule",
            claim_fn   = lambda m, t: "Rofi sering aktif di malam hari",
            confidence = 0.6,
        ),

        SignalRule(
            id         = "tidur_jam_tertentu",
            patterns   = [
                r"\btidur\b.{0,15}\bjam\s*(\d+)\b",
                r"\bjam\s*(\d+)\b.{0,15}\btidur\b",
            ],
            source     = "user",
            category   = "schedule",
            claim_fn   = lambda m, t: f"Rofi punya jadwal tidur tertentu (dari percakapan: '{t[:60].strip()}')",
            confidence = 0.55,
        ),

        # ── PEKERJAAN / BISNIS ──────────────────────────────────────────────
        SignalRule(
            id         = "punya_usaha",
            patterns   = [
                r"\b(usaha|bisnis|toko|cabang|warung)\b.{0,20}\b(saya|aku|punya|buka)\b",
                r"\b(punya|buka|kelola)\b.{0,20}\b(usaha|bisnis|toko|cabang|warung)\b",
            ],
            source     = "user",
            category   = "habit",
            claim_fn   = lambda m, t: "Rofi memiliki usaha atau bisnis sendiri",
            confidence = 0.7,
        ),

        SignalRule(
            id         = "kerja_dari_rumah",
            patterns   = [
                r"\b(kerja|wfh|work from home)\b.{0,20}\b(rumah|rumahan|wfh)\b",
                r"\bwfh\b",
            ],
            source     = "user",
            category   = "habit",
            claim_fn   = lambda m, t: "Rofi bekerja dari rumah (WFH)",
            confidence = 0.65,
        ),

        # ── KESEHATAN ───────────────────────────────────────────────────────
        SignalRule(
            id         = "pantau_berat_badan",
            patterns   = [
                r"\b(timbang|berat badan|bb|turun berat|diet)\b",
                r"\bberat\b.{0,15}\b(badan|naik|turun|turun|target)\b",
            ],
            source     = "user",
            category   = "habit",
            claim_fn   = lambda m, t: "Rofi aktif memantau berat badan atau sedang diet",
            confidence = 0.6,
        ),

        SignalRule(
            id         = "minum_obat",
            patterns   = [
                r"\b(minum|konsumsi)\b.{0,15}\bobat\b",
                r"\bobat\b.{0,15}\b(rutin|tiap|harus)\b",
            ],
            source     = "user",
            category   = "habit",
            claim_fn   = lambda m, t: "Rofi mengonsumsi obat atau suplemen secara rutin",
            confidence = 0.55,
        ),

        # ── TEKNOLOGI / HOBI ────────────────────────────────────────────────
        SignalRule(
            id         = "suka_ngoding",
            patterns   = [
                r"\b(ngoding|coding|nulis kode|programming|develop)\b",
                r"\b(python|javascript|rust|golang|flutter)\b.{0,20}\b(suka|pakai|biasa)\b",
            ],
            source     = "user",
            category   = "topic",
            claim_fn   = lambda m, t: "Rofi suka atau aktif di dunia coding / programming",
            confidence = 0.65,
        ),

        SignalRule(
            id         = "pakai_linux",
            patterns   = [
                r"\b(linux|arch|ubuntu|opensuse|fedora|nixos|hyprland)\b",
            ],
            source     = "user",
            category   = "topic",
            claim_fn   = lambda m, t: "Rofi menggunakan Linux sebagai sistem operasi",
            confidence = 0.7,
        ),

        # ── EMOSI / KONDISI ─────────────────────────────────────────────────
        SignalRule(
            id         = "sering_capek",
            patterns   = [
                r"\b(capek|lelah|exhausted|kelelahan)\b.{0,20}\b(kerja|hari ini|tadi|banget)\b",
                r"\b(kerja|aktivitas)\b.{0,20}\b(capek|lelah|melelahkan)\b",
            ],
            source     = "user",
            category   = "habit",
            claim_fn   = lambda m, t: "Rofi sering merasa lelah setelah aktivitas sehari-hari",
            confidence = 0.5,  # rendah — bisa konteks sesaat
        ),

        SignalRule(
            id         = "suka_santai",
            patterns   = [
                r"\b(pengen|mau|butuh)\b.{0,20}\b(santai|istirahat|rebahan|relax)\b",
                r"\bme time\b",
            ],
            source     = "user",
            category   = "preference",
            claim_fn   = lambda m, t: "Rofi menghargai waktu santai dan istirahat",
            confidence = 0.5,
        ),

        # ── PREFERENSI KOMUNIKASI ───────────────────────────────────────────
        SignalRule(
            id         = "feedback_terlalu_banyak_pertanyaan",
            patterns   = [
                r"\b(jangan|ga usah|gausah|stop)\b.{0,30}\b(tanya|nanya)\b.{0,20}\b(banyak|banyak-banyak|melulu|terus)\b",
                r"\btanya\b.{0,20}\b(satu|1)\b.{0,20}\b(aja|saja|dulu|cukup)\b",
                r"\bbingung\b.{0,30}\b(harus jawab|mau jawab|jawab)\b.{0,20}\b(yang mana|duluan)\b",
                r"\bterlalu banyak pertanyaan\b",
                r"\b(banyak banget|kebanyakan)\b.{0,20}\b(pertanyaan|nanya|tanya)\b",
            ],
            source     = "user",
            category   = "preference",
            claim_fn   = lambda m, t: "Rofi lebih nyaman jika Otto hanya mengajukan satu pertanyaan per respons",
            confidence = 0.85,
        ),

        SignalRule(
            id         = "feedback_respon_terlalu_panjang",
            patterns   = [
                r"\b(jangan|ga usah|gausah)\b.{0,20}\b(panjang|bertele|tele)\b",
                r"\b(singkat|pendek|to the point|ringkas)\b.{0,20}\b(aja|saja|dong|please)\b",
                r"\bterlalu\b.{0,20}\b(panjang|verbose|bertele)\b",
            ],
            source     = "user",
            category   = "preference",
            claim_fn   = lambda m, t: "Rofi lebih suka respons Otto yang singkat dan to the point",
            confidence = 0.8,
        ),

        SignalRule(
            id         = "feedback_respon_terlalu_formal",
            patterns   = [
                r"\b(jangan|ga perlu|gaperlu)\b.{0,20}\b(formal|kaku|sopan banget)\b",
                r"\b(santai|casual|informal)\b.{0,20}\b(aja|saja|dong)\b.{0,20}\b(ngobrol|bicara|jawab)\b",
            ],
            source     = "user",
            category   = "preference",
            claim_fn   = lambda m, t: "Rofi lebih suka Otto bicara santai dan tidak terlalu formal",
            confidence = 0.75,
        ),
    ]
    




def _is_about_rofi(text: str) -> bool:
    """
    Cek apakah kalimat berbicara tentang Rofi sendiri, bukan orang lain.

    Strategi:
      1. Jika ada _THIRD_PERSON_MARKERS → kemungkinan besar tentang orang lain
      2. Jika ada _SELF_MARKERS → kemungkinan tentang diri sendiri (Rofi)
      3. Jika tidak ada keduanya → biarkan lolos (ambiguous, lebih baik false positive
         daripada miss sinyal nyata)

    Catatan: Ini bukan NLP sempurna — hanya filter kasar untuk kasus paling jelas.
    Kalimat ambigu ("dia suka kopi tapi aku juga") tetap lolos karena ada self marker.
    """
    t = text.lower()

    # Cek third person dulu
    has_third_person = any(marker in t for marker in _THIRD_PERSON_MARKERS)

    if not has_third_person:
        return True  # Tidak ada indikasi orang lain → anggap tentang Rofi

    # Ada third person — cek apakah JUGA ada self marker
    # Kalimat seperti "temenku suka kopi tapi aku nggak" → ada keduanya
    # Dalam kasus ini, kalimat ambigu — biarkan lolos tapi dengan catatan
    has_self = any(marker in t for marker in _SELF_MARKERS)

    if has_self:
        # Ambigu: ada "temen" tapi juga ada "aku"
        # Cek urutan: siapa yang disebut lebih dulu setelah kata kunci?
        # Heuristik sederhana: jika third person muncul SEBELUM self marker → tentang orang lain
        first_third = min(
            (t.find(m) for m in _THIRD_PERSON_MARKERS if m in t),
            default=9999,
        )
        first_self = min(
            (t.find(m) for m in _SELF_MARKERS if m in t),
            default=9999,
        )
        # Jika self marker lebih dulu → kalimat dimulai dari perspektif Rofi
        return first_self <= first_third

    # Ada third person, tidak ada self marker → tentang orang lain
    return False

# ─────────────────────────── Scanner ─────────────────────────────────────────

class ConversationScanner:
    """
    Scan satu kalimat → deteksi sinyal → inject ke Profiler.

    Dipanggil oleh brain.py setelah setiap response, non-blocking.

    Contoh:
        scanner = ConversationScanner(profiler)
        await scanner.scan("aku tiap pagi minum kopi oat", source="user")
        # → inject Hypothesis("Rofi suka kopi dengan susu oat", conf=0.75)
    """

    def __init__(self, profiler) -> None:
        self._profiler = profiler
        self._rules    = _build_rules()
        logger.info("[scanner] Siap. %d rules aktif.", len(self._rules))

    async def scan(self, text: str, source: str = "user") -> list[SignalHit]:
        """
        Scan text dan inject hipotesis yang ditemukan.

        Args:
            text   : Teks dari Rofi ("user") atau dari Otto ("otto")
            source : "user" | "otto"

        Return:
            list[SignalHit] — sinyal yang ditemukan (untuk logging/debug)
        """
        about_rofi = _is_about_rofi(text)

        if not text or not text.strip():
            return []

        hits: list[SignalHit] = []

        for rule in self._rules:
            # Skip rule yang hanya untuk "user" jika source adalah "otto"
            if rule.source == "user" and source != "user":
                continue

            match = rule.match(text)
            if not match:
                continue
            # Fix 2: Cek apakah kalimat tentang Rofi, bukan orang lain
            if not about_rofi:
                logger.debug(
                    "[scanner] Skip rule '%s' — kalimat tentang orang lain: \"%s\"",
                    rule.id, text[:60],
                )
                continue

            try:
                claim = rule.claim_fn(match, text)
            except Exception as e:
                logger.warning("[scanner] claim_fn error di rule '%s': %s", rule.id, e)
                continue

            final_confidence = _apply_confidence_modifier(text, rule.confidence)

            # Skip jika confidence jatuh terlalu rendah setelah negasi
            if final_confidence < 0.1:
                logger.debug(
                    "[scanner] Skip rule '%s' — confidence terlalu rendah setelah negasi (%.2f)",
                    rule.id, final_confidence,
                )
                continue
            
            hit = SignalHit(
                rule_id    = rule.id,
                category   = rule.category,
                claim      = claim,
                evidence   = f"terdeteksi dari percakapan: \"{text[:80].strip()}\"",
                confidence = final_confidence,  # ← pakai yang sudah dimodifikasi
            )
            hits.append(hit)

            logger.info(
                "[scanner] ✓ rule='%s' conf=%.0f%% claim='%s'",
                rule.id, final_confidence * 100, claim,
            )

        # Inject ke Profiler (di thread terpisah agar tidak block event loop)
        if hits:
            await asyncio.to_thread(self._inject_hits, hits)

        return hits

    # AFTER
    def _inject_hits(self, hits: list[SignalHit]) -> None:
        for hit in hits:
            fact = {
                "key":        f"rofi.{hit.category}.{hit.rule_id}",
                "value":      hit.claim.replace("Rofi ", "", 1),
                "confidence": hit.confidence,
            }
    
            injected = self._profiler.inject_hypothesis(fact)
    
            if injected:
                logger.info(
                    "[scanner] → inject hipotesis baru [%s] %.0f%%: %s",
                    hit.category, hit.confidence * 100, hit.claim,
                )
            else:
                logger.debug(
                    "[scanner] Skip duplikat (ditangani profiler): '%s'", hit.claim
                )
    
# ─────────────────────────── Quick Test ──────────────────────────────────────

if __name__ == "__main__":
    import asyncio, logging
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")

    class MockProfiler:
        def __init__(self):
            self._hypotheses = []

        def get_all(self):
            return self._hypotheses

        def get_pending(self):
            return [h for h in self._hypotheses if h.status == "pending"]

        # AFTER
        def inject_hypothesis(self, fact: dict):
            claim = f"Rofi {fact.get('value', '')}"
            if claim not in {h.get("claim", "") for h in self._hypotheses}:
                self._hypotheses.append({"claim": claim, "status": "pending",
                                          "category": fact.get("key", "").split(".")[1] if "." in fact.get("key","") else "preference",
                                          "confidence": fact.get("confidence", 0.5)})
                return True
            return False

        def _save(self):
            pass  # no-op untuk test

    async def _test():
        profiler = MockProfiler()
        scanner  = ConversationScanner(profiler)

        test_cases = [
            ("user", "aku tiap pagi minum kopi oat, udah kebiasaan banget"),
            ("user", "lagi dengerin lagu jadul, enak banget santai gini"),
            ("user", "wfh hari ini, capek juga sih kerja dari rumah"),
            ("user", "punya 4 cabang usaha, lumayan ribet ngaturnya"),
            ("user", "pakai opensuse tumbleweed sama hyprland"),
            ("user", "cuaca hari ini gimana?"),   # ← tidak ada sinyal → 0 hit
        ]

        total_hits = 0
        for source, text in test_cases:
            print(f"\n[INPUT] ({source}) \"{text}\"")
            hits = await scanner.scan(text, source=source)
            print(f"  → {len(hits)} sinyal:")
            for h in hits:
                print(f"     [{h.rule_id}] {h.confidence:.0%} — {h.claim}")
            total_hits += len(hits)

        print(f"\n=== TOTAL: {total_hits} sinyal, {len(profiler.get_pending())} hipotesis diinjeksi ===")
        # AFTER
        for h in profiler.get_pending():
            print(f"  [{h['category']}] {h['confidence']:.0%} — {h['claim']}")

    asyncio.run(_test())
