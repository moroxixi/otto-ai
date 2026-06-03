"""
intelligence/context_triggers.py — Context-Triggered Proactivity
=================================================================
Otto tidak hanya proaktif karena jadwal (scheduler),
tapi juga karena PAHAM isi percakapan.

Filosofi:
  Scheduler = proaktif berdasarkan WAKTU
  ContextTrigger = proaktif berdasarkan KONTEKS

Contoh nyata:
  Rofi: "meeting jam 14 nih, deg-degan"
  → Otto deteksi: ada waktu spesifik + emosi
  → Simpan ke pending_triggers
  → Jam 13:45 → Otto tanya "Gimana meetingnya tadi, Rofi?"

  Rofi: "capek banget hari ini"
  → Otto deteksi: emosi negatif
  → Catat ke memory sebagai konteks mood
  → Bisa jadi hipotesis: "Rofi sering lelah di hari tertentu"

Tiga jenis trigger:
  1. TIME_MENTION   — sebutan waktu/deadline → follow-up setelah waktu itu lewat
  2. EMOTION_SIGNAL — sinyal emosi (positif/negatif) → catat ke memory
  3. INTENT_MENTION — niat/rencana → catat, follow-up nanti

Cara integrasi (dari brain.py):
    from intelligence.context_triggers import ContextTriggerEngine
    engine = ContextTriggerEngine(memory)
    await engine.process(user_text, otto_text)

    # Di app.py startup loop (tiap tick):
    due = engine.get_due_triggers()
    for trigger in due:
        await send_followup(trigger.followup_message)
        engine.mark_done(trigger.id)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from uuid import uuid4

logger = logging.getLogger("otto.intelligence.context_triggers")

# ─────────────────────────── Config ──────────────────────────────────────────

TRIGGERS_FILE = Path("/data/asd/otto-ai/data/context_triggers.json")

# Berapa menit sebelum waktu yang disebut Otto mulai follow-up
FOLLOWUP_LEAD_MINUTES = 10

# Maksimal trigger aktif sekaligus (agar tidak spam)
MAX_ACTIVE_TRIGGERS = 5

# Trigger kedaluwarsa setelah N jam jika tidak pernah due
TRIGGER_EXPIRY_HOURS = 24


# ─────────────────────────── Pola Deteksi ────────────────────────────────────

# Pola jam: "jam 14", "jam 2 siang", "14:00", "pukul 9"
_TIME_PATTERNS = [
    r'\bjam\s+(\d{1,2})(?::(\d{2}))?\s*(pagi|siang|sore|malam)?\b',
    r'\bpukul\s+(\d{1,2})(?::(\d{2}))?\b',
    r'\b(\d{1,2}):(\d{2})\b',
]

# Kata penanda deadline / acara
_EVENT_KEYWORDS = [
    "meeting", "rapat", "presentasi", "deadline", "interview",
    "ujian", "test", "appointment", "janji", "acara", "event",
    "demo", "pitch", "submit", "kumpul", "setor",
]

# Sinyal emosi negatif
_NEGATIVE_EMOTION = [
    "capek", "lelah", "stress", "stres", "pusing", "bingung",
    "galau", "sedih", "kesel", "kesal", "frustrasi", "overwhelmed",
    "burnout", "burn out", "mager", "males", "nggak semangat",
    "down", "bad mood", "mood jelek",
]

# Sinyal emosi positif
_POSITIVE_EMOTION = [
    "senang", "happy", "bahagia", "semangat", "excited", "antusias",
    "bangga", "lega", "puas", "berhasil", "sukses", "yes", "mantap",
]

# Sinyal niat/rencana
_INTENT_PATTERNS = [
    r'\b(mau|rencananya|berencana|pengen|pengin|ingin)\s+(coba|beli|mulai|bikin|buat|daftar|ikut|pergi)\b',
    r'\b(besok|minggu depan|bulan depan)\s+(mau|akan|rencananya)\b',
    r'\b(target|goal|tujuan)\s+\w+',
]


# ─────────────────────────── Model Trigger ───────────────────────────────────

@dataclass
class ContextTrigger:
    id:               str   = field(default_factory=lambda: uuid4().hex[:8])
    trigger_type:     str   = ""        # "time_mention" | "emotion" | "intent"
    original_text:    str   = ""        # teks asli yang memicu
    followup_message: str   = ""        # pesan follow-up yang akan dikirim
    due_at:           float = 0.0       # epoch timestamp kapan follow-up dikirim
    created_at:       float = field(default_factory=time.time)
    status:           str   = "active"  # "active" | "done" | "expired"
    context:          dict  = field(default_factory=dict)  # data tambahan

    def is_due(self) -> bool:
        return self.status == "active" and time.time() >= self.due_at

    def is_expired(self) -> bool:
        age_hours = (time.time() - self.created_at) / 3600
        return self.status == "active" and age_hours > TRIGGER_EXPIRY_HOURS

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ContextTrigger":
        known = {k: d[k] for k in d if k in ContextTrigger.__dataclass_fields__}
        known.setdefault("context", {})
        return ContextTrigger(**known)


# ─────────────────────────── Engine ──────────────────────────────────────────

class ContextTriggerEngine:
    """
    Proses percakapan → deteksi konteks → simpan trigger → kirim follow-up.

    Penggunaan (dari brain.py):
        engine = ContextTriggerEngine(memory)
        await engine.process(user_text, otto_text)

    Di app.py (tiap tick / setelah brain.think()):
        due = engine.get_due_triggers()
        for t in due:
            await send_to_rofi(t.followup_message)
            engine.mark_done(t.id)
    """

    def __init__(self, memory) -> None:
        self._memory   = memory
        self._triggers: list[ContextTrigger] = []
        self._load()
        logger.info(
            "[context_triggers] Siap. %d trigger aktif.",
            len(self.get_active_triggers()),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def process(self, user_text: str, otto_text: str = "") -> list[ContextTrigger]:
        """
        Analisis percakapan dan buat trigger jika ditemukan pola.
        Dipanggil setelah setiap brain.think().
        Return: list trigger baru yang dibuat sesi ini.
        """
        new_triggers: list[ContextTrigger] = []

        # Jangan buat lebih dari MAX_ACTIVE_TRIGGERS
        active_count = len(self.get_active_triggers())
        if active_count >= MAX_ACTIVE_TRIGGERS:
            logger.debug(
                "[context_triggers] Skip — sudah %d trigger aktif (maks %d).",
                active_count, MAX_ACTIVE_TRIGGERS,
            )
            return []

        # Expire dulu trigger lama
        self._expire_old_triggers()

        # 1. Deteksi time_mention
        time_trigger = self._detect_time_mention(user_text)
        if time_trigger:
            new_triggers.append(time_trigger)

        # 2. Deteksi emotion
        emotion_trigger = self._detect_emotion(user_text)
        if emotion_trigger:
            new_triggers.append(emotion_trigger)
            # Catat ke memory sebagai konteks mood — non-blocking
            asyncio.create_task(self._log_emotion_to_memory(user_text, emotion_trigger))

        # 3. Deteksi intent/rencana
        intent_trigger = self._detect_intent(user_text)
        if intent_trigger:
            new_triggers.append(intent_trigger)

        if new_triggers:
            self._triggers.extend(new_triggers)
            self._save()
            for t in new_triggers:
                logger.info(
                    "[context_triggers] +trigger [%s] due_in=%.0f mnt | %s",
                    t.trigger_type,
                    (t.due_at - time.time()) / 60,
                    t.followup_message[:60],
                )

        return new_triggers

    def get_due_triggers(self) -> list[ContextTrigger]:
        """Kembalikan trigger yang sudah waktunya dikirim ke Rofi."""
        return [t for t in self._triggers if t.is_due()]

    def get_active_triggers(self) -> list[ContextTrigger]:
        """Kembalikan semua trigger aktif (belum due, belum done, belum expired)."""
        return [t for t in self._triggers if t.status == "active"]

    def mark_done(self, trigger_id: str) -> bool:
        """Tandai trigger sebagai sudah dikirim."""
        for t in self._triggers:
            if t.id == trigger_id:
                t.status = "done"
                self._save()
                logger.info("[context_triggers] Trigger %s → done.", trigger_id)
                return True
        return False

    def get_trigger(self, trigger_id: str) -> Optional[ContextTrigger]:
        for t in self._triggers:
            if t.id == trigger_id:
                return t
        return None

    def summary(self) -> dict:
        return {
            "active":  len(self.get_active_triggers()),
            "due_now": len(self.get_due_triggers()),
            "total":   len(self._triggers),
        }

    # ── Deteksi: Time Mention ─────────────────────────────────────────────────

    def _detect_time_mention(self, text: str) -> Optional[ContextTrigger]:
        """
        Deteksi sebutan waktu + kata event dalam teks.
        Hanya buat trigger jika ada KEDUA elemen: waktu + kata event.

        Contoh yang trigger:
          "meeting jam 14 nih"  ← waktu + event
          "deadline besok jam 9" ← waktu + event

        Contoh yang TIDAK trigger:
          "jam 3 tadi aku makan"  ← waktu ada, tapi bukan event ke depan
          "meeting kapan nih?"    ← event ada, tapi waktu tidak spesifik
        """
        text_lower = text.lower()

        # Cek apakah ada kata event
        has_event = any(kw in text_lower for kw in _EVENT_KEYWORDS)
        if not has_event:
            return None

        # Cari jam spesifik
        hour, minute = self._extract_time(text_lower)
        if hour is None:
            return None

        # Hitung kapan follow-up harus dikirim
        now = datetime.now()
        target = now.replace(hour=hour, minute=minute or 0, second=0, microsecond=0)

        # Jika waktu sudah lewat hari ini → set besok
        if target <= now:
            target += timedelta(days=1)

        # Follow-up LEAD_MINUTES sebelum waktu itu
        followup_at = target - timedelta(minutes=FOLLOWUP_LEAD_MINUTES)

        # Jika followup_at sudah lewat juga → kirim segera dalam 1 menit
        if followup_at <= now:
            followup_at = now + timedelta(minutes=1)

        # Cari kata event yang ditemukan (untuk pesan follow-up)
        found_event = next((kw for kw in _EVENT_KEYWORDS if kw in text_lower), "acara")

        followup = (
            f"Eh Rofi, tadi kamu bilang ada {found_event} jam {hour:02d}:{minute or 0:02d}. "
            f"Gimana, udah siap?"
        )

        return ContextTrigger(
            trigger_type     = "time_mention",
            original_text    = text[:100],
            followup_message = followup,
            due_at           = followup_at.timestamp(),
            context          = {
                "event":   found_event,
                "hour":    hour,
                "minute":  minute or 0,
                "target":  target.isoformat(),
            },
        )

    def _extract_time(self, text: str) -> tuple[Optional[int], Optional[int]]:
        """Ekstrak jam dan menit dari teks. Return (hour, minute) atau (None, None)."""
        for pattern in _TIME_PATTERNS:
            m = re.search(pattern, text)
            if not m:
                continue
            groups = m.groups()
            try:
                hour   = int(groups[0])
                minute = int(groups[1]) if len(groups) > 1 and groups[1] else 0

                # Normalisasi AM/PM berdasarkan konteks siang/malam
                period = groups[2] if len(groups) > 2 else None
                if period == "siang" and hour < 12:
                    hour += 12  # "jam 2 siang" → 14
                elif period == "malam" and hour < 12:
                    hour += 12
                elif period == "pagi" and hour == 12:
                    hour = 0

                if 0 <= hour <= 23:
                    return hour, minute
            except (ValueError, TypeError, IndexError):
                continue
        return None, None

    # ── Deteksi: Emotion ──────────────────────────────────────────────────────

    def _detect_emotion(self, text: str) -> Optional[ContextTrigger]:
        """
        Deteksi sinyal emosi dari teks Rofi.
        Buat follow-up ringan + catat ke memory.

        Tidak buat trigger duplikat jika trigger emosi sudah aktif.
        """
        text_lower = text.lower()

        # Cek apakah sudah ada trigger emosi aktif
        has_active_emotion = any(
            t.trigger_type == "emotion" and t.status == "active"
            for t in self._triggers
        )
        if has_active_emotion:
            return None

        # Deteksi emosi negatif
        neg_found = [kw for kw in _NEGATIVE_EMOTION if kw in text_lower]
        if neg_found:
            emotion_word = neg_found[0]
            # Follow-up 30 menit kemudian — beri ruang dulu, baru tanya
            due_at = time.time() + (30 * 60)
            followup = f"Rofi, tadi kamu bilang {emotion_word}. Sekarang gimana, udah断an断 断断断断断断断断?"
            # Buat kalimat lebih natural
            followup = self._natural_emotion_followup(emotion_word, valence="negative")

            return ContextTrigger(
                trigger_type     = "emotion",
                original_text    = text[:100],
                followup_message = followup,
                due_at           = due_at,
                context          = {"emotion": emotion_word, "valence": "negative"},
            )

        # Deteksi emosi positif — follow-up lebih cepat (5 menit), nada ikut senang
        pos_found = [kw for kw in _POSITIVE_EMOTION if kw in text_lower]
        if pos_found:
            emotion_word = pos_found[0]
            due_at = time.time() + (5 * 60)
            followup = self._natural_emotion_followup(emotion_word, valence="positive")

            return ContextTrigger(
                trigger_type     = "emotion",
                original_text    = text[:100],
                followup_message = followup,
                due_at           = due_at,
                context          = {"emotion": emotion_word, "valence": "positive"},
            )

        return None

    def _natural_emotion_followup(self, emotion_word: str, valence: str) -> str:
        """Buat kalimat follow-up yang terasa natural, bukan kaku."""
        if valence == "negative":
            templates = [
                f"Eh Rofi, masih {emotion_word}? Mau cerita nggak?",
                f"Gimana sekarang, masih {emotion_word}?",
                f"Rofi, tadi bilang {emotion_word} — udah断an断 belum?",
            ]
            # Versi lebih bersih (tanpa karakter rusak dari bug di atas)
            templates = [
                f"Eh Rofi, masih {emotion_word}? Mau cerita nggak?",
                f"Gimana sekarang, masih ngerasa {emotion_word}?",
                f"Hei Rofi — udah mendingan dari tadi yang {emotion_word}?",
            ]
        else:
            templates = [
                f"Hei Rofi, tadi kamu bilang {emotion_word} — cerita dong!",
                f"Wah, ada kabar baik nih kayaknya? Tadi kamu kedengeran {emotion_word}.",
                f"Rofi kelihatan {emotion_word} tadi — ada yang mau diceritain?",
            ]

        import random
        return random.choice(templates)

    # ── Deteksi: Intent ───────────────────────────────────────────────────────

    def _detect_intent(self, text: str) -> Optional[ContextTrigger]:
        """
        Deteksi niat/rencana Rofi yang layak difollow-up nanti.
        Follow-up dikirim ~2 jam kemudian.

        Contoh: "mau coba diet minggu ini" → follow-up besok
        """
        text_lower = text.lower()

        matched_pattern = None
        for pattern in _INTENT_PATTERNS:
            m = re.search(pattern, text_lower)
            if m:
                matched_pattern = m.group(0)
                break

        if not matched_pattern:
            return None

        # Cek apakah sudah ada trigger intent aktif — satu per satu saja
        has_active_intent = any(
            t.trigger_type == "intent" and t.status == "active"
            for t in self._triggers
        )
        if has_active_intent:
            return None

        # Follow-up 2 jam kemudian
        due_at = time.time() + (2 * 60 * 60)

        # Ekstrak objek niat dari teks asli (kata setelah "mau/pengen/ingin")
        intent_object = self._extract_intent_object(text_lower)
        followup = (
            f"Eh Rofi, tadi kamu bilang {intent_object}. Gimana, jadi nggak?"
            if intent_object
            else "Rofi, tadi kayak ada rencana yang mau kamu lakuin — jadi nggak?"
        )

        return ContextTrigger(
            trigger_type     = "intent",
            original_text    = text[:100],
            followup_message = followup,
            due_at           = due_at,
            context          = {"matched": matched_pattern, "intent_object": intent_object or ""},
        )

    def _extract_intent_object(self, text: str) -> Optional[str]:
        """Ekstrak objek dari kalimat niat. Contoh: 'mau coba diet' → 'mau coba diet'."""
        # Ambil ~5 kata setelah kata kunci niat
        m = re.search(
            r'\b(mau|pengen|pengin|ingin|rencananya)\s+([\w\s]{3,40})',
            text
        )
        if m:
            # Ambil maks 5 kata
            raw = m.group(2).strip()
            words = raw.split()[:5]
            return " ".join(words)
        return None

    # ── Memory Logging ────────────────────────────────────────────────────────

    async def _log_emotion_to_memory(self, text: str, trigger: ContextTrigger) -> None:
        """
        Catat sinyal emosi ke long-term memory sebagai konteks mood.
        Ini bukan fakta permanen — lebih sebagai catatan pola.
        """
        try:
            emotion   = trigger.context.get("emotion", "unknown")
            valence   = trigger.context.get("valence", "unknown")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

            key   = f"rofi.mood.last_{valence}"
            value = f"{emotion} (terdeteksi {timestamp})"

            await asyncio.to_thread(
                self._memory.remember,
                key, value, f"observasi_mood ({valence})"
            )
            logger.debug("[context_triggers] Emosi dicatat ke memory: %s = %s", key, value)
        except Exception as e:
            logger.warning("[context_triggers] Gagal log emosi ke memory: %s", e)

    # ── Housekeeping ──────────────────────────────────────────────────────────

    def _expire_old_triggers(self) -> None:
        """Tandai trigger yang sudah kedaluwarsa."""
        for t in self._triggers:
            if t.is_expired():
                t.status = "expired"
                logger.debug("[context_triggers] Trigger %s expired.", t.id)

    # ── Persistensi ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        TRIGGERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Simpan hanya yang aktif + done terbaru (bersih dari expired lama)
        to_save = [
            t for t in self._triggers
            if t.status in ("active", "done")
        ][-50:]  # maks 50 entry
        try:
            TRIGGERS_FILE.write_text(
                json.dumps([t.to_dict() for t in to_save], ensure_ascii=False, indent=2)
            )
        except OSError as e:
            logger.error("[context_triggers] Gagal simpan: %s", e)

    def _load(self) -> None:
        if not TRIGGERS_FILE.exists():
            return
        try:
            data = json.loads(TRIGGERS_FILE.read_text(encoding="utf-8"))
            self._triggers = [ContextTrigger.from_dict(d) for d in data]
            # Expire langsung yang sudah kedaluwarsa
            self._expire_old_triggers()
            logger.info(
                "[context_triggers] Dimuat: %d trigger (%d aktif).",
                len(self._triggers),
                len(self.get_active_triggers()),
            )
        except Exception as e:
            logger.warning("[context_triggers] Gagal load: %s", e)
