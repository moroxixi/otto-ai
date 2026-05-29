"""
intelligence/activity_watcher.py — Pengamat Aktivitas Rofi
============================================================
Modul ini adalah "mata" Otto. Ia diam-diam mencatat semua
interaksi Rofi — bukan untuk langsung menyimpulkan, tapi untuk
memberi bahan mentah ke profiler.py dan curiosity.py.

Filosofi:
  Lapisan 2 Otto — OBSERVATIF
  "Otto belum tahu apa-apa. Ia hanya mencatat."

Yang dicatat:
  - Setiap utterance Rofi (teks, waktu, intent yang terdeteksi)
  - Pola waktu (jam berapa Rofi biasanya ngobrol)
  - Topik yang sering muncul (keyword frequency)
  - Skill apa yang sering dipanggil

Yang TIDAK dilakukan di sini:
  - Tidak menyimpulkan kepribadian Rofi
  - Tidak bertanya ke Rofi
  - Tidak mengubah behavior Otto
  → Itu tugas profiler.py dan curiosity.py

Cara integrasi (dari app.py):
    from intelligence.activity_watcher import ActivityWatcher
    watcher = ActivityWatcher()
    # Panggil setiap ada interaksi:
    await watcher.log(text=text, intent="command", skill="reminder")
    # Ambil ringkasan untuk profiler:
    summary = watcher.get_summary()
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("otto.intelligence.activity_watcher")

# ─────────────────────────── Konfigurasi ────────────────────────────────────

ACTIVITY_LOG  = Path("/data/asd/otto-ai/data/activity.log")
ACTIVITY_JSON = Path("/data/asd/otto-ai/data/activity_summary.json")

# Berapa banyak entri disimpan di memory sebelum flush ke disk
BUFFER_SIZE = 50

# Window analisis untuk pola harian (jam)
HOUR_BUCKETS = list(range(24))


# ─────────────────────────── Model Data ─────────────────────────────────────

class ActivityEntry:
    """Satu baris aktivitas Rofi."""

    __slots__ = ("timestamp", "text", "intent", "skill", "hour", "weekday", "keywords")

    def __init__(
        self,
        text:   str,
        intent: str = "chat",
        skill:  str = "",
    ) -> None:
        now              = datetime.now()
        self.timestamp   = now.isoformat(timespec="seconds")
        self.text        = text.strip()
        self.intent      = intent   # "command" | "chat"
        self.skill       = skill    # nama skill jika command, "" jika chat
        self.hour        = now.hour
        self.weekday     = now.weekday()   # 0=Senin … 6=Minggu
        self.keywords    = _extract_keywords(text)

    def to_dict(self) -> dict:
        return {
            "ts":       self.timestamp,
            "text":     self.text,
            "intent":   self.intent,
            "skill":    self.skill,
            "hour":     self.hour,
            "weekday":  self.weekday,
            "keywords": self.keywords,
        }


# ─────────────────────────── Keyword Extractor ───────────────────────────────

# Kata-kata yang tidak informatif (stopwords sederhana Bahasa Indonesia)
_STOPWORDS = frozenset({
    "aku", "kamu", "dia", "itu", "ini", "yang", "dan", "atau",
    "ke", "di", "dari", "untuk", "dengan", "ada", "bisa", "mau",
    "aja", "dong", "deh", "ya", "yuk", "nih", "sih", "lagi",
    "sudah", "sudah", "belum", "tidak", "nggak", "gak", "tak",
    "tolong", "please", "coba", "ok", "oke", "iya", "nah",
    "gimana", "bagaimana", "kapan", "dimana", "siapa", "apa",
    "otto", "libo",
})

def _extract_keywords(text: str) -> list[str]:
    """
    Ambil kata-kata bermakna dari teks.
    Hasil: list lowercase, tanpa stopwords, min 3 karakter.
    """
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return [w for w in words if w not in _STOPWORDS]


# ─────────────────────────── Activity Watcher ────────────────────────────────

class ActivityWatcher:
    """
    Mencatat dan menganalisis pola aktivitas Rofi.

    Penggunaan:
        watcher = ActivityWatcher()
        await watcher.log("ingatkan aku minum obat 30 menit lagi",
                          intent="command", skill="reminder")
        summary = watcher.get_summary()
    """

    def __init__(self) -> None:
        self._buffer: list[ActivityEntry] = []

        # Counter in-memory (tidak perlu baca disk setiap query)
        self._keyword_counter: Counter    = Counter()
        self._skill_counter:   Counter    = Counter()
        self._hour_counter:    Counter    = Counter()
        self._weekday_counter: Counter    = Counter()
        self._total_logged:    int        = 0

        # Muat riwayat counter dari disk jika ada
        self._load_summary()

        logger.info("[activity_watcher] Siap. %d aktivitas terdapat di riwayat.", self._total_logged)

    # ── Public API ────────────────────────────────────────────────────────────

    async def log(
        self,
        text:   str,
        intent: str = "chat",
        skill:  str = "",
    ) -> None:
        """
        Catat satu aktivitas. Panggil ini dari app.py setiap ada interaksi.

        Args:
            text:   Teks asli dari Rofi
            intent: "command" atau "chat"
            skill:  Nama skill yang dipakai (kosong jika chat)
        """
        entry = ActivityEntry(text=text, intent=intent, skill=skill)

        # Update counter in-memory
        self._keyword_counter.update(entry.keywords)
        if skill:
            self._skill_counter[skill] += 1
        self._hour_counter[entry.hour] += 1
        self._weekday_counter[entry.weekday] += 1
        self._total_logged += 1

        # Buffer → flush ke disk jika penuh
        self._buffer.append(entry)
        if len(self._buffer) >= BUFFER_SIZE:
            await self._flush()

        logger.debug(
            "[watcher] +log | intent=%s skill=%s hour=%d kw=%s",
            intent, skill or "-", entry.hour, entry.keywords[:3],
        )

    def get_summary(self) -> dict:
        """
        Kembalikan ringkasan aktivitas untuk profiler.py.

        Return:
            {
                "total":         int,
                "top_keywords":  [(word, count), ...],    # 20 teratas
                "top_skills":    [(skill, count), ...],   # 10 teratas
                "active_hours":  [(hour, count), ...],    # jam paling aktif
                "active_days":   [(weekday_name, count), ...],
                "patterns":      { ... }                  # pola turunan
            }
        """
        DAYS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]

        top_kw    = self._keyword_counter.most_common(20)
        top_sk    = self._skill_counter.most_common(10)
        top_hours = self._hour_counter.most_common(5)
        top_days  = [
            (DAYS[d], c)
            for d, c in self._weekday_counter.most_common()
        ]

        # Pola sederhana
        patterns = self._derive_patterns(top_hours)

        return {
            "total":        self._total_logged,
            "top_keywords": top_kw,
            "top_skills":   top_sk,
            "active_hours": top_hours,
            "active_days":  top_days,
            "patterns":     patterns,
        }

    def get_hourly_distribution(self) -> dict[int, int]:
        """Distribusi aktivitas per jam (0–23). Berguna untuk visualisasi."""
        return dict(self._hour_counter)

    def get_recent_keywords(self, last_n: int = 50) -> Counter:
        """Keyword dari N aktivitas terakhir di buffer (sebelum flush)."""
        recent = self._buffer[-last_n:]
        c: Counter = Counter()
        for entry in recent:
            c.update(entry.keywords)
        return c

    async def flush(self) -> None:
        """Paksa flush buffer ke disk sekarang. Panggil saat shutdown."""
        await self._flush()

    # ── Pola Turunan ──────────────────────────────────────────────────────────

    def _derive_patterns(self, top_hours: list[tuple[int, int]]) -> dict:
        """
        Turunkan pola sederhana dari data mentah.
        Ini hipotesis kasar — belum dikonfirmasi ke Rofi.
        Konfirmasi adalah tugas curiosity.py.
        """
        patterns: dict = {}

        if not top_hours:
            return patterns

        # Jam paling aktif
        peak_hour = top_hours[0][0]
        patterns["peak_hour"] = peak_hour

        # Klasifikasi sesi berdasarkan jam
        if 5 <= peak_hour <= 9:
            patterns["session_hypothesis"] = "pagi"
            patterns["session_label"]      = "Rofi kemungkinan aktif di pagi hari"
        elif 10 <= peak_hour <= 13:
            patterns["session_hypothesis"] = "siang"
            patterns["session_label"]      = "Rofi kemungkinan aktif di siang hari"
        elif 14 <= peak_hour <= 17:
            patterns["session_hypothesis"] = "sore"
            patterns["session_label"]      = "Rofi kemungkinan aktif di sore hari"
        elif 18 <= peak_hour <= 22:
            patterns["session_hypothesis"] = "malam"
            patterns["session_label"]      = "Rofi kemungkinan aktif di malam hari"
        else:
            patterns["session_hypothesis"] = "larut"
            patterns["session_label"]      = "Rofi kemungkinan aktif larut malam"

        # Cek apakah ada kata yang konsisten muncul di jam tertentu
        # (ini data mentah — profiler yang mengolah lebih lanjut)
        patterns["top_keywords_raw"] = [
            w for w, _ in self._keyword_counter.most_common(10)
        ]

        # Apakah Rofi banyak pakai skill reminder? (indikasi sibuk / terjadwal)
        reminder_count = self._skill_counter.get("reminder", 0)
        if reminder_count > 3:
            patterns["uses_reminders"] = True
            patterns["reminder_count"] = reminder_count

        return patterns

    # ── Persistensi ───────────────────────────────────────────────────────────

    async def _flush(self) -> None:
        """Tulis buffer ke activity.log (append) dan simpan summary ke JSON."""
        if not self._buffer:
            return

        # Pastikan direktori ada
        ACTIVITY_LOG.parent.mkdir(parents=True, exist_ok=True)

        # Append ke log (satu JSON per baris)
        lines = [json.dumps(e.to_dict(), ensure_ascii=False) for e in self._buffer]
        try:
            with ACTIVITY_LOG.open("a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError as e:
            logger.error("[watcher] Gagal tulis activity.log: %s", e)

        # Simpan summary counter ke JSON (biar bisa dimuat ulang)
        self._save_summary()

        count = len(self._buffer)
        self._buffer.clear()
        logger.info("[watcher] Flushed %d entri ke disk.", count)

    def _save_summary(self) -> None:
        """Simpan counter ke activity_summary.json."""
        ACTIVITY_JSON.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total":    self._total_logged,
            "keywords": dict(self._keyword_counter),
            "skills":   dict(self._skill_counter),
            "hours":    {str(k): v for k, v in self._hour_counter.items()},
            "weekdays": {str(k): v for k, v in self._weekday_counter.items()},
            "updated":  datetime.now().isoformat(timespec="seconds"),
        }
        try:
            ACTIVITY_JSON.write_text(
                json.dumps(data, ensure_ascii=False, indent=2)
            )
        except OSError as e:
            logger.error("[watcher] Gagal simpan summary: %s", e)

    def _load_summary(self) -> None:
        """Muat counter dari activity_summary.json jika ada."""
        if not ACTIVITY_JSON.exists():
            return
        try:
            data = json.loads(ACTIVITY_JSON.read_text(encoding="utf-8"))
            self._total_logged    = data.get("total", 0)
            self._keyword_counter = Counter(data.get("keywords", {}))
            self._skill_counter   = Counter(data.get("skills", {}))
            self._hour_counter    = Counter({
                int(k): v for k, v in data.get("hours", {}).items()
            })
            self._weekday_counter = Counter({
                int(k): v for k, v in data.get("weekdays", {}).items()
            })
            logger.info(
                "[watcher] Summary dimuat: %d total, %d keyword unik",
                self._total_logged, len(self._keyword_counter),
            )
        except Exception as e:
            logger.warning("[watcher] Gagal load summary (akan mulai fresh): %s", e)


# ─────────────────────────── Singleton Helper ────────────────────────────────

_watcher_instance: Optional[ActivityWatcher] = None

def get_watcher() -> ActivityWatcher:
    """Ambil instance singleton. Raise jika belum diinit."""
    if _watcher_instance is None:
        raise RuntimeError(
            "ActivityWatcher belum diinisialisasi. Panggil init_watcher() dulu."
        )
    return _watcher_instance

def init_watcher() -> ActivityWatcher:
    """Buat dan simpan watcher singleton. Panggil sekali dari app.py."""
    global _watcher_instance
    _watcher_instance = ActivityWatcher()
    return _watcher_instance


# ─────────────────────────── Integrasi app.py ────────────────────────────────
#
# Tambahkan ini ke server/app.py:
#
#   from intelligence.activity_watcher import init_watcher
#
#   @app.on_event("startup")
#   async def startup():
#       ...
#       app.state.watcher = init_watcher()
#
#   # Di dalam handler WebSocket, setelah executor.dispatch():
#   result = await executor.dispatch(text, history=history)
#   await app.state.watcher.log(
#       text   = text,
#       intent = result.intent.value,
#       skill  = result.skill,
#   )
#
#   @app.on_event("shutdown")
#   async def shutdown():
#       await app.state.watcher.flush()


# ─────────────────────────── Quick Test ─────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # Override path ke /tmp untuk test lokal
    ACTIVITY_LOG  = Path("/tmp/otto_activity.log")
    ACTIVITY_JSON = Path("/tmp/otto_activity_summary.json")

    async def _test():
        watcher = ActivityWatcher()

        # Simulasi beberapa interaksi
        test_data = [
            ("ingatkan aku minum obat 30 menit lagi",  "command", "reminder"),
            ("putar lagu santai",                       "command", "play_santai"),
            ("cuaca hari ini gimana?",                  "chat",    ""),
            ("catat berat badanku 68 kg",               "command", "track_weight"),
            ("jam berapa sekarang?",                    "command", "waktu"),
            ("otto aktif?",                             "command", "status"),
            ("aku lagi ngerjain proposal kerjaan nih",  "chat",    ""),
            ("volume naik dong",                        "command", "volume_up"),
            ("ingatkan meeting jam 14:30",              "command", "reminder"),
            ("kayaknya aku mau tidur",                  "chat",    ""),
        ]

        for text, intent, skill in test_data:
            await watcher.log(text=text, intent=intent, skill=skill)

        # Paksa flush
        await watcher.flush()

        # Tampilkan summary
        summary = watcher.get_summary()
        print("\n=== ACTIVITY SUMMARY ===")
        print(f"Total logged : {summary['total']}")
        print(f"Top keywords : {summary['top_keywords'][:8]}")
        print(f"Top skills   : {summary['top_skills']}")
        print(f"Active hours : {summary['active_hours']}")
        print(f"Active days  : {summary['active_days']}")
        print(f"Patterns     : {json.dumps(summary['patterns'], indent=2, ensure_ascii=False)}")

    asyncio.run(_test())
