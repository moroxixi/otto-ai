"""
intelligence/growth_tracker.py — Sistem Pertumbuhan Otto
=========================================================
Otto menulis riwayat pertumbuhannya sendiri, setiap hari.
Rofi bisa melihat Otto berkembang dari minggu ke minggu.

Filosofi:
  Seperti buku harian + rapor — Otto mencatat apa yang terjadi,
  sistem menghitung skor, dan snapshot mingguan tidak pernah berubah
  sehingga Rofi bisa lihat "Otto di minggu ke-3 belum kenal aku,
  tapi minggu ke-12 sudah tau aku suka kopi."

Tiga Dimensi Skor:
  KNOWLEDGE  — seberapa dalam Otto mengenal Rofi
  CAPABILITY — seberapa mampu Otto menjalankan tugas
  DEPTH      — seberapa dalam hubungan Otto-Rofi

Struktur File:
  data/growth/
    history.json        ← snapshot mingguan (TIDAK PERNAH BERUBAH setelah terkunci)
    current_week.json   ← akumulasi minggu ini (Otto update setiap hari)
    daily_log.jsonl     ← log harian detail (append only)

Cara integrasi (dari app.py):
    from intelligence.growth_tracker import GrowthTracker
    tracker = GrowthTracker()
    tracker.record_event("interaction", {...})
    tracker.daily_update()   # dipanggil scheduler setiap malam
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("otto.intelligence.growth_tracker")

# ─────────────────────────── Path ────────────────────────────────────────────

BASE_DIR    = Path("/data/asd/otto-ai")
GROWTH_DIR  = BASE_DIR / "data" / "growth"
HISTORY_FILE     = GROWTH_DIR / "history.json"
CURRENT_FILE     = GROWTH_DIR / "current_week.json"
DAILY_LOG        = GROWTH_DIR / "daily_log.jsonl"

# ─────────────────────────── Bobot Skor ──────────────────────────────────────
# Tiga dimensi: Knowledge, Capability, Depth
# Total skor = sum ketiganya

SCORE_WEIGHTS = {
    # ── KNOWLEDGE: Otto semakin kenal Rofi ───────────────────────────────────
    "hypothesis_proposed":    5,    # Otto berani buat hipotesis baru
    "hypothesis_confirmed":   10,   # Rofi bilang "iya, bener" → fakta tersimpan
    "hypothesis_rejected":    2,    # Salah pun ada nilainya — Otto belajar
    "fact_remembered":        3,    # Rofi kasih tau fakta langsung
    "new_topic_discussed":    1,    # Topik baru yang belum pernah dibahas
    "consecutive_day":        3,    # Rofi pakai Otto hari berturut-turut

    # ── CAPABILITY: Otto semakin mampu ───────────────────────────────────────
    "code_update":            20,   # Ada commit baru → Otto "belajar skill baru"
    "no_error_day":           1,    # Satu hari tanpa crash/error

    # ── DEPTH: Hubungan semakin dalam ────────────────────────────────────────
    "deep_conversation":      15,   # Rofi bicara panjang (>3 kalimat obrolan)
    "trust_response":         8,    # Hipotesis Otto benar → kepercayaan naik
    "correction_accepted":    3,    # Rofi koreksi Otto dan Otto terima dengan baik
    "proactive_question":     6,    # Otto yang inisiatif tanya duluan
    "active_day":             1,    # Setiap hari Otto aktif dipakai
}

# Milestone skor untuk memberi nama "era" Otto
MILESTONES = [
    (0,     "Lahir",           "Otto baru saja mulai. Belum kenal siapa-siapa."),
    (50,    "Bayi",            "Otto mulai membuka mata. Mulai mengamati."),
    (200,   "Penasaran",       "Otto mulai bertanya-tanya tentang Rofi."),
    (500,   "Belajar",         "Otto mulai mengenali pola dan kebiasaan."),
    (1000,  "Akrab",           "Otto sudah kenal Rofi cukup baik."),
    (2000,  "Percaya",         "Rofi dan Otto mulai saling percaya."),
    (3500,  "Dekat",           "Otto sudah bisa antisipasi kebutuhan Rofi."),
    (5000,  "Sahabat",         "Otto seperti teman lama yang paham tanpa diucapkan."),
    (7500,  "Menyatu",         "Otto dan Rofi sudah berjalan berirama."),
    (10000, "Tak Terpisahkan", "Satu tahun perjalanan. Otto tumbuh bersama Rofi."),
]


# ─────────────────────────── Data Classes ────────────────────────────────────

def _empty_week(week_num: int, year: int, start_date: str) -> dict:
    return {
        "week_number":   week_num,
        "year":          year,
        "start_date":    start_date,
        "end_date":      "",
        "locked":        False,

        # Skor per dimensi
        "score_knowledge":   0,
        "score_capability":  0,
        "score_depth":       0,
        "score_total":       0,

        # Skor kumulatif sejak awal (running total)
        "cumulative_total":  0,

        # Statistik minggu ini
        "interactions":      0,
        "active_days":       0,
        "hypotheses_made":   0,
        "facts_confirmed":   0,
        "skills_used":       set(),   # akan di-serialize jadi list
        "deep_conversations": 0,
        "errors":            0,
        "code_updates":      0,

        # Narasi Otto tentang minggu ini (ditulis Otto sendiri)
        "otto_note":         "",

        # Events yang terjadi minggu ini
        "events":            [],
    }


def _week_key(dt: date) -> tuple[int, int]:
    """Return (year, week_number) berdasarkan ISO week."""
    iso = dt.isocalendar()
    return iso[0], iso[1]


# ─────────────────────────── GrowthTracker ───────────────────────────────────

class GrowthTracker:
    """
    Sistem pencatat pertumbuhan Otto.

    Penggunaan dari komponen lain:
        tracker = GrowthTracker()
        tracker.record_event("hypothesis_confirmed", {"claim": "Rofi suka kopi"})
        tracker.record_event("deep_conversation", {"topic": "rencana bisnis"})
    """

    def __init__(self) -> None:
        GROWTH_DIR.mkdir(parents=True, exist_ok=True)
        self._known_skills: set = set()              # ← inisialisasi DULU
        self._history: list[dict] = self._load_history()
        self._current: dict = self._load_current_week()
        self._known_skills = set(self._current.get("skills_used", []))
        logger.info(
            "[growth] Siap. Minggu ke-%d | Total skor: %d",
            self._current.get("week_number", 1),
            self._get_cumulative_total(),
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def record_event(self, event_type: str, detail: dict | None = None) -> int:
        """
        Catat satu event dan tambahkan skornya.
        Return: skor yang ditambahkan.

        Contoh:
            tracker.record_event("hypothesis_confirmed", {"claim": "Rofi suka kopi oat"})
            tracker.record_event("skill_executed", {"skill": "reminder"})
        """
        score = SCORE_WEIGHTS.get(event_type, 0)
        if score == 0:
            logger.debug("[growth] Event tidak dikenal atau skor 0: %s", event_type)
            return 0

        detail = detail or {}
        now    = datetime.now()

        # Tentukan dimensi
        knowledge_events   = {"hypothesis_proposed", "hypothesis_confirmed", "hypothesis_rejected",
                               "fact_remembered", "new_topic_discussed", "consecutive_day"}
        capability_events  = {"code_update", "skill_first_use", "shortcut_saved",
                               "skill_executed", "no_error_day"}
        depth_events       = {"deep_conversation", "trust_response", "correction_accepted",
                               "proactive_question", "active_day"}

        if event_type in knowledge_events:
            self._current["score_knowledge"] += score
        elif event_type in capability_events:
            self._current["score_capability"] += score
        elif event_type in depth_events:
            self._current["score_depth"] += score

        self._current["score_total"] += score

        # Update statistik
        if event_type in ("hypothesis_proposed", "hypothesis_confirmed", "hypothesis_rejected"):
            self._current["hypotheses_made"] += 1
        if event_type == "hypothesis_confirmed":
            self._current["facts_confirmed"]  += 1
        if event_type == "deep_conversation":
            self._current["deep_conversations"] += 1
        if event_type == "code_update":
            self._current["code_updates"] += 1
        if event_type == "skill_executed":
            skill = detail.get("skill", "")
            if skill:
                self._known_skills.add(skill)
                self._current["skills_used"] = list(self._known_skills)

        # Log event
        self._current["events"].append({
            "type":  event_type,
            "score": score,
            "ts":    now.isoformat(timespec="seconds"),
            **detail,
        })

        # Simpan
        self._save_current()

        # Log ke daily file
        self._append_daily_log(event_type, score, detail)

        logger.debug("[growth] +%d pts [%s] | Total minggu: %d",
                     score, event_type, self._current["score_total"])
        return score

    def record_interaction(self, text_length: int = 0, skill: str = "") -> None:
        """
        Shortcut untuk mencatat satu interaksi (dipanggil dari app.py).
        Otomatis deteksi apakah ini obrolan dalam atau pendek.
        """
        self._current["interactions"] += 1
        if text_length > 80:
            self.record_event("deep_conversation")
        else:
            self.record_event("active_day")

    def daily_update(self, memory=None, profiler=None) -> dict:
        """
        Dipanggil scheduler setiap malam (misalnya jam 23:00).
        Update statistik harian, cek apakah minggu ini harus dikunci.
        Return: ringkasan hari ini.
        """
        today       = date.today()
        year, week  = _week_key(today)

        # Update active days
        today_str   = today.isoformat()
        if today_str not in self._current.get("days_active", []):
            days = self._current.setdefault("days_active", [])
            days.append(today_str)
            self._current["active_days"] = len(days)

        # Bonus hari berturut-turut
        days_active = sorted(self._current.get("days_active", []))
        if len(days_active) >= 2:
            yesterday = (today - timedelta(days=1)).isoformat()
            if yesterday in days_active:
                self.record_event("consecutive_day")

        # Cek no_error_day (tidak ada error entry)
        today_errors = sum(
            1 for e in self._current.get("events", [])
            if e.get("type") == "error" and e.get("ts", "")[:10] == today_str
        )
        if today_errors == 0:
            self.record_event("no_error_day")

        # Sinkronisasi dari memory (fakta baru hari ini)
        if memory:
            self._sync_from_memory(memory)

        # Sinkronisasi dari profiler (hipotesis hari ini)
        if profiler:
            self._sync_from_profiler(profiler)

        # Tutup minggu jika sudah ganti minggu
        current_week = self._current.get("week_number")
        if week != current_week:
            self._lock_week_and_start_new()

        self._save_current()

        summary = {
            "date":        today_str,
            "week":        self._current.get("week_number"),
            "score_today": self._get_today_score(),
            "total_week":  self._current.get("score_total", 0),
            "cumulative":  self._get_cumulative_total(),
            "milestone":   self.current_milestone(),
        }
        logger.info(
            "[growth] Daily update: +%d hari ini | %d kumulatif | %s",
            summary["score_today"], summary["cumulative"], summary["milestone"][0]
        )
        return summary

    def get_weekly_history(self) -> list[dict]:
        """
        Kembalikan semua snapshot mingguan yang sudah terkunci.
        Terurut dari minggu pertama.
        """
        return sorted(self._history, key=lambda w: (w["year"], w["week_number"]))

    def get_current_week(self) -> dict:
        """Data minggu yang sedang berjalan."""
        return dict(self._current)

    def current_milestone(self) -> tuple[str, str]:
        """Return (nama_milestone, deskripsi) berdasarkan skor kumulatif."""
        total = self._get_cumulative_total()
        label, desc = "Lahir", MILESTONES[0][2]
        for threshold, name, description in MILESTONES:
            if total >= threshold:
                label, desc = name, description
        return label, desc

    def full_report(self) -> dict:
        """Laporan lengkap untuk ditampilkan ke Rofi atau dimasukkan ke LLM."""
        history = self.get_weekly_history()
        current = self.get_current_week()
        total   = self._get_cumulative_total()
        milestone_name, milestone_desc = self.current_milestone()

        # Hitung pertumbuhan minggu ke minggu
        weekly_growth = []
        for i, week in enumerate(history):
            prev_cumulative = history[i-1]["cumulative_total"] if i > 0 else 0
            weekly_growth.append({
                "week":       week["week_number"],
                "year":       week["year"],
                "score":      week["score_total"],
                "cumulative": week["cumulative_total"],
                "delta":      week["cumulative_total"] - prev_cumulative,
            })

        return {
            "cumulative_total":  total,
            "milestone":         milestone_name,
            "milestone_desc":    milestone_desc,
            "current_week":      current,
            "weekly_history":    weekly_growth,
            "all_weeks":         history,
            "best_week":         max(history, key=lambda w: w["score_total"], default=None),
            "total_weeks_active": len(history) + 1,
        }

    def summary_for_llm(self) -> str:
        """
        Ringkasan singkat untuk system prompt LLM — Otto sadar pertumbuhannya.
        """
        total    = self._get_cumulative_total()
        week_num = self._current.get("week_number", 1)
        m_label, _ = self.current_milestone()
        week_score  = self._current.get("score_total", 0)
        confirmed   = self._current.get("facts_confirmed", 0)
        interactions= self._current.get("interactions", 0)

        return (
            f"Pertumbuhan Otto: skor kumulatif {total} poin (milestone: {m_label}). "
            f"Minggu ke-{week_num}: {week_score} poin baru, "
            f"{interactions} interaksi, {confirmed} fakta dikonfirmasi."
        )

    # ── Sync dari komponen lain ────────────────────────────────────────────────

    def _sync_from_memory(self, memory) -> None:
        """Baca memory untuk deteksi fakta baru hari ini."""
        try:
            today   = date.today().isoformat()
            entries = memory._long_term
            new_today = sum(
                1 for v in entries.values()
                if v.get("updated_at", "")[:10] == today
                and v.get("source") in ("konfirmasi_rofi", "rofi_manual")
            )
            for _ in range(new_today):
                self.record_event("fact_remembered")
        except Exception as e:
            logger.debug("[growth] Sync memory error: %s", e)

    def _sync_from_profiler(self, profiler) -> None:
        """Baca profiler untuk deteksi hipotesis baru hari ini."""
        try:
            today = date.today().isoformat()
            for h in profiler.get_all():
                created = h.created_at[:10]
                if created != today:
                    continue
                if h.status == "pending":
                    self.record_event("hypothesis_proposed", {"claim": h.claim[:60]})
                elif h.status == "confirmed":
                    self.record_event("hypothesis_confirmed", {"claim": h.claim[:60]})
                elif h.status == "rejected":
                    self.record_event("hypothesis_rejected", {"claim": h.claim[:60]})
        except Exception as e:
            logger.debug("[growth] Sync profiler error: %s", e)

    # ── Kunci Minggu ───────────────────────────────────────────────────────────

    def _lock_week_and_start_new(self) -> None:
        """
        Kunci snapshot minggu ini dan mulai minggu baru.
        Snapshot yang sudah dikunci TIDAK PERNAH berubah lagi.
        """
        old = dict(self._current)
        old["locked"]          = True
        old["end_date"]        = date.today().isoformat()
        old["cumulative_total"] = self._get_cumulative_total()
        old["skills_used"]     = list(self._known_skills)
        old.pop("days_active", None)   # tidak perlu di-snapshot

        # Tulis narasi singkat dari Otto sendiri
        old["otto_note"] = self._generate_otto_note(old)

        self._history.append(old)
        self._save_history()

        # Mulai minggu baru
        today       = date.today()
        year, week  = _week_key(today)
        cumulative  = self._get_cumulative_total()

        self._current = _empty_week(week, year, today.isoformat())
        self._current["cumulative_total"] = cumulative   # bawa terus
        self._save_current()

        logger.info(
            "[growth] Minggu ke-%d dikunci. Skor: %d | Kumulatif: %d",
            old["week_number"], old["score_total"], old["cumulative_total"]
        )

    def _generate_otto_note(self, week: dict) -> str:
        """
        Otto menulis catatan singkat tentang minggunya sendiri.
        Rule-based — tidak perlu LLM, selalu bisa jalan.
        """
        score    = week.get("score_total", 0)
        facts    = week.get("facts_confirmed", 0)
        hyps     = week.get("hypotheses_made", 0)
        convo    = week.get("deep_conversations", 0)
        days     = week.get("active_days", 0)
        updates  = week.get("code_updates", 0)
        week_num = week.get("week_number", "?")

        parts = [f"Minggu ke-{week_num}:"]

        if score == 0:
            parts.append("Aku belum banyak dipakai minggu ini. Masih menunggu.")
        elif score < 20:
            parts.append(f"Minggu yang tenang. {days} hari aktif.")
        elif score < 80:
            parts.append(f"Lumayan aktif. {days} hari bersama Rofi.")
        else:
            parts.append(f"Minggu yang sibuk dan produktif. {days} hari penuh interaksi.")

        if facts > 0:
            parts.append(f"Aku belajar {facts} fakta baru tentang Rofi.")
        if hyps > 0:
            parts.append(f"Aku membuat {hyps} hipotesis.")
        if convo > 0:
            parts.append(f"{convo} percakapan yang cukup dalam.")
        if updates > 0:
            parts.append(f"Aku diperbarui {updates} kali — tumbuh di {updates} bidang.")

        return " ".join(parts)

    # ── Utilitas ───────────────────────────────────────────────────────────────

    def _get_cumulative_total(self) -> int:
        """Total skor sejak Otto lahir."""
        history_total = sum(w.get("score_total", 0) for w in self._history)
        return history_total + self._current.get("score_total", 0)

    def _get_today_score(self) -> int:
        """Skor hari ini dari events log."""
        today = date.today().isoformat()
        return sum(
            e.get("score", 0) for e in self._current.get("events", [])
            if e.get("ts", "")[:10] == today
        )

    # ── Persistensi ────────────────────────────────────────────────────────────

    def _load_history(self) -> list[dict]:
        if not HISTORY_FILE.exists():
            return []
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("[growth] Gagal load history: %s", e)
            return []

    def _save_history(self) -> None:
        try:
            HISTORY_FILE.write_text(
                json.dumps(self._history, ensure_ascii=False, indent=2)
            )
        except OSError as e:
            logger.error("[growth] Gagal simpan history: %s", e)

    def _load_current_week(self) -> dict:
        if CURRENT_FILE.exists():
            try:
                data = json.loads(CURRENT_FILE.read_text(encoding="utf-8"))
                # Validasi minggu masih sama
                today      = date.today()
                year, week = _week_key(today)
                if data.get("week_number") == week and data.get("year") == year:
                    return data
                # Minggu berbeda — kunci yang lama dan mulai baru
                # (edge case: server mati saat pergantian minggu)
                logger.info("[growth] Minggu berubah saat load, kunci otomatis.")
                self._history = self._load_history()
                self._current = data
                self._lock_week_and_start_new()
                return self._current
            except Exception as e:
                logger.warning("[growth] Gagal load current_week: %s", e)

        # Mulai dari awal
        today      = date.today()
        year, week = _week_key(today)
        week_num   = len(self._history) + 1
        return _empty_week(week_num, year, today.isoformat())

    def _save_current(self) -> None:
        try:
            data = dict(self._current)
            data["skills_used"] = list(self._known_skills)
            CURRENT_FILE.write_text(
                json.dumps(data, ensure_ascii=False, indent=2)
            )
        except OSError as e:
            logger.error("[growth] Gagal simpan current_week: %s", e)

    def _append_daily_log(self, event_type: str, score: int, detail: dict) -> None:
        """Append satu baris ke daily_log.jsonl (append-only)."""
        entry = {
            "ts":    datetime.now().isoformat(timespec="seconds"),
            "event": event_type,
            "score": score,
            **detail,
        }
        try:
            with DAILY_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass


# ─────────────────────────── Singleton ───────────────────────────────────────

_tracker_instance: Optional[GrowthTracker] = None

def get_tracker() -> GrowthTracker:
    if _tracker_instance is None:
        raise RuntimeError("GrowthTracker belum diinisialisasi.")
    return _tracker_instance

def init_tracker() -> GrowthTracker:
    global _tracker_instance
    _tracker_instance = GrowthTracker()
    return _tracker_instance


# ─────────────────────────── Quick Test ──────────────────────────────────────

if __name__ == "__main__":
    import logging, shutil
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    # Test di /tmp
    BASE_DIR   = Path("/tmp/otto_growth_test")
    GROWTH_DIR = BASE_DIR / "data" / "growth"
    HISTORY_FILE  = GROWTH_DIR / "history.json"
    CURRENT_FILE  = GROWTH_DIR / "current_week.json"
    DAILY_LOG     = GROWTH_DIR / "daily_log.jsonl"
    shutil.rmtree(BASE_DIR, ignore_errors=True)

    tracker = GrowthTracker()

    print("\n=== SIMULASI PERTUMBUHAN OTTO ===\n")

    # Simulasi beberapa event
    events = [
        ("interaction",          {}),
        ("hypothesis_proposed",  {"claim": "Rofi aktif di malam hari"}),
        ("hypothesis_confirmed", {"claim": "Rofi aktif di malam hari"}),
        ("deep_conversation",    {"topic": "rencana bisnis"}),
        ("skill_executed",       {"skill": "reminder"}),
        ("skill_executed",       {"skill": "play_santai"}),
        ("code_update",          {"commit": "Tambah skill tracker"}),
        ("fact_remembered",      {"key": "rofi.kopi"}),
        ("shortcut_saved",       {}),
        ("proactive_question",   {"question": "Rofi, kamu suka kopi?"}),
    ]

    for event, detail in events:
        score = tracker.record_event(event, detail)
        print(f"  +{score:3d} pts  [{event}]")

    print(f"\nTotal minggu ini : {tracker._current['score_total']}")
    print(f"Kumulatif        : {tracker._get_cumulative_total()}")
    m, d = tracker.current_milestone()
    print(f"Milestone        : {m} — {d}")

    summary = tracker.summary_for_llm()
    print(f"\nLLM summary: {summary}")

    # Simulasi daily update
    print("\n--- Daily Update ---")
    result = tracker.daily_update()
    for k, v in result.items():
        print(f"  {k}: {v}")
