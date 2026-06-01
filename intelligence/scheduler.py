"""
intelligence/scheduler.py — Otak Waktu Otto
============================================
Scheduler menjalankan task-task background secara terjadwal.
Bukan cron biasa — scheduler ini sadar konteks:
  - Tidak jalankan task jika Rofi sedang aktif ngobrol
  - Tidak tanya jika di luar SAFE_HOURS
  - Prioritaskan task berdasarkan urgency

Arsitektur:
  ┌─────────────────────────────────────────────┐
  │              Scheduler (asyncio)             │
  │                                              │
  │  Task Loop:                                  │
  │    [tick]  setiap 60 detik                   │
  │      ├─ activity_watcher.flush()  (5 menit) │
  │      ├─ profiler.analyze()        (malam)   │  ← sekali sehari jam 22
  │      ├─ curiosity.try_ask()       (30 menit)│  ← hanya jika profiler sudah analyze
  │      └─ self_check()              (6 jam)   │
  └─────────────────────────────────────────────┘

Filosofi Otto:
  Amati seharian → profiler analyze malam → curiosity tanya keesokan hari
  Bukan asisten reaktif — Otto belajar dari ritme Rofi secara alami.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import Optional, Callable, Awaitable

logger = logging.getLogger("otto.intelligence.scheduler")


# ──────────────────────────── Konfigurasi ────────────────────────────────────

INTERVALS = {
    "activity_flush":   5  * 60,    # 5 menit  — flush log aktivitas ke disk
    "profile_analyze":  23 * 3600,  # 23 jam   — guard tambahan: hanya jam malam
    "curiosity_check":  30 * 60,    # 30 menit — guard: hanya jika profiler sudah jalan
    "self_check":       6  * 3600,  # 6 jam    — cek kesehatan sistem
    "growth_daily":     24 * 3600,  # 24 jam   — update riwayat pertumbuhan
}

TICK_SECONDS = 60

# Profiler hanya analyze mulai jam ini (malam hari)
NIGHTLY_ANALYZE_HOUR = 22

# Jeda setelah percakapan sebelum curiosity boleh tanya
POST_CONVERSATION_COOLDOWN = 5 * 60


# ──────────────────────────── Scheduled Task ─────────────────────────────────

class ScheduledTask:
    def __init__(
        self,
        name: str,
        interval: int,
        fn: Callable[[], Awaitable[None]],
        run_at_start: bool = False,
    ) -> None:
        self.name          = name
        self.interval      = interval
        self.fn            = fn
        self.run_at_start  = run_at_start
        self._last_run: Optional[datetime] = None
        self._run_count    = 0
        self._error_count  = 0

    def is_due(self, now: datetime) -> bool:
        if self._last_run is None:
            return self.run_at_start
        return (now - self._last_run).total_seconds() >= self.interval

    async def run(self) -> bool:
        try:
            await self.fn()
            self._last_run  = datetime.now()
            self._run_count += 1
            logger.debug("[scheduler] ✓ %s selesai (total: %d)", self.name, self._run_count)
            return True
        except Exception as e:
            self._error_count += 1
            logger.error(
                "[scheduler] ✗ %s error (ke-%d): %s",
                self.name, self._error_count, e, exc_info=True,
            )
            self._last_run = datetime.now()
            return False

    @property
    def stats(self) -> dict:
        return {
            "name":         self.name,
            "interval_min": self.interval // 60,
            "last_run":     self._last_run.isoformat() if self._last_run else None,
            "run_count":    self._run_count,
            "error_count":  self._error_count,
        }


# ──────────────────────────── Scheduler ──────────────────────────────────────

class Scheduler:
    def __init__(self, activity_watcher, profiler, curiosity, memory=None) -> None:
        self._activity_watcher = activity_watcher
        self._profiler         = profiler
        self._curiosity        = curiosity
        self._memory           = memory  

        self._running          = False
        self._loop_task: Optional[asyncio.Task] = None
        self._tasks: list[ScheduledTask] = []

        self._last_conversation_at: Optional[datetime] = None
        self._on_question_cb: Optional[Callable] = None

        # Tracking analyze harian
        self._last_analyze_date: Optional[str] = None
        self._profiler_analyzed_today: bool = False

        self._register_tasks()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _register_tasks(self) -> None:
        self._tasks = [
            ScheduledTask(
                name         = "activity_flush",
                interval     = INTERVALS["activity_flush"],
                fn           = self._task_activity_flush,
                run_at_start = True,
            ),
            ScheduledTask(
                name         = "profile_analyze",
                interval     = INTERVALS["profile_analyze"],
                fn           = self._task_profile_analyze,
                run_at_start = False,
            ),
            ScheduledTask(
                name         = "curiosity_check",
                interval     = INTERVALS["curiosity_check"],
                fn           = self._task_curiosity_check,
                run_at_start = False,
            ),
            ScheduledTask(
                name         = "self_check",
                interval     = INTERVALS["self_check"],
                fn           = self._task_self_check,
                run_at_start = False,
            ),
            ScheduledTask(
                name         = "growth_daily",
                interval     = INTERVALS["growth_daily"],
                fn           = self._task_growth_daily,
                run_at_start = False,
            ),
        ]
        logger.info("[scheduler] %d task terdaftar.", len(self._tasks))

    def set_question_callback(self, cb: Callable) -> None:
        self._on_question_cb = cb

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            logger.warning("[scheduler] Sudah berjalan, skip start.")
            return
        self._running   = True
        self._loop_task = asyncio.create_task(
            self._main_loop(), name="otto.scheduler"
        )
        logger.info("[scheduler] Dimulai. Tick setiap %d detik.", TICK_SECONDS)

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("[scheduler] Dihentikan.")

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Main Loop ─────────────────────────────────────────────────────────────

    async def _main_loop(self) -> None:
        logger.info("[scheduler] Loop mulai.")
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.error("[scheduler] Error di tick: %s", e, exc_info=True)
            await asyncio.sleep(TICK_SECONDS)
        logger.info("[scheduler] Loop selesai.")

    async def _tick(self) -> None:
        now       = datetime.now()
        today_str = date.today().isoformat()

        # Reset flag jika hari sudah berganti
        if self._last_analyze_date and self._last_analyze_date != today_str:
            self._profiler_analyzed_today = False
            logger.debug("[scheduler] Hari baru — reset profiler_analyzed_today.")

        for task in self._tasks:
            if not self._running:
                break

            if not task.is_due(now):
                continue

            # Curiosity tidak jalan saat cooldown
            if task.name == "curiosity_check" and self._in_cooldown():
                logger.debug("[scheduler] curiosity_check skip — cooldown aktif.")
                continue

            logger.debug("[scheduler] Jalankan task: %s", task.name)
            await task.run()

    # ── Konteks Awareness ─────────────────────────────────────────────────────

    def notify_conversation_active(self) -> None:
        self._last_conversation_at = datetime.now()
        logger.debug("[scheduler] Rofi aktif ngobrol — curiosity_check ditunda.")

    def _in_cooldown(self) -> bool:
        if self._last_conversation_at is None:
            return False
        elapsed = (datetime.now() - self._last_conversation_at).total_seconds()
        return elapsed < POST_CONVERSATION_COOLDOWN

    # ── Task: Activity Flush ──────────────────────────────────────────────────

    async def _task_activity_flush(self) -> None:
        """Flush buffer aktivitas ke disk."""
        if hasattr(self._activity_watcher, "flush"):
            await self._activity_watcher.flush()
        elif hasattr(self._activity_watcher, "save"):
            self._activity_watcher.save()
        logger.debug("[scheduler] activity_flush selesai.")

    # ── Task: Profile Analyze (MALAM HARI) ───────────────────────────────────

    async def _task_profile_analyze(self) -> None:
        """
        Profiler analyze — hanya jalan malam hari (jam >= NIGHTLY_ANALYZE_HOUR)
        dan hanya sekali per hari.

        Filosofi: Otto amati seharian penuh, baru simpulkan malam.
        Keesokan harinya curiosity bisa tanya berdasarkan pola kemarin.
        """
        now       = datetime.now()
        today_str = date.today().isoformat()

        # Guard 1: hanya jam malam
        if now.hour < NIGHTLY_ANALYZE_HOUR:
            logger.debug(
                "[scheduler] profile_analyze skip — belum jam %02d:00 (sekarang %02d:%02d).",
                NIGHTLY_ANALYZE_HOUR, now.hour, now.minute,
            )
            return

        # Guard 2: sudah jalan hari ini?
        if self._last_analyze_date == today_str:
            logger.debug("[scheduler] profile_analyze skip — sudah jalan hari ini (%s).", today_str)
            return

        # Jalankan analyze
        if not hasattr(self._profiler, "analyze"):
            return

        if asyncio.iscoroutinefunction(self._profiler.analyze):
            result = await self._profiler.analyze()
        else:
            result = self._profiler.analyze()

        new_count = len(result) if isinstance(result, list) else 0

        self._last_analyze_date       = today_str
        self._profiler_analyzed_today = True

        logger.info(
            "[scheduler] Profiler malam (%s): %d hipotesis baru ditemukan.",
            today_str, new_count,
        )

    # ── Task: Curiosity Check ─────────────────────────────────────────────────

    async def _task_curiosity_check(self) -> None:
        """
        Curiosity kirim pertanyaan ke Rofi.

        Guard: hanya jalan jika profiler sudah analyze hari ini.
        Artinya: curiosity hanya punya bahan segar setelah malam analyze.
        """
        if not self._profiler_analyzed_today:
            logger.debug(
                "[scheduler] curiosity_check skip — profiler belum analyze hari ini."
            )
            return

        question, hyp_id = await self._curiosity.try_ask()

        if question and hyp_id:
            logger.info("[scheduler] Curiosity siap tanya hipotesis %s.", hyp_id)
            if self._on_question_cb:
                await self._on_question_cb(question, hyp_id)
            else:
                logger.warning(
                    "[scheduler] Tidak ada question_callback. "
                    "Set via set_question_callback(). Pertanyaan: '%s'", question,
                )

    # ── Task: Self Check ──────────────────────────────────────────────────────

    async def _task_self_check(self) -> None:
        stats = self.get_stats()
        logger.info(
            "[scheduler] Self-check: %d task, total run: %d, error: %d",
            stats["task_count"],
            stats["total_runs"],
            stats["total_errors"],
        )

    # ── Task: Growth Daily ────────────────────────────────────────────────────

    async def _task_growth_daily(self) -> None:
        """Update riwayat pertumbuhan harian."""
        from intelligence.growth_tracker import get_tracker
        try:
            tracker = get_tracker()
            summary = tracker.daily_update(
                memory = self._memory,
                profiler = self._profiler,
            )
            logger.info(
                "[scheduler] Growth daily: +%d hari ini | kumulatif: %d | %s",
                summary.get("score_today", 0),
                summary.get("cumulative", 0),
                summary.get("milestone", ("?",))[0],
            )
        except Exception as e:
            logger.error("[scheduler] growth_daily error: %s", e)

    # ── Utilitas ──────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "running":        self._running,
            "task_count":     len(self._tasks),
            "total_runs":     sum(t._run_count  for t in self._tasks),
            "total_errors":   sum(t._error_count for t in self._tasks),
            "profiler_analyzed_today": self._profiler_analyzed_today,
            "last_analyze_date":       self._last_analyze_date,
            "tasks":          [t.stats for t in self._tasks],
        }

    def force_run(self, task_name: str) -> bool:
        for task in self._tasks:
            if task.name == task_name:
                asyncio.create_task(task.run(), name=f"otto.scheduler.force.{task_name}")
                logger.info("[scheduler] Force run: %s", task_name)
                return True
        logger.warning("[scheduler] Task tidak ditemukan: %s", task_name)
        return False

    def __repr__(self) -> str:
        status = "running" if self._running else "stopped"
        return f"<Scheduler {status}, {len(self._tasks)} tasks>"


# ──────────────────────────── Singleton Helper ───────────────────────────────

_scheduler_instance: Optional[Scheduler] = None


def get_scheduler() -> Scheduler:
    if _scheduler_instance is None:
        raise RuntimeError(
            "Scheduler belum diinisialisasi. "
            "Panggil init_scheduler(activity_watcher, profiler, curiosity) dulu."
        )
    return _scheduler_instance

def init_scheduler(activity_watcher, profiler, curiosity, memory=None) -> Scheduler:
    global _scheduler_instance
    _scheduler_instance = Scheduler(activity_watcher, profiler, curiosity, memory=memory)
    return _scheduler_instance


# ──────────────────────────── Quick Test ─────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s  %(message)s")

    class MockWatcher:
        async def flush(self): print("  [watcher] flush()")

    class MockProfiler:
        async def analyze(self): print("  [profiler] analyze()"); return []

    class MockCuriosity:
        async def try_ask(self):
            print("  [curiosity] try_ask()")
            return "Rofi, kamu biasa aktif jam berapa?", "hyp_test_001"

    async def _test():
        scheduler = Scheduler(MockWatcher(), MockProfiler(), MockCuriosity())

        async def on_q(q, hid): print(f"\n  >>> Pertanyaan: {q} (id: {hid})\n")
        scheduler.set_question_callback(on_q)

        # Force profiler_analyzed_today = True untuk test curiosity
        scheduler._profiler_analyzed_today = True

        for t in scheduler._tasks:
            t.interval = 2
            t.run_at_start = True

        print("=== START ===")
        await scheduler.start()
        await asyncio.sleep(6)

        print("\n=== STATS ===")
        for t in scheduler.get_stats()["tasks"]:
            print(f"  {t['name']:20} run={t['run_count']} err={t['error_count']}")

        await scheduler.stop()

    asyncio.run(_test())
