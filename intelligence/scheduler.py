"""
intelligence/scheduler.py — Otak Waktu Otto
============================================
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import Optional, Callable, Awaitable

logger = logging.getLogger("otto.intelligence.scheduler")


# ──────────────────────────── Konfigurasi ────────────────────────────────────

INTERVALS = {
    "activity_flush":        5  * 60,
    "profile_analyze":       23 * 3600,
    "curiosity_check":       30 * 60,
    "self_check":            6  * 3600,
    "growth_daily":          24 * 3600,
    "context_trigger_check": 60,
}

TICK_SECONDS = 60
NIGHTLY_ANALYZE_HOUR = 12
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
    # FIX BUG 1: tambah context_engine dan speaker ke signature __init__
    # Versi sebelumnya: def __init__(self, ..., memory=None)
    # tapi body langsung assign self._context_engine = context_engine → NameError
    def __init__(self, activity_watcher, profiler, curiosity, memory=None,
                 context_engine=None, speaker=None) -> None:
        self._activity_watcher = activity_watcher
        self._profiler         = profiler
        self._curiosity        = curiosity
        self._memory           = memory
        self._context_engine   = context_engine
        self._speaker          = speaker

        self._running          = False
        self._loop_task: Optional[asyncio.Task] = None
        self._tasks: list[ScheduledTask] = []

        self._last_conversation_at: Optional[datetime] = None
        self._on_question_cb: Optional[Callable] = None
        self._background_tasks: set[asyncio.Task] = set()

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
            ScheduledTask(
                name         = "context_trigger_check",
                interval     = INTERVALS["context_trigger_check"],
                fn           = self._task_context_trigger_check,
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
        for t in list(self._background_tasks):
            t.cancel()
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

        if self._last_analyze_date and self._last_analyze_date != today_str:
            self._profiler_analyzed_today = False
            logger.debug("[scheduler] Hari baru — reset profiler_analyzed_today.")

        for task in self._tasks:
            if not self._running:
                break
            if not task.is_due(now):
                continue
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
        if hasattr(self._activity_watcher, "flush"):
            await self._activity_watcher.flush()
        elif hasattr(self._activity_watcher, "save"):
            self._activity_watcher.save()
        logger.debug("[scheduler] activity_flush selesai.")

    # ── Task: Profile Analyze ─────────────────────────────────────────────────

    async def _task_profile_analyze(self) -> None:
        now       = datetime.now()
        today_str = date.today().isoformat()

        if now.hour < NIGHTLY_ANALYZE_HOUR:
            logger.debug(
                "[scheduler] profile_analyze skip — belum jam %02d:00 (sekarang %02d:%02d).",
                NIGHTLY_ANALYZE_HOUR, now.hour, now.minute,
            )
            return

        if self._last_analyze_date == today_str:
            logger.debug("[scheduler] profile_analyze skip — sudah jalan hari ini (%s).", today_str)
            return

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
        has_pending = len(self._profiler.get_pending()) > 0
        if not self._profiler_analyzed_today and not has_pending:
            logger.debug("skip — belum ada hipotesis sama sekali.")
            return

        question, hyp_id = await self._curiosity.try_ask()
        if question and hyp_id:
            self._profiler.increment_asked(hyp_id)
            logger.info("[scheduler] Curiosity siap tanya hipotesis %s.", hyp_id)
            if self._on_question_cb:
                await self._on_question_cb(question, hyp_id)
            else:
                logger.warning(
                    "[scheduler] Tidak ada question_callback. Pertanyaan: '%s'", question,
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
        from intelligence.growth_tracker import get_tracker
        try:
            tracker = get_tracker()
            summary = tracker.daily_update(
                memory   = self._memory,
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

    # ── Task: Context Trigger Check ───────────────────────────────────────────

    async def _task_context_trigger_check(self) -> None:
        """
        Polling context triggers yang sudah due → ucapkan lewat speaker laptop.
        Jalan setiap 60 detik — tidak butuh HP terhubung.

        Filosofi: trigger dibuat saat Rofi ngobrol, tapi disampaikan
        secara mandiri oleh Otto — seperti asisten yang ingat sendiri.
        """
        if self._context_engine is None or self._speaker is None:
            return

        due = self._context_engine.get_due_triggers()
        if not due:
            return

        # Ambil satu per tick — jangan spam kalau ada banyak sekaligus
        trigger = due[0]
        self._context_engine.mark_done(trigger.id)

        msg = trigger.followup_message
        logger.info(
            "[scheduler] Context trigger due [%s] → ucap laptop: %s",
            trigger.trigger_type, msg[:60],
        )

        try:
            await self._speaker.ucapkan_laptop_async(msg)
        except Exception as e:
            logger.error("[scheduler] Gagal ucap trigger via laptop: %s", e)

    # ── Utilitas ──────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "running":                 self._running,
            "task_count":              len(self._tasks),
            "total_runs":              sum(t._run_count   for t in self._tasks),
            "total_errors":            sum(t._error_count for t in self._tasks),
            "profiler_analyzed_today": self._profiler_analyzed_today,
            "last_analyze_date":       self._last_analyze_date,
            "tasks":                   [t.stats for t in self._tasks],
        }

    def force_run(self, task_name: str) -> bool:
        for task in self._tasks:
            if task.name == task_name:
                t = asyncio.create_task(task.run(), name=f"otto.scheduler.force.{task_name}")
                self._background_tasks.add(t)

                def _on_done(fut: asyncio.Task, _name: str = task_name) -> None:
                    self._background_tasks.discard(fut)
                    if not fut.cancelled() and fut.exception() is not None:
                        logger.error(
                            "[scheduler] force_run '%s' selesai dengan exception:",
                            _name,
                            exc_info=fut.exception(),
                        )

                t.add_done_callback(_on_done)
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

def init_scheduler(activity_watcher, profiler, curiosity, memory=None,
                   context_engine=None, speaker=None) -> Scheduler:
    global _scheduler_instance
    _scheduler_instance = Scheduler(
        activity_watcher, profiler, curiosity,
        memory         = memory,
        context_engine = context_engine,
        speaker        = speaker,
    )
    return _scheduler_instance


# ──────────────────────────── Quick Test ─────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s  %(message)s")

    class MockWatcher:
        async def flush(self): print("  [watcher] flush()")

    class MockProfiler:
        def get_pending(self): return []
        async def analyze(self): return []

    class MockCuriosity:
        async def try_ask(self):
            return "Rofi, kamu biasa aktif jam berapa?", "hyp_test_001"

    class MockEngine:
        def get_due_triggers(self): return []
        def mark_done(self, tid): pass

    class MockSpeaker:
        async def ucapkan_laptop_async(self, text):
            print(f"  [speaker] ucapkan: {text[:50]}")

    async def _test():
        scheduler = Scheduler(
            MockWatcher(), MockProfiler(), MockCuriosity(),
            context_engine = MockEngine(),
            speaker        = MockSpeaker(),
        )

        async def on_q(q, hid): print(f"\n  >>> Pertanyaan: {q}\n")
        scheduler.set_question_callback(on_q)
        scheduler._profiler_analyzed_today = True

        for t in scheduler._tasks:
            t.interval     = 2
            t.run_at_start = True

        print("=== START ===")
        await scheduler.start()
        await asyncio.sleep(6)

        print("\n=== STATS ===")
        for t in scheduler.get_stats()["tasks"]:
            print(f"  {t['name']:25} run={t['run_count']} err={t['error_count']}")

        await scheduler.stop()

    asyncio.run(_test())
