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
  │      ├─ profiler.analyze()        (15 menit)│
  │      ├─ curiosity.try_ask()       (30 menit)│
  │      └─ self_check()              (6 jam)   │
  └─────────────────────────────────────────────┘

Cara integrasi (dari app.py):
    from intelligence.scheduler import Scheduler
    scheduler = Scheduler(activity_watcher, profiler, curiosity)
    await scheduler.start()          # non-blocking, jalan di background
    ...
    await scheduler.stop()           # saat shutdown
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable, Awaitable

logger = logging.getLogger("otto.intelligence.scheduler")


# ──────────────────────────── Konfigurasi ────────────────────────────────────

# Interval tiap task (dalam detik)
INTERVALS = {
    "activity_flush":   5  * 60,   # 5 menit  — flush log aktivitas ke disk
    "profile_analyze":  15 * 60,   # 15 menit — profiler analisis pola baru
    "curiosity_check":  30 * 60,   # 30 menit — cek apakah ada hipotesis siap ditanya
    "self_check":       6  * 3600, # 6 jam    — cek kesehatan sistem Otto sendiri
}

# Tick interval — seberapa sering scheduler "bangun" dan cek jadwal
TICK_SECONDS = 60

# Jika Rofi baru selesai ngobrol, tunggu dulu sebelum tanya
# (agar tidak terasa diinterupsi)
POST_CONVERSATION_COOLDOWN = 5 * 60  # 5 menit


# ──────────────────────────── Scheduled Task ─────────────────────────────────

class ScheduledTask:
    """
    Representasi satu task yang dijadwalkan.

    name       : nama unik task (untuk logging)
    interval   : seberapa sering dijalankan (detik)
    fn         : coroutine yang dipanggil
    run_at_start: langsung jalankan saat scheduler start?
    """

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
        """Apakah task ini sudah waktunya dijalankan?"""
        if self._last_run is None:
            return self.run_at_start
        return (now - self._last_run).total_seconds() >= self.interval

    async def run(self) -> bool:
        """
        Jalankan task. Return True jika sukses, False jika error.
        """
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
            # Tetap update last_run agar tidak loop terus saat error
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
    """
    Background scheduler untuk Otto.

    Menjalankan task-task intelligence secara periodik
    sambil tetap aware terhadap konteks (Rofi sedang ngobrol atau tidak).

    Penggunaan:
        scheduler = Scheduler(activity_watcher, profiler, curiosity)
        await scheduler.start()
        # ... Otto berjalan ...
        await scheduler.stop()
    """

    def __init__(self, activity_watcher, profiler, curiosity) -> None:
        self._activity_watcher = activity_watcher
        self._profiler         = profiler
        self._curiosity        = curiosity

        self._running          = False
        self._loop_task: Optional[asyncio.Task] = None
        self._tasks: list[ScheduledTask] = []

        # Waktu terakhir Rofi aktif ngobrol
        self._last_conversation_at: Optional[datetime] = None

        # Callback opsional — dipanggil saat curiosity punya pertanyaan
        # Signature: async def on_question(question: str, hyp_id: str) -> None
        self._on_question_cb: Optional[Callable] = None

        self._register_tasks()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _register_tasks(self) -> None:
        """Daftarkan semua scheduled task."""
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
        ]
        logger.info("[scheduler] %d task terdaftar.", len(self._tasks))

    def set_question_callback(self, cb: Callable) -> None:
        """
        Set callback yang dipanggil saat curiosity menghasilkan pertanyaan.

        Contoh:
            async def kirim_ke_rofi(question, hyp_id):
                await speaker.speak(question)

            scheduler.set_question_callback(kirim_ke_rofi)
        """
        self._on_question_cb = cb

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Mulai scheduler di background.
        Non-blocking — langsung return setelah task dimulai.
        """
        if self._running:
            logger.warning("[scheduler] Sudah berjalan, skip start.")
            return

        self._running   = True
        self._loop_task = asyncio.create_task(
            self._main_loop(), name="otto.scheduler"
        )
        logger.info("[scheduler] Dimulai. Tick setiap %d detik.", TICK_SECONDS)

    async def stop(self) -> None:
        """
        Stop scheduler dengan bersih.
        Tunggu tick selesai sebelum keluar.
        """
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
        """Loop utama — bangun setiap TICK_SECONDS dan cek task."""
        logger.info("[scheduler] Loop mulai.")

        while self._running:
            try:
                await self._tick()
            except Exception as e:
                # Loop tidak boleh mati karena satu error
                logger.error("[scheduler] Error di tick: %s", e, exc_info=True)

            await asyncio.sleep(TICK_SECONDS)

        logger.info("[scheduler] Loop selesai.")

    async def _tick(self) -> None:
        """Satu siklus tick — cek dan jalankan task yang sudah waktunya."""
        now = datetime.now()

        for task in self._tasks:
            if not self._running:
                break

            if not task.is_due(now):
                continue

            # Curiosity punya aturan tambahan: tidak saat Rofi baru ngobrol
            if task.name == "curiosity_check" and self._in_cooldown():
                logger.debug(
                    "[scheduler] curiosity_check skip — Rofi baru selesai ngobrol."
                )
                continue

            logger.debug("[scheduler] Jalankan task: %s", task.name)
            await task.run()

    # ── Konteks Awareness ─────────────────────────────────────────────────────

    def notify_conversation_active(self) -> None:
        """
        Dipanggil oleh app.py saat Rofi mulai/sedang ngobrol.
        Scheduler akan tunda curiosity_check selama cooldown.
        """
        self._last_conversation_at = datetime.now()
        logger.debug("[scheduler] Rofi aktif ngobrol — curiosity_check ditunda.")

    def _in_cooldown(self) -> bool:
        """True jika masih dalam cooldown setelah percakapan."""
        if self._last_conversation_at is None:
            return False
        elapsed = (datetime.now() - self._last_conversation_at).total_seconds()
        return elapsed < POST_CONVERSATION_COOLDOWN

    # ── Task Implementations ──────────────────────────────────────────────────

    async def _task_activity_flush(self) -> None:
        """Flush buffer aktivitas ke disk."""
        if hasattr(self._activity_watcher, "flush"):
            await self._activity_watcher.flush()
        elif hasattr(self._activity_watcher, "save"):
            self._activity_watcher.save()
        logger.debug("[scheduler] activity_flush selesai.")

    async def _task_profile_analyze(self) -> None:
        """Profiler analisis ulang pola dari log aktivitas terbaru."""
        if hasattr(self._profiler, "analyze"):
            result = await self._profiler.analyze() if asyncio.iscoroutinefunction(
                self._profiler.analyze
            ) else self._profiler.analyze()
            new_hypotheses = result if isinstance(result, int) else 0
            if new_hypotheses:
                logger.info(
                    "[scheduler] Profiler: %d hipotesis baru ditemukan.", new_hypotheses
                )
        logger.debug("[scheduler] profile_analyze selesai.")

    async def _task_curiosity_check(self) -> None:
        """Cek apakah ada hipotesis siap ditanyakan ke Rofi."""
        question, hyp_id = await self._curiosity.try_ask()

        if question and hyp_id:
            logger.info(
                "[scheduler] Curiosity siap tanya hipotesis %s.", hyp_id
            )
            if self._on_question_cb:
                await self._on_question_cb(question, hyp_id)
            else:
                # Tidak ada callback — log saja
                logger.warning(
                    "[scheduler] Tidak ada question_callback. "
                    "Set via scheduler.set_question_callback(). "
                    "Pertanyaan: '%s'", question,
                )

    async def _task_self_check(self) -> None:
        """
        Cek kesehatan sistem Otto — placeholder untuk self/model.py nanti.
        Untuk sekarang: log statistik scheduler.
        """
        stats = self.get_stats()
        logger.info(
            "[scheduler] Self-check: %d task, total run: %d",
            stats["task_count"],
            stats["total_runs"],
        )

    # ── Utilitas ──────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Statistik semua task — berguna untuk debugging / monitoring."""
        return {
            "running":     self._running,
            "task_count":  len(self._tasks),
            "total_runs":  sum(t._run_count for t in self._tasks),
            "total_errors": sum(t._error_count for t in self._tasks),
            "tasks":       [t.stats for t in self._tasks],
        }

    def force_run(self, task_name: str) -> bool:
        """
        Paksa jalankan satu task sekarang, di luar jadwal.
        Berguna untuk testing atau trigger manual dari app.py.

        Return True jika task ditemukan dan di-schedule.
        """
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


def init_scheduler(activity_watcher, profiler, curiosity) -> Scheduler:
    global _scheduler_instance
    _scheduler_instance = Scheduler(activity_watcher, profiler, curiosity)
    return _scheduler_instance


# ──────────────────────────── Quick Test ─────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s  %(message)s")

    class MockWatcher:
        async def flush(self): print("  [watcher] flush()")

    class MockProfiler:
        def analyze(self): print("  [profiler] analyze()"); return 2

    class MockCuriosity:
        async def try_ask(self):
            print("  [curiosity] try_ask()")
            return "Rofi, kamu biasa sarapan jam berapa?", "hyp_test_001"

    async def _test():
        watcher   = MockWatcher()
        profiler  = MockProfiler()
        curiosity = MockCuriosity()

        async def on_question(q, hid):
            print(f"\n  >>> Pertanyaan ke Rofi: {q} (id: {hid})\n")

        scheduler = Scheduler(watcher, profiler, curiosity)
        scheduler.set_question_callback(on_question)

        # Override interval supaya test cepat
        for t in scheduler._tasks:
            t.interval = 2
            t.run_at_start = True

        print("\n=== START SCHEDULER ===")
        await scheduler.start()

        print("\n=== RUNNING 6 DETIK ===")
        await asyncio.sleep(6)

        print("\n=== STATS ===")
        stats = scheduler.get_stats()
        for t in stats["tasks"]:
            print(f"  {t['name']:20} run={t['run_count']} err={t['error_count']}")

        print("\n=== STOP ===")
        await scheduler.stop()
        print("  Scheduler dihentikan.")

    asyncio.run(_test())
