"""
core/executor.py — Dispatcher Perintah Otto
============================================
Tanggung jawab:
  - Terima teks dari brain.py
  - Deteksi apakah teks adalah PERINTAH (aksi) atau OBROLAN (LLM)
  - Jika perintah → dispatch ke skill yang tepat (system, media, reminder, dsb)
  - Jika obrolan → kembalikan ke brain untuk dijawab LLM
  - Kembalikan ExecutorResult yang seragam ke app.py

Alur:
  app.py → executor.dispatch(text)
               ├─ intent = COMMAND → skill.run(params) → ExecutorResult
               └─ intent = CHAT    → brain.think(text)  → ExecutorResult
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger("otto.executor")


# ─────────────────────────── Tipe ───────────────────────────────────────────

class Intent(Enum):
    COMMAND = "command"   # ada skill yang bisa handle
    CHAT    = "chat"      # tidak ada skill → lempar ke LLM


@dataclass
class ExecutorResult:
    success:  bool
    text:     str                      # teks untuk diucapkan / dikirim ke user
    intent:   Intent = Intent.CHAT
    skill:    str    = ""              # nama skill yang dipakai (jika COMMAND)
    data:     dict[str, Any] = field(default_factory=dict)  # payload tambahan
    error:    str    = ""


# ─────────────────────────── Skill Registry ─────────────────────────────────

@dataclass
class SkillHandler:
    """Satu entri di registry skill."""
    name:     str
    pattern:  re.Pattern                            # regex untuk deteksi intent
    handler:  Callable[..., Coroutine]              # async function
    examples: list[str] = field(default_factory=list)


class Executor:
    """
    Dispatcher utama Otto.

    Contoh pemakaian:
        executor = Executor(brain)
        result   = await executor.dispatch("matiin lampu ruang tamu")
        print(result.text)
    """

    def __init__(self, brain) -> None:
        self.brain    = brain
        self._skills: list[SkillHandler] = []
        self._register_builtin_skills()
        logger.info("Executor siap. %d skill terdaftar.", len(self._skills))

    # ── Public API ───────────────────────────────────────────────────────────

    async def dispatch(self, text, history=None):
        text = text.strip()
        if not text:
            return ExecutorResult(
                success=False,
                text="Aku tidak dengar apa-apa.",
                error="empty input"
            )

        # ── [1] Skill via regex — paling cepat, tidak butuh shortcut ──────
        match = self._find_skill(text)
        if match:
            skill, groups = match
            logger.info("[executor] COMMAND → skill=%s", skill.name)
            return await self._run_skill(skill, text, groups)

        # ── [2] Shortcut — LLM pernah jawab ini sebelumnya ────────────────
        from core.shortcut import check as shortcut_check, record as shortcut_record

        cached = shortcut_check(text)
        if cached:
            logger.info("[executor] SHORTCUT → '%s'", text[:50])
            return ExecutorResult(
                success = cached.get("success", True),
                text    = cached.get("text", ""),
                intent  = Intent.COMMAND,
                skill   = "shortcut",
            )

        # ── [3] LLM — tidak ada skill, tidak ada shortcut ─────────────────
        logger.info("[executor] CHAT → brain '%s'", text[:60])
        result = await self._run_chat(text, history or [])

        # Catat ke shortcut supaya request berikutnya bypass LLM
        shortcut_record(text, {
            "success": result.success,
            "text":    result.text,
            "data":    result.data,
        })

        return result    





    def register(
        self,
        name:     str,
        pattern:  str,
        handler:  Callable,
        examples: list[str] | None = None,
        flags:    int = re.IGNORECASE,
    ) -> None:
        """
        Daftarkan skill eksternal secara dinamis.

        Contoh dari skills/reminder.py::

            executor.register(
                name     = "reminder",
                pattern  = r"ingatkan aku (.+) jam (\\d+)",
                handler  = reminder_handler,
                examples = ["ingatkan aku minum obat jam 8"],
            )
        """
        compiled = re.compile(pattern, flags)
        self._skills.append(SkillHandler(
            name     = name,
            pattern  = compiled,
            handler  = handler,
            examples = examples or [],
        ))
        logger.debug("Skill '%s' terdaftar.", name)

    def list_skills(self) -> list[dict]:
        """Kembalikan daftar skill yang aktif (untuk debug / introspeksi)."""
        return [
            {
                "name":     s.name,
                "pattern":  s.pattern.pattern,
                "examples": s.examples,
            }
            for s in self._skills
        ]

    # ── Skill Matching ────────────────────────────────────────────────────────

    def _find_skill(self, text: str) -> tuple[SkillHandler, re.Match] | None:
        """Coba cocokkan teks ke semua skill. Return (skill, match) atau None."""
        for skill in self._skills:
            m = skill.pattern.search(text)
            if m:
                return skill, m
        return None

    # ── Skill Runner ──────────────────────────────────────────────────────────

    async def _run_skill(
        self,
        skill:  SkillHandler,
        text:   str,
        match:  re.Match,
    ) -> ExecutorResult:
        try:
            result = await skill.handler(text=text, match=match, brain=self.brain)
            # Skill bisa return string atau ExecutorResult
            if isinstance(result, ExecutorResult):
                result.intent = Intent.COMMAND
                result.skill  = skill.name
                return result
            return ExecutorResult(
                success = True,
                text    = str(result),
                intent  = Intent.COMMAND,
                skill   = skill.name,
            )
        except Exception as e:
            logger.error("[executor] Skill '%s' error: %s", skill.name, e, exc_info=True)
            return ExecutorResult(
                success = False,
                text    = f"Aduh, ada yang error waktu jalanin {skill.name}.",
                intent  = Intent.COMMAND,
                skill   = skill.name,
                error   = str(e),
            )

    # ── Chat Runner ───────────────────────────────────────────────────────────

    async def _run_chat(
        self,
        text:    str,
        history: list[dict],
    ) -> ExecutorResult:
        try:
            # Ambil short-term history dari memory jika tidak disuplai
            if not history:
                history = self.brain.memory.get_short_term()

            resp = await self.brain.think(text, history=history)
            return ExecutorResult(
                success = True,
                text    = resp.text,
                intent  = Intent.CHAT,
                data    = {
                    "model":   resp.model,
                    "latency": resp.latency_ms,
                    "tokens":  resp.prompt_tokens + resp.completion_tokens,
                },
            )
        except Exception as e:
            logger.error("[executor] Brain error: %s", e, exc_info=True)
            return ExecutorResult(
                success = False,
                text    = "Maaf, aku lagi ada masalah koneksi ke otak. Coba lagi.",
                intent  = Intent.CHAT,
                error   = str(e),
            )

    # ── Built-in Skills ───────────────────────────────────────────────────────

    def _register_builtin_skills(self) -> None:
        """
        Skill bawaan ringan — tidak perlu file skill terpisah.
        Skill besar (system, media, reminder) ada di skills/*.py dan
        didaftarkan dari sana lewat executor.register().
        """

        # ── Skill: jam / waktu ───────────────────────────────────────────────
        async def skill_time(text, match, brain, **_):
            from datetime import datetime
            now = datetime.now()
            hari = ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"]
            return f"Sekarang {hari[now.weekday()]}, {now.strftime('%d %B %Y')}, jam {now.strftime('%H:%M')}."

        self.register(
            name     = "waktu",
            pattern  = r"\b(jam berapa|sekarang jam|hari apa|tanggal berapa|waktu sekarang)\b",
            handler  = skill_time,
            examples = ["jam berapa sekarang?", "hari ini tanggal berapa?"],
        )

        # ── Skill: ping / status Otto ────────────────────────────────────────
        async def skill_ping(text, match, brain, **_):
            short = brain.memory.short_term_count()
            long  = brain.memory.long_term_count()
            keys  = len(brain._keys)
            return (
                f"Otto aktif. "
                f"Memory: {short} pesan pendek, {long} fakta tersimpan. "
                f"{keys} API key siap."
            )

        self.register(
            name     = "status",
            pattern  = r"\b(otto|kamu) (baik|aktif|hidup|nyala|status|ping)\b",
            handler  = skill_ping,
            examples = ["otto aktif?", "kamu baik-baik aja?"],
        )

        # ── Skill: ingat (simpan fakta manual) ──────────────────────────────
        async def skill_remember(text, match, brain, **_):
            # "otto ingat, aku suka kopi oat"
            fact = match.group(1).strip()
            key  = f"rofi.manual.{len(fact)}"   # key sementara — profiler akan refine
            brain.memory.remember(key, fact, source="rofi_manual")
            return f"Oke, aku catat: '{fact}'."

        self.register(
            name     = "ingat",
            pattern  = r"\b(?:otto\s+)?ingat(?:kan)?\b[,\s]+(.+)",
            handler  = skill_remember,
            examples = ["otto ingat, aku alergi kacang", "ingat aku suka musik jazz"],
        )

        # ── Skill: lupa (hapus fakta) ────────────────────────────────────────
        async def skill_forget(text, match, brain, **_):
            keyword = match.group(1).strip()
            found   = brain.memory.search(keyword)
            if not found:
                return f"Aku tidak nemuin yang berhubungan dengan '{keyword}'."
            for k in found:
                brain.memory.forget(k)
            return f"Oke, aku hapus {len(found)} catatan tentang '{keyword}'."

        self.register(
            name     = "lupa",
            pattern  = r"\b(?:otto\s+)?(?:lupa|hapus|forget)\b[,\s]+(.+)",
            handler  = skill_forget,
            examples = ["otto lupa tentang kopi", "hapus catatan tentang jadwal"],
        )

        # ── Skill: apa yang kamu tau ─────────────────────────────────────────
        async def skill_recall(text, match, brain, **_):
            keyword = match.group(1).strip() if match.lastindex else ""
            if keyword:
                found = brain.memory.search(keyword)
            else:
                found = brain.memory.all_confirmed()

            if not found:
                return "Aku belum punya catatan yang relevan."

            lines = [f"Ini yang aku ingat tentang '{keyword}':" if keyword else "Yang aku tahu (sudah dikonfirmasi):"]
            for k, v in list(found.items())[:8]:
                lines.append(f"- {k}: {v['value']} ({v['source']})")
            return "\n".join(lines)

        self.register(
            name     = "recall",
            pattern  = r"\b(?:kamu|otto)\s+(?:tahu|tau|ingat|catat)\s+(?:apa|tentang)\s*(.*)",
            handler  = skill_recall,
            examples = ["kamu tau apa tentang aku?", "otto ingat tentang kopi?"],
        )

        # ── Skill: load skill eksternal ──────────────────────────────────────
        # Skills dari skills/*.py bisa mendaftarkan diri sendiri via:
        #   from core.executor import executor_instance
        #   executor_instance.register(...)
        # Atau dipanggil manual dari app.py saat startup.


# ─────────────────────────── Singleton Helper ───────────────────────────────

_executor_instance: Executor | None = None

def get_executor() -> Executor:
    """Ambil instance executor yang sudah dibuat. Raise jika belum diinit."""
    if _executor_instance is None:
        raise RuntimeError("Executor belum diinisialisasi. Panggil init_executor(brain) dulu.")
    return _executor_instance

def init_executor(brain) -> Executor:
    """Buat dan simpan executor singleton. Dipanggil sekali dari app.py."""
    global _executor_instance
    _executor_instance = Executor(brain)
    return _executor_instance


# ─────────────────────────── Quick Test ─────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # Mock brain sederhana untuk test tanpa Groq
    class MockMemory:
        def get_short_term(self): return []
        def short_term_count(self): return 0
        def long_term_count(self): return 2
        def remember(self, k, v, source=""): print(f"[mock] remember {k}={v}")
        def search(self, kw): return {}
        def forget(self, k): pass
        def all_confirmed(self): return {}
        def summary_for_llm(self): return ""

    class MockBrain:
        memory = MockMemory()
        _keys  = ["key1", "key2"]
        async def think(self, text, history=None):
            from dataclasses import dataclass
            @dataclass
            class R:
                text: str
                model: str = "mock"
                latency_ms: float = 0
                prompt_tokens: int = 0
                completion_tokens: int = 0
            return R(text=f"[mock LLM] Jawaban untuk: {text}")

    async def _test():
        ex = Executor(MockBrain())

        tests = [
            "jam berapa sekarang?",
            "otto aktif?",
            "otto ingat, aku suka nasi goreng",
            "kamu tau apa tentang aku?",
            "menurutmu kenapa langit biru?",   # → CHAT
            "matiin lampu dong",               # → CHAT (skill belum ada)
        ]

        for t in tests:
            print(f"\n[INPUT]  {t}")
            r = await ex.dispatch(t)
            print(f"[INTENT] {r.intent.value} | skill={r.skill or '-'} | ok={r.success}")
            print(f"[OUTPUT] {r.text}")

    asyncio.run(_test())
