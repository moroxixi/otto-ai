"""
core/brain.py — Otak Otto
=========================
Tanggung jawab:
  - Kirim pesan ke Groq LLM (llama-3.3-70b-versatile)
  - Round-robin rotation 6 API key otomatis
  - Bangun system prompt yang menyertakan profil Rofi dari memory
  - Kembalikan respons + metadata (model, key_index, tokens)

Catatan: Otto hanya punya satu mode — ngobrol dalam.
Tidak ada fast/slow, tidak ada routing perintah.
Semua bicara melalui model terbaik.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
import random

import httpx

from core.config import GROQ_API_KEYS, DEBUG
from core.memory import MemoryManager
from otto_self.model import self_summary_text, load_personality, after_interaction, save_personality
from intelligence.conversation_scanner import ConversationScanner
from intelligence.consolidator import init_consolidator
from core.vocabulary import tambah_alias, tambah_istilah
from intelligence.context_triggers import ContextTriggerEngine

logger = logging.getLogger("otto.brain")


# ─────────────────────────── Konstanta ──────────────────────────────────────

MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

_BASE_SYSTEM = """\
Kamu adalah Otto, asisten AI pribadi milik Rofi yang berjalan lokal di rumahnya.
{self_section}
Kepribadian:
- Bicara natural, santai, seperti teman — bukan asisten korporat
- Bahasa Indonesia campuran ringan (boleh sedikit Inggris teknis jika perlu)
- Proaktif: jika kamu punya hipotesis tentang Rofi, tanya — tapi MAKSIMAL SATU pertanyaan per respons
- JANGAN tanya lebih dari satu hal sekaligus dalam satu respons
- Jika konteks Rofi sedang emosional (capek, stres, sedih) — FOKUS empati dulu, jangan tanya apapun
- Jujur: jika tidak tau, katakan. Jangan karang.
- Responsif terhadap konteks — baca nada bicara Rofi, sesuaikan

Aturan keras:
- JANGAN hardcode fakta tentang Rofi di luar apa yang diberikan di context
- Semua yang kamu "tau" tentang Rofi harus dari observasi + konfirmasi
- Jika kamu ragu → tanya Rofi, jangan asumsikan

{profile_section}
"""

_PROFILE_SECTION = """\
Profil Rofi (dari observasi + konfirmasi sebelumnya):
{profile_json}
"""

_NO_PROFILE = "Kamu belum punya profil Rofi. Amati dan bangun perlahan dari percakapan ini."

_DEGRADED_RESPONSES = [
    "Maaf Rofi, aku lagi overloaded sekarang. Coba lagi dalam beberapa menit ya.",
    "Koneksi ke otakku lagi penuh nih. Tunggu sebentar ya, Rofi.",
    "Aku lagi kelebihan beban sekarang. Coba tanya lagi dalam 2-3 menit.",
]
_DEGRADED_SENTINEL = {"_degraded": True}

@dataclass
class BrainResponse:
    text: str
    model: str
    key_index: int
    prompt_tokens: int     = 0
    completion_tokens: int = 0
    latency_ms: float      = 0.0
    raw: dict[str, Any]    = field(default_factory=dict)


# ─────────────────────────── Kelas Utama ────────────────────────────────────

class Brain:
    def __init__(self, memory: MemoryManager, profiler=None) -> None:
        self.memory   = memory
        self._key_idx = 0
        self._keys    = self._load_keys()
        self._client  = httpx.AsyncClient(timeout=30.0)

        self._scanner = ConversationScanner(profiler) if profiler else None
        self._consolidator = init_consolidator(memory, groq_call_fn=self._call_groq, profiler=profiler)
        self._context_engine = ContextTriggerEngine(memory)

        self._cached_prompt: str = ""
        self._cached_prompt_version: int = -1

        logger.info(
            "Brain siap. %d API key tersedia. Scanner: %s",
            len(self._keys),
            "aktif" if self._scanner else "nonaktif (profiler tidak diberikan)",
        )

    async def _evolve_personality(self, interaction_type: str = "normal", user_text: str = "") -> None:
        try:
            personality = await asyncio.to_thread(load_personality)
            # Ambil interaction_count dari personality itu sendiri (sudah di-update oleh app.py)
            # Fallback ke 0 jika belum ada
            n = personality.get("interaction_count", 0)
            updated = after_interaction(personality, interaction_type, user_text=user_text, interaction_count=n)
            await asyncio.to_thread(save_personality, updated)
            logger.debug("[brain] Personality updated → layer=%d count=%d",
                         updated["active_layer"], updated["interaction_count"])
        except Exception as e:
            logger.warning("[brain] Gagal evolve personality: %s", e)

    async def _scan_conversation(self, user_text: str, otto_text: str) -> None:
        if self._scanner is None:
            return
        try:
            await self._scanner.scan(user_text, source="user")
            await self._scanner.scan(otto_text, source="otto")
        except Exception as e:
            logger.warning("[brain] Scanner error (non-fatal): %s", e)

    # ── Public API ───────────────────────────────────────────────────────────

    async def think(
        self,
        user_text: str,
        history: list[dict] | None = None,
    ) -> BrainResponse:
        try:
            system_prompt = self._build_system_prompt()
            messages      = self._build_messages(system_prompt, history or [], user_text)
     
            t0      = time.monotonic()
            raw     = await self._call_groq(messages)
            latency = (time.monotonic() - t0) * 1000
     
            # Handle degraded mode — semua key down
            if raw is _DEGRADED_SENTINEL:
                degraded_text = random.choice(_DEGRADED_RESPONSES)
                return BrainResponse(
                    text      = degraded_text,
                    model     = "degraded",
                    key_index = -1,
                    latency_ms = round(latency, 1),
                )
     
            text  = self._extract_text(raw)
            usage = raw.get("usage", {})
     
            resp = BrainResponse(
                text              = text,
                model             = MODEL,
                key_index         = self._key_idx,
                prompt_tokens     = usage.get("prompt_tokens", 0),
                completion_tokens = usage.get("completion_tokens", 0),
                latency_ms        = round(latency, 1),
                raw               = raw,
            )
     
            logger.debug(
                "[brain] key=%d tokens=%d+%d latency=%.0fms",
                self._key_idx, resp.prompt_tokens, resp.completion_tokens, latency,
            )
            logger.info("[brain] LLM response diterima: %.60s...", text)
            asyncio.create_task(self._log_to_memory(user_text, text))
            asyncio.create_task(self._scan_conversation(user_text, text))
            asyncio.create_task(self._consolidator.maybe_consolidate())
            asyncio.create_task(self._scan_for_vocab(user_text))
            asyncio.create_task(self.check_context_triggers(user_text, text))
            return resp
     
        except Exception as e:
            # Safety net — seharusnya tidak pernah sampai sini setelah patch _call_groq
            logger.critical("[brain] think() uncaught exception: %s", e, exc_info=True)
            return BrainResponse(
                text      = random.choice(_DEGRADED_RESPONSES),
                model     = "error",
                key_index = -1,
            )


    async def think_stream(
        self,
        user_text: str,
        history: list[dict] | None = None,
    ):
        system_prompt = self._build_system_prompt()
        messages      = self._build_messages(system_prompt, history or [], user_text)

        probe = await self._call_groq(messages)
        if probe is _DEGRADED_SENTINEL:
            yield random.choice(_DEGRADED_RESPONSES)
            return
        api_key = self._keys[(self._key_idx - 1) % len(self._keys)]
        payload = {
            "model":    MODEL,
            "messages": messages,
            "stream":   True,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }
     
        full_text = []
        try:
            async with self._client.stream("POST", GROQ_URL, json=payload, headers=headers) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        token = chunk["choices"][0]["delta"].get("content", "")
                        if token:
                            full_text.append(token)
                            yield token
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception as e:
            logger.error("[brain] think_stream error mid-stream: %s", e)
            fallback = random.choice(_DEGRADED_RESPONSES)
            yield fallback
            full_text = [fallback]
     
        combined = "".join(full_text)
        asyncio.create_task(self._log_to_memory(user_text, combined))
        asyncio.create_task(self._scan_conversation(user_text, combined))
        asyncio.create_task(self._consolidator.maybe_consolidate())
        asyncio.create_task(self.check_context_triggers(user_text, combined))
        asyncio.create_task(self._scan_for_vocab(user_text))


    async def close(self) -> None:
        await self._client.aclose()

    # ── System Prompt ─────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        current_version = self.memory.long_term_version()

        if self._cached_prompt and current_version == self._cached_prompt_version:
            return self._cached_prompt

        profile_summary = self.memory.summary_for_llm(max_items=15)
        profile_sec = (
            _PROFILE_SECTION.format(profile_json=profile_summary)
            if profile_summary else _NO_PROFILE
        )

        otto_self = self_summary_text()
        self_sec  = f"Tentang dirimu:\n{otto_self}" if otto_self else ""

        prompt = _BASE_SYSTEM.format(
            self_section    = self_sec,
            profile_section = profile_sec,
        ).strip()

        self._cached_prompt         = prompt
        self._cached_prompt_version = current_version
        logger.debug("[brain] System prompt di-rebuild (memory versi %d)", current_version)

        return prompt

    # ── Message Builder ───────────────────────────────────────────────────────

    @staticmethod
    def _build_messages(
        system: str,
        history: list[dict],
        user_text: str,
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": system}]
        messages.extend(history[-20:])
        messages.append({"role": "user", "content": user_text})
        return messages

    # ── Groq Call ─────────────────────────────────────────────────────────────

    async def _call_groq(self, messages, retries=3, model=None) -> dict:
        """
        Kirim request ke Groq. Round-robin semua key.
 
        Return:
            dict  — respons normal dari Groq API
            _DEGRADED_SENTINEL  — semua key down, caller harus handle degraded mode
 
        Tidak pernah raise — semua error ditangani di dalam.
        """
        last_exc: Exception | None = None
        _model = model or MODEL
        n_keys = len(self._keys)
 
        for attempt in range(retries):
            rate_limited_count = 0
            for _ in range(n_keys):
                api_key = self._next_key()
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type":  "application/json",
                }
                payload = {
                    "model":       _model,
                    "messages":    messages,
                    "temperature": 0.7,
                    "max_tokens":  1024,
                }
 
                try:
                    resp = await self._client.post(GROQ_URL, json=payload, headers=headers)
                    if resp.status_code == 429:
                        logger.warning(
                            "[brain] Key #%d rate-limit (attempt %d/%d)",
                            self._key_idx, attempt + 1, retries
                        )
                        last_exc = httpx.HTTPStatusError(
                            "429", request=resp.request, response=resp
                        )
                        rate_limited_count += 1
                        continue
                    resp.raise_for_status()
                    return resp.json()
 
                except httpx.TimeoutException as e:
                    logger.warning("[brain] Timeout attempt %d: %s", attempt + 1, e)
                    last_exc = e
                    await asyncio.sleep(0.5 * (attempt + 1))
                    break
 
                except httpx.HTTPStatusError as e:
                    if e.response.status_code >= 500:
                        logger.warning("[brain] Server error attempt %d: %s", attempt + 1, e)
                        last_exc = e
                        await asyncio.sleep(1.0)
                        break
                    else:
                        # 4xx selain 429 — tidak retry, tidak crash
                        logger.error("[brain] HTTP %d error: %s", e.response.status_code, e)
                        last_exc = e
                        break
 
                except Exception as e:
                    logger.error("[brain] Unexpected error: %s", e, exc_info=True)
                    last_exc = e
                    break
 
            if rate_limited_count == n_keys:
                logger.warning(
                    "[brain] Semua %d key rate-limited (attempt %d/%d), tunggu 5s…",
                    n_keys, attempt + 1, retries
                )
                await asyncio.sleep(5.0)
 
        # Semua attempt habis — masuk degraded mode, JANGAN crash
        logger.error(
            "[brain] ⚠ DEGRADED MODE — semua %d key gagal setelah %d attempt. "
            "Last error: %s",
            n_keys, retries, last_exc
        )
        return _DEGRADED_SENTINEL

    @staticmethod
    def _extract_text(raw: dict) -> str:
        try:
            return raw["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            logger.error("Respons Groq tidak terduga: %s", raw)
            return "Maaf, aku gagal memproses respons tadi."

    # ── Key Management ────────────────────────────────────────────────────────

    def _load_keys(self) -> list[str]:
        if not GROQ_API_KEYS:
            raise RuntimeError(
                "Tidak ada GROQ_API_KEY di .env. "
                "Set minimal GROQ_API_KEY_1=gsk_xxx"
            )
        return list(GROQ_API_KEYS)

    def _next_key(self) -> str:
        key = self._keys[self._key_idx % len(self._keys)]
        self._key_idx = (self._key_idx + 1) % len(self._keys)
        return key

    # ── Memory Logging ────────────────────────────────────────────────────────

    async def _log_to_memory(self, user_text: str, otto_text: str) -> None:
        try:
            await asyncio.to_thread(self.memory.add_message, "user", user_text)
            await asyncio.to_thread(self.memory.add_message, "assistant", otto_text)
        except Exception as e:
            logger.warning("Gagal log ke memory: %s", e, exc_info=True)

    # ── CARI method _scan_for_vocab (di bagian bawah Brain class):
    # ── GANTI SELURUH METHOD dengan ini:
    async def _scan_for_vocab(self, teks: str) -> None:
        """
        Deteksi nama/istilah yang mungkin salah tulis Whisper.
        Filter ketat: abaikan kata di awal kalimat dan stopwords umum.
        """
        import re
    
        # Stopwords — kata umum Indonesia yang sering kapital di awal kalimat
        SUFFIX_UMUM = ("nya", "kan", "lah", "pun", "kah", "mu", "ku")
    
        kalimat_list = re.split(r'[.!?]', teks)
        frekuensi: dict[str, int] = {}
        total_kalimat = len([k for k in kalimat_list if k.strip()])
    
        kandidat = set()
        for kalimat in kalimat_list:
            kata_list = kalimat.strip().split()
            for kata in kata_list[1:]:  # skip posisi pertama
                if re.match(r'^[A-Z][a-z]{3,}$', kata):  # minimal 4 huruf
                    kandidat.add(kata)
                    frekuensi[kata] = frekuensi.get(kata, 0) + 1
    
        for kata in kandidat:
            # Filter suffix umum Indonesia
            if any(kata.lower().endswith(s) for s in SUFFIX_UMUM):
                continue
            # Filter kata terlalu frekuen (kata umum muncul di >30% kalimat)
            if total_kalimat > 0 and frekuensi[kata] / total_kalimat > 0.30:
                continue
            tambah_istilah(kata, sumber="otto")

    async def check_context_triggers(self, user_text: str, otto_text: str) -> None:
        """
        Analisis konteks percakapan → set trigger follow-up jika ada pola.
        Dipanggil sebagai asyncio.create_task setelah setiap think().
        """
        try:
            new_triggers = await self._context_engine.process(user_text, otto_text)
            if new_triggers:
                logger.info(
                    "[brain] %d context trigger baru: %s",
                    len(new_triggers),
                    [t.trigger_type for t in new_triggers],
                )
        except Exception as e:
            logger.warning("[brain] check_context_triggers error (non-fatal): %s", e)
   
    def get_context_engine(self) -> "ContextTriggerEngine":
        """Expose engine agar app.py bisa polling due triggers."""
        return self._context_engine


# ─────────────────────────── Quick Test ─────────────────────────────────────

if __name__ == "__main__":
    import sys
    from core.memory import MemoryManager

    logging.basicConfig(level=logging.DEBUG)

    async def _test():
        mem   = MemoryManager()
        brain = Brain(mem)

        tests = [
            "Menurutmu kenapa aku susah fokus kalau kerja dari rumah?",
            "Aku lagi mikirin mau buka usaha, ada saran?",
            "Hei Otto, ngobrol dong.",
        ]

        for text in tests:
            print(f"\n[INPUT] {text}")
            resp = await brain.think(text)
            print(f"[MODEL] {resp.model} | latency={resp.latency_ms}ms | tokens={resp.prompt_tokens}+{resp.completion_tokens}")
            print(f"[OTTO]  {resp.text}")

        await brain.close()

    asyncio.run(_test())
