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

import httpx

from core.config import GROQ_API_KEYS, DEBUG
from core.memory import MemoryManager
from otto_self.model import self_summary_text, load_personality, after_interaction, save_personality
from intelligence.conversation_scanner import ConversationScanner
from intelligence.consolidator import init_consolidator

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
- Proaktif: jika kamu punya hipotesis tentang Rofi, tanya — jangan simpan sendiri
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
    """
    Otak Otto. Instantiate sekali, pakai terus.

    Contoh:
        brain = Brain(memory)
        resp  = await brain.think("Menurutmu kenapa aku susah fokus?")
        print(resp.text)
    """

    def __init__(self, memory: MemoryManager, profiler=None) -> None:
        self.memory   = memory
        self._key_idx = 0
        self._keys    = self._load_keys()
        self._client  = httpx.AsyncClient(timeout=30.0)

        # Scanner real-time — inject hipotesis dari percakapan langsung
        # profiler boleh None (scanner akan skip jika tidak ada)
        self._scanner = ConversationScanner(profiler) if profiler else None
        self._consolidator = init_consolidator(memory, groq_call_fn=self._call_groq)

        self._cached_prompt: str = ""
        self._cached_prompt_version: int = -1

        logger.info(
            "Brain siap. %d API key tersedia. Scanner: %s",
            len(self._keys),
            "aktif" if self._scanner else "nonaktif (profiler tidak diberikan)",
        )



    async def _evolve_personality(self, interaction_type: str = "normal") -> None:
        """Panggil after_interaction() dan simpan ke disk. Non-blocking."""
        try:
            personality = await asyncio.to_thread(load_personality)
            updated = after_interaction(personality, interaction_type)
            await asyncio.to_thread(save_personality, updated)
            logger.debug("[brain] Personality updated → layer=%d count=%d",
                         updated["active_layer"], updated["interaction_count"])
        except Exception as e:
            logger.warning("[brain] Gagal evolve personality: %s", e)



    async def _scan_conversation(self, user_text: str, otto_text: str) -> None:
        '''
        Jalankan ConversationScanner secara non-blocking.
        Dipanggil sebagai create_task — tidak boleh raise exception ke caller.
        '''
        if self._scanner is None:
            return
        try:
            await self._scanner.scan(user_text, source="user")
            # Otto text juga di-scan tapi dengan source="otto"
            # (sebagian besar rules hanya aktif untuk source="user")
            await self._scanner.scan(otto_text, source="otto")
        except Exception as e:
            logger.warning("[brain] Scanner error (non-fatal): %s", e)


    # ── Public API ───────────────────────────────────────────────────────────

    async def think(
        self,
        user_text: str,
        history: list[dict] | None = None,
    ) -> BrainResponse:
        """
        Kirim pesan ke Groq dan kembalikan BrainResponse.

        Args:
            user_text : Teks dari Rofi (hasil STT atau ketikan)
            history   : Riwayat percakapan [{"role": ..., "content": ...}]
        """
        system_prompt = self._build_system_prompt()
        messages      = self._build_messages(system_prompt, history or [], user_text)

        t0      = time.monotonic()
        raw     = await self._call_groq(messages)
        latency = (time.monotonic() - t0) * 1000

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
        asyncio.create_task(self._evolve_personality("normal"))
        return resp

    async def think_stream(
        self,
        user_text: str,
        history: list[dict] | None = None,
    ):
        """
        Generator async — yield token demi token (untuk streaming ke WebSocket).

        Contoh:
            async for token in brain.think_stream("Hei Otto"):
                await ws.send_text(token)
        """
        system_prompt = self._build_system_prompt()
        messages      = self._build_messages(system_prompt, history or [], user_text)

        payload = {
            "model":    MODEL,
            "messages": messages,
            "stream":   True,
        }

        api_key = self._next_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }

        full_text = []
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

        asyncio.create_task(
            self._log_to_memory(user_text, "".join(full_text))
        )
        asyncio.create_task(
            self._scan_conversation(user_text, "".join(full_text))  # ← TAMBAH
        )
        asyncio.create_task(
            self._consolidator.maybe_consolidate()  # ← TAMBAH
        )
        asyncio.create_task(
                self._evolve_personality("normal")
        )
        

    async def close(self) -> None:
        await self._client.aclose()

    # ── System Prompt ─────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """
        Kembalikan system prompt — dari cache jika long-term memory tidak berubah.

        Cara kerja:
          - Setiap remember() / forget() menaikkan _long_term_version di memory
          - Brain bandingkan versi tersimpan vs versi sekarang
          - Sama → kembalikan cache (skip rebuild string panjang)
          - Beda → rebuild, simpan ke cache
        """
        current_version = self.memory.long_term_version()

        if self._cached_prompt and current_version == self._cached_prompt_version:
            return self._cached_prompt   # cache hit

        # Cache miss — rebuild
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

    async def _call_groq(
        self,
        messages: list[dict],
        retries: int = 3,
    ) -> dict:
        """
        Panggil Groq dengan retry + round-robin key rotation.
        Jika satu key rate-limit → otomatis coba key berikutnya.
        """
        last_exc: Exception | None = None

        for attempt in range(retries * len(self._keys)):
            api_key = self._next_key()
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            }
            payload = {
                "model":       MODEL,
                "messages":    messages,
                "temperature": 0.7,
                "max_tokens":  1024,
            }

            try:
                resp = await self._client.post(GROQ_URL, json=payload, headers=headers)
                if resp.status_code == 429:
                    logger.warning("Key #%d rate-limit, coba key lain…", self._key_idx)
                    last_exc = httpx.HTTPStatusError(
                        "429", request=resp.request, response=resp
                    )
                    continue
                resp.raise_for_status()
                return resp.json()

            except httpx.TimeoutException as e:
                logger.warning("Timeout attempt %d: %s", attempt + 1, e)
                last_exc = e
                await asyncio.sleep(0.5 * (attempt + 1))

            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    logger.warning("Server error attempt %d: %s", attempt + 1, e)
                    last_exc = e
                    await asyncio.sleep(1.0)
                else:
                    raise

        raise RuntimeError(f"Groq gagal setelah {retries} retry: {last_exc}") from last_exc

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
