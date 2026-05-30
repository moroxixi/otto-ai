"""
core/brain.py — Otak Otto
=========================
Tanggung jawab:
  - Kirim pesan ke Groq LLM (llama-3.1-8b-instant / llama-3.3-70b-versatile)
  - Round-robin rotation 6 API key otomatis
  - Tentukan model mana yang dipakai (fast vs deep)
  - Bangun system prompt yang menyertakan profil Rofi dari memory
  - Deteksi intent: perintah cepat vs obrolan vs analisis profil
  - Kembalikan respons + metadata (model, key_index, tokens)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

from core.config import GROQ_API_KEYS, DEBUG
from core.memory import MemoryManager
from otto_self.model import self_summary_text

logger = logging.getLogger("otto.brain")


# ─────────────────────────── Tipe & Konstanta ───────────────────────────────

class BrainMode(Enum):
    FAST   = "fast"    # llama-3.1-8b-instant   → perintah & aksi
    DEEP   = "deep"    # llama-3.3-70b-versatile → ngobrol & profil
    AUTO   = "auto"    # tentukan otomatis dari teks


MODEL_FAST = "llama-3.1-8b-instant"
MODEL_DEEP = "llama-3.3-70b-versatile"

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Kata kunci yang mendorong pemilihan model DEEP
_DEEP_TRIGGERS = {
    "kenapa", "menurutmu", "apa pendapat", "analisis", "cerita", "jelaskan",
    "bagaimana perasaan", "saran", "rekomendasi", "menurutmu", "pikir",
    "opini", "bandingkan", "hubungan", "pola", "kebiasaan",
}

# System prompt dasar — Otto tidak hardcode fakta Rofi, hanya terima dari memory
_BASE_SYSTEM = """\
Kamu adalah Otto, asisten AI pribadi milik Rofi yang berjalan lokal di rumahnya.
{self_section}
Kepribadian:
- Bicara natural, santai, seperti teman — bukan asisten korporat
- Bahasa Indonesia campuran ringan (boleh sedikit Inggris teknis jika perlu)
- Proaktif: jika kamu punya hipotesis tentang Rofi, tanya — jangan simpan sendiri
- Jujur: jika tidak tau, katakan. Jangan karang.
- Singkat jika perintah, lebih dalam jika obrolan

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
    mode: BrainMode
    key_index: int
    prompt_tokens: int  = 0
    completion_tokens: int = 0
    latency_ms: float  = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────── Kelas Utama ────────────────────────────────────

class Brain:
    """
    Otak Otto. Instantiate sekali, pakai terus.

    Contoh:
        brain = Brain(memory)
        resp  = await brain.think("Otto, matiin lampu dong")
        print(resp.text)
    """

    def __init__(self, memory: MemoryManager) -> None:
        self.memory    = memory
        self._key_idx  = 0          # pointer round-robin
        self._keys     = self._load_keys()
        self._client   = httpx.AsyncClient(timeout=30.0)
        logger.info("Brain siap. %d API key tersedia.", len(self._keys))

    # ── Public API ───────────────────────────────────────────────────────────

    async def think(
        self,
        user_text: str,
        history:   list[dict] | None = None,
        mode:      BrainMode = BrainMode.AUTO,
        force_model: str | None = None,
    ) -> BrainResponse:
        """
        Kirim pesan ke Groq dan kembalikan BrainResponse.

        Args:
            user_text   : Teks dari Rofi (sudah di-transcribe atau diketik)
            history     : Riwayat percakapan [{"role": ..., "content": ...}]
            mode        : FAST / DEEP / AUTO
            force_model : Override model string (opsional)
        """
        resolved_mode  = self._resolve_mode(user_text, mode)
        model          = force_model or self._pick_model(resolved_mode)
        system_prompt  = self._build_system_prompt()
        messages       = self._build_messages(system_prompt, history or [], user_text)

        t0             = time.monotonic()
        raw            = await self._call_groq(model, messages)
        latency        = (time.monotonic() - t0) * 1000

        text           = self._extract_text(raw)
        usage          = raw.get("usage", {})

        resp = BrainResponse(
            text              = text,
            model             = model,
            mode              = resolved_mode,
            key_index         = self._key_idx,
            prompt_tokens     = usage.get("prompt_tokens", 0),
            completion_tokens = usage.get("completion_tokens", 0),
            latency_ms        = round(latency, 1),
            raw               = raw,
        )

        logger.debug(
            "[brain] mode=%s model=%s key=%d tokens=%d+%d latency=%.0fms",
            resolved_mode.value, model, self._key_idx,
            resp.prompt_tokens, resp.completion_tokens, latency,
        )

        # Simpan percakapan ini ke memory (non-blocking)
        asyncio.create_task(self._log_to_memory(user_text, text))

        return resp

    async def think_stream(
        self,
        user_text: str,
        history:   list[dict] | None = None,
        mode:      BrainMode = BrainMode.AUTO,
    ):
        """
        Generator async — yield token demi token (untuk streaming ke WebSocket).

        Contoh:
            async for token in brain.think_stream("Hei Otto"):
                await ws.send_text(token)
        """
        resolved_mode = self._resolve_mode(user_text, mode)
        model         = self._pick_model(resolved_mode)
        system_prompt = self._build_system_prompt()
        messages      = self._build_messages(system_prompt, history or [], user_text)

        payload = {
            "model":    model,
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

    async def close(self) -> None:
        await self._client.aclose()

    # ── Mode & Model ─────────────────────────────────────────────────────────

    def _resolve_mode(self, text: str, mode: BrainMode) -> BrainMode:
        if mode != BrainMode.AUTO:
            return mode
        low = text.lower()
        if any(kw in low for kw in _DEEP_TRIGGERS):
            return BrainMode.DEEP
        # Kalimat pendek → fast
        if len(text.split()) <= 8:
            return BrainMode.FAST
        return BrainMode.DEEP

    def _pick_model(self, mode: BrainMode) -> str:
        return MODEL_FAST if mode == BrainMode.FAST else MODEL_DEEP

    # ── System Prompt ─────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        profile_summary = self.memory.summary_for_llm(max_items=15)
        profile_sec = _PROFILE_SECTION.format(profile_json=profile_summary) if profile_summary else _NO_PROFILE

        otto_self = self_summary_text()   # sekarang benar-benar dipakai
        self_sec  = f"Tentang dirimu:\n{otto_self}" if otto_self else ""

        return _BASE_SYSTEM.format(
            self_section=self_sec,
            profile_section=profile_sec,
        ).strip()

    # ── Message Builder ───────────────────────────────────────────────────────

    @staticmethod
    def _build_messages(
        system: str,
        history: list[dict],
        user_text: str,
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": system}]
        # Batasi history agar tidak meledak konteks: ambil 20 pesan terakhir
        messages.extend(history[-20:])
        messages.append({"role": "user", "content": user_text})
        return messages

    # ── Groq Call ─────────────────────────────────────────────────────────────

    async def _call_groq(
        self,
        model:    str,
        messages: list[dict],
        retries:  int = 3,
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
                "model":       model,
                "messages":    messages,
                "temperature": 0.7,
                "max_tokens":  1024,
            }

            try:
                resp = await self._client.post(GROQ_URL, json=payload, headers=headers)
                if resp.status_code == 429:
                    # Rate limit → coba key lain
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
        """Simpan pasangan percakapan ke short-term memory."""
        try:
            await asyncio.to_thread(self.memory.add_message, "user", user_text)
            await asyncio.to_thread(self.memory.add_message, "assistant", otto_text)
        except Exception as e:
            logger.warning("Gagal log ke memory: %s", e)


# ─────────────────────────── Quick Test ─────────────────────────────────────

if __name__ == "__main__":
    import sys
    from core.memory import MemoryManager

    logging.basicConfig(level=logging.DEBUG)

    async def _test():
        mem   = MemoryManager()
        brain = Brain(mem)

        test_cases = [
            ("Otto, matiin lampu dong",           BrainMode.AUTO),
            ("Menurutmu kenapa aku susah bangun pagi?", BrainMode.AUTO),
            ("Jam berapa sekarang?",               BrainMode.FAST),
        ]

        for text, mode in test_cases:
            print(f"\n[INPUT] {text}")
            resp = await brain.think(text, mode=mode)
            print(f"[MODEL] {resp.model} | latency={resp.latency_ms}ms | tokens={resp.prompt_tokens}+{resp.completion_tokens}")
            print(f"[OTTO]  {resp.text}")

        await brain.close()

    asyncio.run(_test())
