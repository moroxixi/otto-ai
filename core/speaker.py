# core/speaker.py
"""
Speaker Otto — TTS dengan Kokoro ONNX
======================================
Mode output:
  - synthesize(text)          → return bytes WAV (untuk dikirim ke iPhone via WS)
  - speak_local(text)         → putar langsung di speaker laptop (pw-play)
  - synthesize_or_speak(text, output="ws"|"laptop"|"both")
"""

from __future__ import annotations

import asyncio
import io
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger("otto.speaker")


class Speaker:
    def __init__(self):
        from core.config import KOKORO, AUDIO
        from kokoro_onnx import Kokoro

        model_path  = str(KOKORO["model"])
        voices_path = str(KOKORO["voices"])

        self._kokoro = Kokoro(model_path, voices_path)
        self._voice  = KOKORO["voice"]
        self._speed  = KOKORO["speed"]
        self._lang   = KOKORO["lang"]
        self._play_cmd = AUDIO.get("play_cmd", "pw-play")

        logger.info(
            "Kokoro TTS siap — voice=%s speed=%.1f lang=%s",
            self._voice, self._speed, self._lang
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def synthesize(self, text: str) -> bytes:
        """
        Generate audio → return WAV bytes.
        Dipakai untuk kirim ke iPhone via WebSocket.
        """
        if not text.strip():
            return b""
        samples, sr = self._generate(text)
        return self._to_wav_bytes(samples, sr)

    def speak_local(self, text: str) -> None:
        """
        Generate audio → putar langsung via pw-play di laptop.
        Dipakai untuk output proaktif Otto (bukan lewat iPhone).
        """
        if not text.strip():
            return
        samples, sr = self._generate(text)
        wav_bytes = self._to_wav_bytes(samples, sr)
        self._play_bytes_local(wav_bytes)

    async def speak_local_async(self, text: str) -> None:
        """Async wrapper untuk speak_local — agar tidak block event loop."""
        await asyncio.to_thread(self.speak_local, text)

    def synthesize_or_speak(
        self,
        text: str,
        output: str = "ws",
    ) -> bytes:
        """
        output="ws"     → return WAV bytes (untuk WebSocket)
        output="laptop" → putar lokal, return b""
        output="both"   → putar lokal DAN return WAV bytes
        """
        if not text.strip():
            return b""

        samples, sr = self._generate(text)
        wav_bytes = self._to_wav_bytes(samples, sr)

        if output in ("laptop", "both"):
            self._play_bytes_local(wav_bytes)

        if output in ("ws", "both"):
            return wav_bytes

        return b""

    def set_voice(self, voice: str) -> None:
        """Ganti voice on-the-fly."""
        self._voice = voice
        logger.info("Voice diganti ke: %s", voice)

    def list_voices(self) -> list[str]:
        """Daftar semua voice yang tersedia di file bin."""
        try:
            return self._kokoro.get_voices()
        except Exception:
            return []

    # ── Internal ──────────────────────────────────────────────────────────────

    def _generate(self, text: str):
        """Panggil Kokoro → return (samples: np.ndarray, sample_rate: int)."""
        try:
            samples, sr = self._kokoro.create(
                text,
                voice=self._voice,
                speed=self._speed,
                lang=self._lang,
            )
            return samples, sr
        except Exception as e:
            logger.error("Kokoro gagal generate audio: %s", e)
            raise

    def _to_wav_bytes(self, samples: np.ndarray, sr: int) -> bytes:
        """Convert numpy samples → WAV bytes in-memory."""
        buf = io.BytesIO()
        sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()

    def _play_bytes_local(self, wav_bytes: bytes) -> None:
        """
        Putar WAV bytes langsung via pw-play.
        Tulis ke temp file dulu karena pw-play butuh file path.
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                tmp_path = f.name

            subprocess.run(
                [self._play_cmd, tmp_path],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error("pw-play error: %s", e.stderr.decode())
        except Exception as e:
            logger.error("Gagal putar audio lokal: %s", e)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
