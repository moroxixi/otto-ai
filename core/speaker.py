# core/speaker.py
"""
Speaker Otto — TTS dengan Kokoro ONNX + Piper PCM Streaming
=============================================================
Mode output:
  - synthesize(text)              → return bytes WAV (fallback / non-streaming)
  - speak_local(text)             → putar langsung di speaker laptop (pw-play)
  - synthesize_or_speak(...)      → routing ws / laptop / both
  - stream_to_ws(ws, text)        → stream raw PCM chunks ke WebSocket (seperti Libo)
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

# ── Konstanta PCM streaming ───────────────────────────────────────────────────
PIPER_SAMPLE_RATE   = 22050
PIPER_CHANNELS      = 1
PIPER_SAMPLE_WIDTH  = 2          # 16-bit PCM
CHUNK_BYTES         = int(PIPER_SAMPLE_RATE * PIPER_CHANNELS * PIPER_SAMPLE_WIDTH * 0.5)
PREBUFFER_CHUNKS    = 3
SILENCE_PADDING_MS  = 1200


class Speaker:
    def __init__(self):
        from core.config import KOKORO, AUDIO, PIPER
        from kokoro_onnx import Kokoro

        self._piper_bin    = str(PIPER["binary"])
        self._piper_model  = str(PIPER["model"])

        model_path  = str(KOKORO["model"])
        voices_path = str(KOKORO["voices"])

        self._kokoro   = Kokoro(model_path, voices_path)
        self._voice    = KOKORO["voice"]
        self._speed    = KOKORO["speed"]
        self._lang     = KOKORO["lang"]
        self._play_cmd = AUDIO.get("play_cmd", "pw-play")
        self._sink_id  = AUDIO.get("sink_id", 58)

        logger.info(
            "Kokoro TTS siap — voice=%s speed=%.1f lang=%s",
            self._voice, self._speed, self._lang
        )
        self._active_procs: list[subprocess.Popen] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""
        samples, sr = self._generate(text)
        return self._to_wav_bytes(samples, sr)

    def speak_local(self, text: str) -> None:
        if not text.strip():
            return
        self._gc_procs()
        samples, sr = self._generate(text)
        wav_bytes = self._to_wav_bytes(samples, sr)
        self._play_bytes_local(wav_bytes)

    async def speak_local_async(self, text: str) -> None:
        await asyncio.to_thread(self.speak_local, text)

    def synthesize_or_speak(self, text: str, output: str = "ws") -> bytes:
        if not text.strip():
            return b""
        samples, sr = self._generate(text)
        wav_bytes = self._to_wav_bytes(samples, sr)
        if output in ("laptop", "both"):
            self._gc_procs()
            self._play_bytes_local(wav_bytes)
        if output in ("ws", "both"):
            return wav_bytes
        return b""

    async def stream_to_ws(self, websocket, text: str) -> None:
        """
        Stream raw PCM dari Piper langsung ke WebSocket.

        Protocol:
          → send_json {"type": "audio_stream_start"}
          → send_bytes <pcm_chunk> × N
          → send_bytes <silence_padding>
          → send_json {"type": "audio_stream_end"}
        """
        if not text.strip():
            return

        logger.info("[speaker] Stream PCM TTS: '%s'", text[:60])

        proc = await asyncio.create_subprocess_exec(
            self._piper_bin,
            "--model",           self._piper_model,
            "--output-raw",
            "--sentence-silence", "0.3",
            stdin  = asyncio.subprocess.PIPE,
            stdout = asyncio.subprocess.PIPE,
            stderr = asyncio.subprocess.DEVNULL,
        )

        proc.stdin.write(text.encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()

        prebuffer_list: list[bytes] = []
        stream_started = False
        chunk_count    = 0

        try:
            while True:
                pcm = await proc.stdout.read(CHUNK_BYTES)
                if not pcm:
                    break

                if not stream_started:
                    prebuffer_list.append(pcm)
                    if len(prebuffer_list) >= PREBUFFER_CHUNKS:
                        await websocket.send_json({"type": "audio_stream_start"})
                        stream_started = True
                        for chunk in prebuffer_list:
                            await websocket.send_bytes(chunk)
                            chunk_count += 1
                        prebuffer_list = []
                else:
                    await websocket.send_bytes(pcm)
                    chunk_count += 1

            # Flush sisa prebuffer kalau teks sangat pendek
            if not stream_started:
                await websocket.send_json({"type": "audio_stream_start"})
                for chunk in prebuffer_list:
                    await websocket.send_bytes(chunk)
                    chunk_count += 1

            # Silence padding — kata terakhir tidak terpotong
            silence_frames = int(PIPER_SAMPLE_RATE * SILENCE_PADDING_MS / 1000)
            silence_bytes  = b'\x00' * (silence_frames * PIPER_CHANNELS * PIPER_SAMPLE_WIDTH)
            for i in range(0, len(silence_bytes), CHUNK_BYTES):
                await websocket.send_bytes(silence_bytes[i:i + CHUNK_BYTES])

        finally:
            await proc.wait()
            await websocket.send_json({"type": "audio_stream_end"})
            logger.info(
                "[speaker] Stream selesai: %d chunk + %dms silence.",
                chunk_count, SILENCE_PADDING_MS
            )

    async def ucapkan_laptop_async(self, text: str) -> None:
        if not text.strip():
            return

        import uuid
        wav_path = f"/tmp/otto_laptop_{uuid.uuid4().hex[:8]}.wav"
        try:
            proc_piper = await asyncio.create_subprocess_exec(
                self._piper_bin,
                "--model", self._piper_model,
                "--output_file", wav_path,
                stdin  = asyncio.subprocess.PIPE,
                stdout = asyncio.subprocess.DEVNULL,
                stderr = asyncio.subprocess.DEVNULL,
            )
            await proc_piper.communicate(text.encode("utf-8"))

            if not Path(wav_path).exists() or Path(wav_path).stat().st_size == 0:
                logger.warning("[speaker] Piper gagal buat WAV untuk laptop.")
                return

            proc_play = await asyncio.create_subprocess_exec(
                self._play_cmd,
                "--target", str(self._sink_id),
                wav_path,
                stdout = asyncio.subprocess.DEVNULL,
                stderr = asyncio.subprocess.DEVNULL,
            )
            await proc_play.wait()
        except Exception as e:
            logger.error("[speaker] ucapkan_laptop_async gagal: %s", e)
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def set_voice(self, voice: str) -> None:
        self._voice = voice
        logger.info("Voice diganti ke: %s", voice)

    def list_voices(self) -> list[str]:
        try:
            return self._kokoro.get_voices()
        except Exception:
            return []

    # ── Internal ──────────────────────────────────────────────────────────────

    def _gc_procs(self) -> None:
        before = len(self._active_procs)
        self._active_procs = [p for p in self._active_procs if p.poll() is None]
        cleaned = before - len(self._active_procs)
        if cleaned:
            logger.debug("[speaker] GC: %d proses lama dibersihkan.", cleaned)

    def _generate(self, text: str):
        """Kokoro → (samples: np.ndarray, sample_rate: int)."""
        try:
            return self._kokoro.create(
                text,
                voice = self._voice,
                speed = self._speed,
                lang  = self._lang,
            )
        except Exception as e:
            logger.error("Kokoro gagal generate audio: %s", e)
            raise

    def _to_wav_bytes(self, samples: np.ndarray, sr: int) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()

    def _play_bytes_local(self, wav_bytes: bytes) -> None:
        """
        FIX BUG 1: tambahkan --target self._sink_id agar audio
        selalu keluar ke sink yang benar (sink_id 58), konsisten
        dengan ucapkan_laptop_async.
        """
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                tmp_path = f.name
            proc = subprocess.Popen(
                [self._play_cmd, "--target", str(self._sink_id), tmp_path],
                stdout = subprocess.DEVNULL,
                stderr = subprocess.PIPE,
            )
            self._active_procs.append(proc)
            proc.wait()
            self._gc_procs()
        except Exception as e:
            logger.error("Gagal putar audio lokal: %s", e)
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def shutdown(self) -> None:
        killed = 0
        for proc in self._active_procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                killed += 1
        self._active_procs.clear()
        if killed:
            logger.info("[speaker] %d subprocess audio dihentikan saat shutdown.", killed)
