# core/transcriber.py
# Whisper STT — rekam audio dari PipeWire, ubah jadi teks
#
# Dua mode:
#   "command" → tiny model, cepat, untuk perintah pendek
#   "chat"    → medium model, akurat, untuk ngobrol panjang

import io
import wave
import tempfile
import subprocess
from pathlib import Path

from faster_whisper import WhisperModel

from core.config import WHISPER, AUDIO


class Transcriber:

    def __init__(self):
        self._models: dict[str, WhisperModel] = {}
        print("[transcriber] Loading Whisper tiny...")
        self._models["tiny"] = WhisperModel(
            "tiny", device="cpu", compute_type="int8",
            num_workers=1, cpu_threads=2
        )
        print("[transcriber] Whisper tiny siap.")
        # medium tidak dipreload — terlalu berat

    def _get_model(self, mode: str) -> WhisperModel:
        key = "tiny" if mode == "command" else "medium"
        if key not in self._models:
            print(f"[transcriber] Loading Whisper {key}...")
            self._models[key] = WhisperModel(
                key, device="cpu", compute_type="int8",
                num_workers=1, cpu_threads=3
            )
        return self._models[key]

    # ─── REKAM ────────────────────────────────────────────────────────────────

    def record(self, duration: float = 5.0) -> bytes:
        """
        Rekam audio dari PipeWire selama `duration` detik.
        Return: raw PCM bytes (s16, mono, 16kHz)

        Pakai pw-record karena Otto jalan di Wayland + PipeWire.
        """
        frames = int(AUDIO["sample_rate"] * duration)

        cmd = [
            AUDIO["record_cmd"],           # pw-record
            "--target", str(AUDIO["sink_id"]),
            "--rate",   str(AUDIO["sample_rate"]),
            "--channels", str(AUDIO["channels"]),
            "--format", AUDIO["format"],   # s16
            f"--frames={frames}",
            "-",                           # output ke stdout
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=duration + 3,
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return b""
        except FileNotFoundError:
            raise RuntimeError(
                "pw-record tidak ditemukan. Pastikan PipeWire terinstall."
            )

    def pcm_to_wav(self, pcm: bytes) -> bytes:
        """Bungkus raw PCM ke WAV bytes agar bisa dibaca Whisper."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(AUDIO["channels"])
            wf.setsampwidth(2)  # s16 = 2 bytes
            wf.setframerate(AUDIO["sample_rate"])
            wf.writeframes(pcm)
        return buf.getvalue()

    # ─── TRANSKRIPSI ──────────────────────────────────────────────────────────

    def transcribe(self, audio: bytes, mode: str = "command") -> str:
        """
        Transkripsi audio (WAV bytes atau raw PCM) ke teks.

        mode: "command" → model tiny, cepat
              "chat"    → model medium, akurat

        Return: teks hasil transkripsi, atau "" jika gagal/kosong
        """
        if not audio:
            return ""

        # Whisper butuh file — tulis ke tempfile lalu hapus setelah selesai
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            if audio[:4] != b"RIFF":
                # Bukan WAV — convert via ffmpeg (handle WebM/Opus dari browser)
                webm_tmp = tmp_path.with_suffix(".webm")
                webm_tmp.write_bytes(audio)
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(webm_tmp),
                    "-ar", "16000", "-ac", "1", "-f", "wav", str(tmp_path)
                ], capture_output=True)
                webm_tmp.unlink(missing_ok=True)
            else:
                tmp.write(audio)

        try:
            model = self._get_model(mode)
            segments, info = model.transcribe(
                str(tmp_path),
                language=WHISPER["language"],
                beam_size=5,
                vad_filter=True,           # skip bagian sunyi
                vad_parameters={
                    "min_silence_duration_ms": 500,
                },
            )
            teks = " ".join(seg.text.strip() for seg in segments)
            return teks.strip()
        except Exception as e:
            print(f"[transcriber] Error: {e}")
            return ""
        finally:
            tmp_path.unlink(missing_ok=True)

    # ─── SHORTCUT ─────────────────────────────────────────────────────────────

    def dengarkan(self, duration: float = 5.0, mode: str = "command") -> str:
        """
        Shortcut: rekam → transkripsi → return teks.
        Ini yang dipanggil dari server/websocket.py.

        Contoh:
            teks = transcriber.dengarkan(duration=4.0, mode="command")
            # → "Otto putar lagu"
        """
        pcm = self.record(duration=duration)
        return self.transcribe(pcm, mode=mode)


# Singleton
transcriber = Transcriber()
