# core/speaker.py
# Piper TTS — ubah teks jadi suara, putar via PipeWire
#
# Pipeline: teks → piper (generate WAV) → pw-play (putar ke speaker)
# Semua via subprocess agar ringan dan tidak perlu binding Python

import subprocess
import tempfile
import asyncio
from pathlib import Path

from core.config import PIPER, AUDIO


class Speaker:

    def __init__(self):
        self._piper_bin  = Path(PIPER["binary"])
        self._model      = Path(PIPER["model"])
        self._config     = Path(PIPER["config"])
        self._is_speaking = False  # flag agar tidak tumpang tindih
        self._check_piper()

    def _check_piper(self) -> None:
        """Cek binary dan model ada saat startup."""
        if not self._piper_bin.exists():
            raise FileNotFoundError(
                f"Piper tidak ditemukan di {self._piper_bin}. "
                "Install dengan: sudo cp piper /usr/local/bin/piper"
            )
        if not self._model.exists():
            raise FileNotFoundError(
                f"Model TTS tidak ditemukan: {self._model}"
            )

    # ─── GENERATE AUDIO ───────────────────────────────────────────────────────

    def _generate_wav(self, teks: str, output_path: Path) -> bool:
        """
        Jalankan Piper untuk generate WAV dari teks.
        Return True jika berhasil.
        """
        cmd = [
            str(self._piper_bin),
            "--model",  str(self._model),
            "--config", str(self._config),
            "--output_file", str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                input=teks.encode("utf-8"),
                capture_output=True,
                timeout=30,
            )
            return result.returncode == 0 and output_path.exists()
        except subprocess.TimeoutExpired:
            print("[speaker] Piper timeout — teks terlalu panjang?")
            return False
        except Exception as e:
            print(f"[speaker] Error generate WAV: {e}")
            return False

    # ─── PUTAR AUDIO ──────────────────────────────────────────────────────────

    def _play_wav(self, wav_path: Path) -> None:
        """Putar WAV ke speaker. Coba pw-play dulu, fallback ke aplay."""
        try:
            result = subprocess.run(
                [AUDIO["play_cmd"], "--target", str(AUDIO["sink_id"]), str(wav_path)],
                capture_output=True, timeout=60
            )
            if result.returncode == 0:
                return
            raise RuntimeError(f"pw-play returncode {result.returncode}")
        except Exception as e:
            print(f"[speaker] pw-play gagal ({e}), fallback ke aplay")
            try:
                subprocess.run(
                    ["aplay", str(wav_path)],
                    capture_output=True, timeout=60
                )
            except Exception as e2:
                print(f"[speaker] aplay juga gagal: {e2}")

    # ─── PUBLIC API ───────────────────────────────────────────────────────────

    def synthesize(self, teks: str) -> bytes:
        """
        Generate WAV dari teks → return bytes.
        Dipakai app.py untuk kirim audio ke client via WebSocket.

        Contoh:
            wav_bytes = speaker.synthesize("Halo, Rofi.")
            audio_b64 = base64.b64encode(wav_bytes).decode()
        """
        if not teks.strip():
            return b""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            if self._generate_wav(teks, tmp_path):
                return tmp_path.read_bytes()
            return b""
        finally:
            tmp_path.unlink(missing_ok=True)

    def bicara(self, teks: str) -> None:
        """
        Sinkron: generate WAV + putar. Blok sampai selesai.
        Cocok untuk perintah singkat yang perlu selesai dulu.

        Contoh:
            speaker.bicara("Oke, memutar lagu.")
        """
        if not teks.strip():
            return

        self._is_speaking = True
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            if self._generate_wav(teks, tmp_path):
                self._play_wav(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
            self._is_speaking = False

    async def bicara_async(self, teks: str) -> None:
        """
        Async versi bicara() — agar server tidak block saat Otto bicara.
        Dipanggil dari websocket handler.

        Contoh:
            await speaker.bicara_async("Selamat pagi, Rofi.")
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.bicara, teks)

    def stop(self) -> None:
        """
        Hentikan bicara jika sedang berlangsung.
        Dipakai saat Rofi interupsi Otto di tengah kalimat.
        """
        if self._is_speaking:
            # Kill pw-play yang sedang jalan
            subprocess.run(
                ["pkill", "-f", "pw-play"],
                capture_output=True
            )
            self._is_speaking = False

    @property
    def sedang_bicara(self) -> bool:
        return self._is_speaking


# Singleton
speaker = Speaker()
