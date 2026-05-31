# core/transcriber.py
# Whisper STT — rekam audio dari PipeWire, ubah jadi teks
#
# Strategi: selalu pakai medium untuk akurasi maksimal.
# Otto fokus mengobrol, bukan perintah singkat.

import io
import re
import wave
import tempfile
import subprocess
from pathlib import Path

from faster_whisper import WhisperModel

from core.config import WHISPER, AUDIO
from core.vocabulary import WHISPER_INITIAL_PROMPT, NAMA_ALIAS


#   "command" → sama dengan medium (tiny sudah dihapus)
#   "chat"    → paksa medium


class Transcriber:

    def __init__(self):
        self._models: dict[str, WhisperModel] = {}
        print("[transcriber] Loading Whisper medium...")
        self._models["medium"] = WhisperModel(
            "medium", device="cpu", compute_type="int8", num_workers=2, cpu_threads=4, local_files_only=True 
        )
        print("[transcriber] Whisper medium siap.")

        

    # ─── INTERNAL HELPERS ─────────────────────────────────────────────────────

    def _durasi_wav(self, wav_bytes: bytes) -> float:
        """
        Baca durasi audio dari WAV bytes tanpa decode penuh.
        Hampir 0ms — hanya baca header WAV.
        Return: durasi dalam detik, atau 99.0 jika gagal (→ pakai medium)
        """
        try:
            buf = io.BytesIO(wav_bytes)
            with wave.open(buf, "rb") as w:
                return w.getnframes() / w.getframerate()
        except Exception:
            return 99.0  # fallback → medium

    def _pilih_model(self, durasi: float) -> tuple[WhisperModel, str]:
    # tiny sudah dihapus — selalu pakai medium
        return self._models["medium"], "medium"

    def _normalize_nama(self, teks: str) -> str:
        """
        Ganti variasi nama salah tangkap Whisper → nama yang benar.
        Hanya di awal kalimat atau setelah tanda baca, bukan di tengah kalimat.
        
        Contoh: "auto putar lagu" → "Otto putar lagu"
                "aku suka oto" → tidak diubah (di tengah kalimat)
        """
        kata_kata = teks.split()
        hasil = []
        for i, kata in enumerate(kata_kata):
            kata_bersih = re.sub(r'[^\w]', '', kata.lower())
            if kata_bersih in NAMA_ALIAS:
                is_awal = (i == 0)
                is_setelah_tanda = (i > 0 and kata_kata[i - 1][-1] in '.,!?')
                if is_awal or is_setelah_tanda:
                    hasil.append(NAMA_ALIAS[kata_bersih])
                    continue
            hasil.append(kata)
        return " ".join(hasil)

    # ─── REKAM ────────────────────────────────────────────────────────────────

    def record(self, duration: float = 5.0) -> bytes:
        """
        Rekam audio dari PipeWire selama `duration` detik.
        Return: raw PCM bytes (s16, mono, 16kHz)
        """
        frames = int(AUDIO["sample_rate"] * duration)

        cmd = [
            AUDIO["record_cmd"],
            "--target", str(AUDIO["sink_id"]),
            "--rate",   str(AUDIO["sample_rate"]),
            "--channels", str(AUDIO["channels"]),
            "--format", AUDIO["format"],   # s16
            f"--frames={frames}",
            "-",
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
            wf.setsampwidth(2)          # s16 = 2 bytes
            wf.setframerate(AUDIO["sample_rate"])
            wf.writeframes(pcm)
        return buf.getvalue()

    # ─── TRANSKRIPSI ──────────────────────────────────────────────────────────

    def transcribe(self, audio: bytes, mode: str = "auto") -> str:
        """
        Transkripsi audio ke teks.

        mode: "auto"    → pilih model berdasarkan durasi (REKOMENDASI)
              "command" → paksa tiny  (jika pemanggil yakin audio pendek)
              "chat"    → paksa medium (jika pemanggil yakin audio panjang)

        Return: teks hasil transkripsi, atau "" jika gagal/kosong.
        """
        if not audio:
            return ""

        # Konversi PCM → WAV jika belum (Whisper butuh WAV)
        if audio[:4] != b"RIFF":
            wav_audio = self.pcm_to_wav(audio)
        else:
            wav_audio = audio

        # Tulis ke tempfile (Whisper butuh path file, bukan bytes)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

            # Kalau WebM/Opus dari browser → convert dulu via ffmpeg
            if audio[:4] not in (b"RIFF", b"\x00\x00\x00\x00"):
                webm_tmp = tmp_path.with_suffix(".webm")
                webm_tmp.write_bytes(audio)
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(webm_tmp),
                    "-ar", "16000", "-ac", "1", "-f", "wav", str(tmp_path)
                ], capture_output=True)
                webm_tmp.unlink(missing_ok=True)
                # Re-baca WAV hasil konversi untuk deteksi durasi
                wav_audio = tmp_path.read_bytes()
            else:
                tmp_path.write_bytes(wav_audio)

        try:
            # Pilih model
            if mode == "command":
                model, label = self._models["medium"], "medium" 
            elif mode == "chat":
                model, label = self._models["medium"], "medium"
            else:
                # "auto" → keputusan berdasarkan durasi audio
                durasi = self._durasi_wav(wav_audio)
                model, label = self._pilih_model(durasi)
                print(f"[transcriber] Durasi {durasi:.1f}s → model [{label}]")

            segments, _ = model.transcribe(
                str(tmp_path),
                language=WHISPER["language"],
                beam_size=5,
                initial_prompt=WHISPER_INITIAL_PROMPT,  # ← bias ke kosakata Otto
                vad_filter=True,           # skip bagian sunyi → hemat komputasi
                vad_parameters={
                    "min_silence_duration_ms": 500,
                },
            )
            semua_segment = list(segments)
            teks = " ".join(seg.text.strip() for seg in semua_segment).strip()
            teks = self._normalize_nama(teks)

            print(f"[transcriber] [{label}] → '{teks}'")
            return teks

        except Exception as e:
            print(f"[transcriber] Error: {e}")
            return ""
        finally:
            tmp_path.unlink(missing_ok=True)

    # ─── SHORTCUT ─────────────────────────────────────────────────────────────

    def dengarkan(self, duration: float = 5.0, mode: str = "auto") -> str:
        """
        Shortcut: rekam → transkripsi → return teks.
        Ini yang dipanggil dari server/app.py via asyncio.to_thread

        mode default diubah ke "auto" — model dipilih otomatis
        berdasarkan durasi audio yang direkam.

        Contoh:
            teks = transcriber.dengarkan(duration=4.0)
            # → "Halo Otto"   (pakai medium karena < 3.5s)

            teks = transcriber.dengarkan(duration=10.0)
            # → teks panjang...     (pakai medium karena > 3.5s)
        """
        pcm = self.record(duration=duration)
        return self.transcribe(pcm, mode=mode)


# Singleton — dipakai di seluruh proyek
transcriber = Transcriber()
