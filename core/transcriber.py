# core/transcriber.py
# Dual-mode Whisper: small (<= 8 detik) → latency rendah
#                    small (> 8 detik)  → akurasi tinggi

import io
import re
import wave
import tempfile as _tempfile
from pathlib import Path

from faster_whisper import WhisperModel
from core.config import WHISPER
from core.vocabulary import WHISPER_INITIAL_PROMPT, get_alias_map

# ── Threshold durasi (detik) ──────────────────────────────────────────────────
SMALL_THRESHOLD_SEC = 8


def _wav_duration(audio: bytes) -> float:
    """Hitung durasi audio WAV dalam detik. Return 0 jika gagal parse."""
    try:
        with wave.open(io.BytesIO(audio), "rb") as wf:
            frames = wf.getnframes()
            rate   = wf.getframerate()
            return frames / rate if rate > 0 else 0.0
    except Exception:
        return 0.0


class Transcriber:

    def __init__(self):
        # Ambil nama model dari config — konsisten dengan WHISPER dict
        model_command = WHISPER.get("model_command", "small")
        model_chat    = WHISPER.get("model_chat", "small")
        device        = WHISPER.get("device", "cpu")
        compute_type  = WHISPER.get("compute_type", "int8")

        # ── Load model command (kalimat pendek, latency rendah) ───────────
        print(f"[transcriber] Loading Whisper {model_command}...")
        self._small = WhisperModel(
            model_command,
            device           = device,
            compute_type     = compute_type,
            num_workers      = 2,
            cpu_threads      = 4,
            local_files_only = True,
        )
        print(f"[transcriber] Whisper {model_command} siap.")

        # ── Load model chat (kalimat panjang, akurasi lebih baik) ─────────
        print(f"[transcriber] Loading Whisper {model_chat}...")
        self._small = WhisperModel(
            model_chat,
            device           = device,
            compute_type     = compute_type,
            num_workers      = 2,
            cpu_threads      = 4,
            local_files_only = True,
        )
        print(f"[transcriber] Whisper {model_chat} siap.")

    def transcribe(self, audio: bytes) -> str:
        """
        Terima WAV bytes → return teks.
        Otomatis pilih model:
          # durasi <= 8 detik → small  (latency rendah)
          # durasi  > 8 detik → small (akurasi lebih baik)
        """
        if not audio:
            return ""

        durasi = _wav_duration(audio)
        if durasi <= SMALL_THRESHOLD_SEC:
            model = self._small
            mode  = f"small ({durasi:.1f}s)"
        else:
            model = self._small
            mode  = f"small ({durasi:.1f}s)"

        with _tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = Path(f.name)
            f.write(audio)
        try:
            print(f"[transcriber] Pakai {mode}")
            segments, _ = model.transcribe(
                str(tmp),
                language       = WHISPER.get("language", "id"),
                beam_size      = 5,
                initial_prompt = WHISPER_INITIAL_PROMPT,
                vad_filter     = True,
                vad_parameters = {
                    "threshold":               0.25,
                    "min_silence_duration_ms": 200,
                    "speech_pad_ms":           300,
                },
            )
            teks = " ".join(s.text.strip() for s in segments).strip()
            teks = self._normalize_nama(teks)
            print(f"[transcriber] → '{teks}'")
            return teks
        except Exception as e:
            print(f"[transcriber] Error: {e}")
            return ""
        finally:
            tmp.unlink(missing_ok=True)


    def _normalize_nama(self, teks: str) -> str:
        alias = get_alias_map()          # ← baca fresh setiap kali
        kata_kata = teks.split()
        hasil = []
        for i, kata in enumerate(kata_kata):
            kata_bersih = re.sub(r'[^\w]', '', kata.lower())
            if kata_bersih in alias:
                is_awal         = (i == 0)
                is_setelah_tanda = (i > 0 and kata_kata[i-1][-1] in '.,!?')
                if is_awal or is_setelah_tanda:
                    hasil.append(alias[kata_bersih])
                    continue
            hasil.append(kata)
        return " ".join(hasil)


# ── Singleton ─────────────────────────────────────────────────────────────────
_instance: Transcriber | None = None

def get_transcriber() -> Transcriber:
    global _instance
    if _instance is None:
        _instance = Transcriber()
    return _instance
