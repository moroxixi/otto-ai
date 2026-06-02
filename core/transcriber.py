# core/transcriber.py
# Dual-mode Whisper: small (<= 3.5 detik) → latency rendah
#                   medium (> 3.5 detik)  → akurasi tinggi

import io
import re
import wave
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel
from core.config import WHISPER
from core.vocabulary import WHISPER_INITIAL_PROMPT, NAMA_ALIAS

# ── Threshold durasi (detik) ──────────────────────────────────────────────────
TINY_THRESHOLD_SEC = 3.5


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
        model_chat    = WHISPER.get("model_chat", "medium")
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
        self._medium = WhisperModel(
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
          - durasi <= 3.5 detik → small  (kalimat singkat, latency rendah)
          - durasi  > 3.5 detik → medium (kalimat panjang, akurasi lebih baik)
        """
        if not audio:
            return ""

        durasi = _wav_duration(audio)
        if durasi <= TINY_THRESHOLD_SEC:
            model = self._small
            mode  = f"small ({durasi:.1f}s)"
        else:
            model = self._medium
            mode  = f"medium ({durasi:.1f}s)"

        tmp = Path(tempfile.mktemp(suffix=".wav"))
        try:
            tmp.write_bytes(audio)
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
        kata_kata = teks.split()
        hasil = []
        for i, kata in enumerate(kata_kata):
            kata_bersih = re.sub(r'[^\w]', '', kata.lower())
            if kata_bersih in NAMA_ALIAS:
                is_awal          = (i == 0)
                is_setelah_tanda = (i > 0 and kata_kata[i-1][-1] in '.,!?')
                if is_awal or is_setelah_tanda:
                    hasil.append(NAMA_ALIAS[kata_bersih])
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
