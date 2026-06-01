# core/transcriber.py — versi ringkas (setelah frontend kirim WAV)

import io
import re
import wave
import base64
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel
from core.config import WHISPER
from core.vocabulary import WHISPER_INITIAL_PROMPT, NAMA_ALIAS


class Transcriber:

    def __init__(self):
        print("[transcriber] Loading Whisper medium...")
        self._model = WhisperModel(
            "medium", device="cpu", compute_type="int8",
            num_workers=2, cpu_threads=4, local_files_only=True
        )
        print("[transcriber] Whisper medium siap.")

    def transcribe(self, audio: bytes) -> str:
        """Terima WAV bytes → return teks."""
        if not audio:
            return ""

        tmp = Path(tempfile.mktemp(suffix=".wav"))
        try:
            tmp.write_bytes(audio)
            segments, _ = self._model.transcribe(
                str(tmp),
                language=WHISPER["language"],
                beam_size=5,
                initial_prompt=WHISPER_INITIAL_PROMPT,
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.25,
                    "min_silence_duration_ms": 200,
                    "speech_pad_ms": 300,
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
                is_awal = (i == 0)
                is_setelah_tanda = (i > 0 and kata_kata[i-1][-1] in '.,!?')
                if is_awal or is_setelah_tanda:
                    hasil.append(NAMA_ALIAS[kata_bersih])
                    continue
            hasil.append(kata)
        return " ".join(hasil)


_instance: Transcriber | None = None

def get_transcriber() -> Transcriber:
    global _instance
    if _instance is None:
        _instance = Transcriber()
    return _instance
