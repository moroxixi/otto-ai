# core/
Fondasi Otto — tidak bisa jalan tanpa semua file di sini.

| File | Fungsi |
|---|---|
| config.py | Semua konstanta, path, API key rotation |
| memory.py | Ingatan jangka pendek & panjang |
| transcriber.py | Whisper STT — suara → teks |
| speaker.py | Piper TTS — teks → suara |
| brain.py | LLM engine (Groq) |
| executor.py | Dispatcher aksi dari intent |
