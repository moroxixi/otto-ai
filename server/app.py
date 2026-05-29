"""
server/app.py — Entry Point Otto
==================================
Alur lengkap:
  iPhone (WebSocket) → app.py
      → transcriber.transcribe(audio_bytes)   [Whisper]
      → executor.dispatch(teks)               [brain / skill]
      → speaker.synthesize(result.text)       [Piper TTS]
      → kirim audio balik ke iPhone

Koneksi: wss://<ip>:8000/ws
Protocol pesan (JSON):
  Client → Server:
    { "type": "audio",  "data": "<base64 PCM>" }
    { "type": "text",   "data": "teks langsung" }
    { "type": "ping" }

  Server → Client:
    { "type": "transcript", "data": "teks hasil STT" }
    { "type": "response",   "data": "teks Otto",  "audio": "<base64 WAV>" }
    { "type": "error",      "data": "pesan error" }
    { "type": "pong" }
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Pastikan root project ada di path saat dijalankan langsung
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import SERVER, DEBUG, PATHS
from core.memory import memory          # singleton MemoryManager
from core.brain import Brain
from core.transcriber import Transcriber
from core.speaker import Speaker
from core.executor import init_executor, Executor
from intelligence.activity_watcher import init_watcher
from intelligence.profiler import init_profiler
from intelligence.curiosity import init_curiosity, get_curiosity
from intelligence.scheduler import init_scheduler  

logger = logging.getLogger("otto.app")

# ─────────────────────────── Startup / Shutdown ──────────────────────────────

brain:       Brain      | None = None
transcriber: Transcriber| None = None
speaker:     Speaker    | None = None
executor:    Executor   | None = None
watcher   = None
profiler  = None
curiosity = None
scheduler = None
active_ws: WebSocket | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Init semua komponen saat server start, cleanup saat stop."""
    global brain, transcriber, speaker, executor

    logging.basicConfig(
        level   = logging.DEBUG if DEBUG else logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt = "%H:%M:%S",
    )

    logger.info("── Otto starting up ──")

    # 1. Brain (Groq LLM)
    brain = Brain(memory)
    logger.info("✓ Brain siap")

    # 2. Transcriber (Whisper)
    transcriber = Transcriber()
    logger.info("✓ Transcriber siap")

    # 3. Speaker (Piper TTS)
    speaker = Speaker()
    logger.info("✓ Speaker siap")

    # 4. Executor (dispatcher + skills)
    executor = init_executor(brain)
    _load_skills(executor)
    logger.info("✓ Executor siap — %d skill aktif", len(executor.list_skills()))

    # Intelligence layer
    watcher   = init_watcher()
    profiler  = init_profiler(watcher)
    curiosity = init_curiosity(profiler)

    scheduler = init_scheduler(watcher, profiler, curiosity)
    scheduler.set_question_callback(_broadcast_curiosity_with_pending)
    await scheduler.start()
    logger.info("✓ Scheduler + Curiosity siap")

    logger.info("── Otto siap. WebSocket: wss://%s:%d/ws ──", SERVER["host"], SERVER["port"])

    yield  # server jalan di sini

    # Cleanup
    logger.info("Otto shutting down…")
    if scheduler:
        await scheduler.stop()    # ← tambah ini
    if brain:
        await brain.close()


def _load_skills(ex: Executor) -> None:
    """
    Import semua skill dari skills/*.py.
    Setiap modul skill mendaftarkan dirinya ke executor saat di-import.
    Jika modul belum ada, skip dengan peringatan — tidak crash.
    """
    skill_modules = [
        "skills.system",
        "skills.media",
        "skills.reminder",
    ]
    for mod_name in skill_modules:
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            # Skill yang punya fungsi register() → panggil dengan executor
            if hasattr(mod, "register"):
                mod.register(ex)
                logger.info("  ✓ Skill '%s' dimuat", mod_name)
            else:
                logger.debug("  ○ '%s' tidak punya register(), skip", mod_name)
        except ModuleNotFoundError:
            logger.warning("  ✗ Skill '%s' belum ada, skip", mod_name)
        except Exception as e:
            logger.error("  ✗ Skill '%s' error saat load: %s", mod_name, e)


# ─────────────────────────── App ─────────────────────────────────────────────

app = FastAPI(title="Otto AI", lifespan=lifespan)

# Serve static files (index.html untuk iPhone)
static_dir = ROOT / "server" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """Redirect ke UI jika ada, atau info JSON."""
    index = static_dir / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return {"status": "Otto aktif", "ws": "/ws"}


@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "memory":  {
            "short": memory.short_term_count(),
            "long":  memory.long_term_count(),
        },
        "skills": len(executor.list_skills()) if executor else 0,
    }


# ─────────────────────────── WebSocket ───────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global active_ws
    await ws.accept()
    active_ws = ws                                      # ← simpan referensi
    client = ws.client.host if ws.client else "unknown"
    logger.info("[ws] Client terhubung: %s", client)

    # Kirim sapaan
    await _send_json(ws, "response", "Otto aktif. Hei, ada yang bisa aku bantu?")

    try:
        while True:
            raw = await ws.receive_json()
            msg_type = raw.get("type", "")

            if msg_type == "ping":
                await ws.send_json({"type": "pong"})

            elif msg_type == "audio":
                await _handle_audio(ws, raw)

            elif msg_type == "text":
                await _handle_text(ws, raw.get("data", "").strip())

            else:
                logger.warning("[ws] Tipe pesan tidak dikenal: %s", msg_type)
                await _send_error(ws, f"Tipe pesan '{msg_type}' tidak dikenal.")

    except WebSocketDisconnect:
        active_ws = None
        logger.info("[ws] Client disconnect: %s", client)
    except Exception as e:
        logger.error("[ws] Error tak terduga: %s", e, exc_info=True)
        try:
            await _send_error(ws, "Terjadi error internal.")
        except Exception:
            pass


# ─────────────────────────── Handler ─────────────────────────────────────────

async def _handle_audio(ws: WebSocket, msg: dict) -> None:
    """
    Terima audio base64 → STT → dispatch → TTS → kirim balik.
    """
    b64 = msg.get("data", "")
    if not b64:
        await _send_error(ws, "Data audio kosong.")
        return

    try:
        audio_bytes = base64.b64decode(b64)
        logger.info("[debug] Audio diterima: %d bytes", len(audio_bytes))
    except Exception:
        await _send_error(ws, "Format base64 audio tidak valid.")
        return

    # 1. STT — pilih model berdasarkan panjang audio
    # Heuristik: < 2 detik PCM 16kHz 16bit = 64000 bytes → tiny, else medium
    model_hint = "command" if len(audio_bytes) < 64_000 else "chat"
    try:
        transcript = await asyncio.to_thread(
            transcriber.transcribe, audio_bytes, mode="command"
        )
    except Exception as e:
        logger.error("[ws] STT error: %s", e)
        await _send_error(ws, "Gagal transkripsi audio.")
        return

    if not transcript.strip():
        await _send_error(ws, "Tidak ada suara yang terdeteksi.")
        return

    # Kirim transcript dulu agar UI bisa tampilkan teks sebelum jawaban
    await _send_json(ws, "transcript", transcript)

    # 2. Proses teks
    await _handle_text(ws, transcript)


async def _handle_text(ws: WebSocket, text: str) -> None:
    if not text:
        return

    # Beritahu scheduler bahwa Rofi sedang aktif (tunda curiosity)
    if scheduler:
        scheduler.notify_conversation_active()   # ← tambah ini

    # Cek dulu: apakah ini jawaban untuk pertanyaan curiosity?
    pending_id = memory.get_temp("curiosity_pending")
    if pending_id:
        verdict = await curiosity.handle_response(pending_id, text)
        memory.delete_temp("curiosity_pending")
        if verdict != "unclear":
            ack = "Oke, aku catat." if verdict == "confirmed" else "Oke, aku koreksi."
            await _send_json(ws, "response", ack)
            return   # tidak diteruskan ke executor

    # Dispatch ke executor
    try:
        result = await executor.dispatch(text)
    except Exception as e:
        logger.error("[ws] Executor error: %s", e)
        await _send_error(ws, "Otto sedang ada masalah.")
        return

    reply = result.text

    # TTS
    audio_b64 = ""
    try:
        audio_bytes = await asyncio.to_thread(speaker.synthesize, reply)
        if audio_bytes:
            audio_b64 = base64.b64encode(audio_bytes).decode()
    except Exception as e:
        logger.warning("[ws] TTS gagal, kirim teks saja: %s", e)

    # Kirim respons + audio
    await ws.send_json({
        "type":  "response",
        "data":  reply,
        "audio": audio_b64,
        "meta": {
            "intent": result.intent.value,
            "skill":  result.skill,
        },
    })


# ─────────────────────────── Helper ──────────────────────────────────────────

async def _send_json(ws: WebSocket, msg_type: str, data: str) -> None:
    await ws.send_json({"type": msg_type, "data": data})

async def _send_error(ws: WebSocket, msg: str) -> None:
    await ws.send_json({"type": "error", "data": msg})




async def _broadcast_curiosity(question: str) -> None:
    """Kirim pertanyaan Otto ke iPhone lewat TTS."""
    if active_ws is None:
        logger.info("[curiosity] Tidak ada client aktif, pertanyaan ditunda.")
        return

    audio_b64 = ""
    try:
        audio_bytes = await asyncio.to_thread(speaker.synthesize, question)
        if audio_bytes:
            audio_b64 = base64.b64encode(audio_bytes).decode()
    except Exception as e:
        logger.warning("[curiosity] TTS gagal: %s", e)

    try:
        await active_ws.send_json({
            "type":  "response",
            "data":  question,
            "audio": audio_b64,
            "meta":  {"intent": "curiosity", "skill": "curiosity"},
        })
        logger.info("[curiosity] Pertanyaan terkirim ke iPhone: %s", question)
    except Exception as e:
        logger.warning("[curiosity] Gagal kirim ke ws: %s", e)


async def _broadcast_curiosity_with_pending(question: str, hyp_id: str) -> None:
    """
    Callback untuk Scheduler — simpan pending lalu broadcast.
    Scheduler sudah handle timing & cooldown, jadi langsung kirim.
    """
    memory.set_temp("curiosity_pending", hyp_id)
    await _broadcast_curiosity(question)




# ─────────────────────────── Main ────────────────────────────────────────────

if __name__ == "__main__":
    ssl_keyfile  = str(PATHS.get("ssl_key",  ROOT / "ssl" / "key.pem"))
    ssl_certfile = str(PATHS.get("ssl_cert", ROOT / "ssl" / "cert.pem"))

    use_ssl = Path(ssl_keyfile).exists() and Path(ssl_certfile).exists()

    uvicorn.run(
        "server.app:app",
        host      = SERVER["host"],
        port      = SERVER["port"],
        reload    = SERVER.get("reload", False),
        log_level = "debug" if DEBUG else "info",
        ssl_keyfile  = ssl_keyfile  if use_ssl else None,
        ssl_certfile = ssl_certfile if use_ssl else None,
    )
