"""
server/app.py — Entry Point Otto
==================================
Alur lengkap:
  iPhone (WebSocket) → app.py
      → transcriber.transcribe(audio_bytes)   [Whisper medium]
      → brain.think(teks)                     [Groq 70b]
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
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import SERVER, DEBUG, PATHS
from core.memory import memory
from core.brain import Brain, BrainResponse
from core.speaker import Speaker
from intelligence.activity_watcher import init_watcher
from intelligence.profiler import init_profiler
from intelligence.curiosity import init_curiosity
from intelligence.scheduler import init_scheduler
from intelligence.growth_tracker import init_tracker, get_tracker
from core.transcriber import Transcriber, get_transcriber

logger = logging.getLogger("otto.app")

# ─────────────────────────── State Global ────────────────────────────────────

brain:       Brain       | None = None
speaker:     Speaker     | None = None
watcher   = None
profiler  = None
curiosity = None
scheduler = None
tracker   = None
transcriber: Transcriber | None = None
active_ws: WebSocket | None = None


# ─────────────────────────── Startup / Shutdown ──────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global brain, transcriber, speaker
    global watcher, profiler, curiosity, scheduler, tracker

    logging.basicConfig(
        level   = logging.DEBUG if DEBUG else logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt = "%H:%M:%S",
    )

    logger.info("── Otto starting up ──")

    # Core
    brain       = Brain(memory)
    logger.info("✓ Brain siap")

    transcriber = get_transcriber()
    logger.info("✓ Transcriber siap")

    speaker     = Speaker()
    logger.info("✓ Speaker siap")

    # Intelligence layer
    watcher   = init_watcher()
    profiler  = init_profiler(watcher)
    curiosity = init_curiosity(profiler)
    scheduler = init_scheduler(watcher, profiler, curiosity)
    tracker   = init_tracker()

    scheduler.set_question_callback(_broadcast_curiosity_with_pending)
    await scheduler.start()
    logger.info("✓ Intelligence layer siap")

    logger.info("── Otto siap. WebSocket: wss://%s:%d/ws ──", SERVER["host"], SERVER["port"])

    yield

    logger.info("Otto shutting down…")
    if scheduler:
        await scheduler.stop()
    if brain:
        await brain.close()


# ─────────────────────────── App ─────────────────────────────────────────────

app = FastAPI(title="Otto AI", lifespan=lifespan)

static_dir = ROOT / "server" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    index = static_dir / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return {"status": "Otto aktif", "ws": "/ws"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "memory": {
            "short": memory.short_term_count(),
            "long":  memory.long_term_count(),
        },
    }


@app.get("/growth/data")
async def growth_data():
    try:
        t = get_tracker()
        return t.full_report()
    except Exception as e:
        return {"error": str(e), "cumulative_total": 0, "weekly_history": [], "current_week": None}


@app.get("/growth")
async def growth_page():
    page = ROOT / "server" / "static" / "growth_history.html"
    if page.exists():
        return HTMLResponse(page.read_text())
    return {"error": "growth_history.html belum ada di server/static/"}


# ─────────────────────────── WebSocket ───────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global active_ws
    await ws.accept()
    active_ws = ws
    client = ws.client.host if ws.client else "unknown"
    logger.info("[ws] Client terhubung: %s", client)

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
    b64 = msg.get("data", "")
    if not b64:
        await _send_error(ws, "Data audio kosong.")
        return

    try:
        audio_bytes = base64.b64decode(b64)
        logger.info("[ws] Audio diterima: %d bytes", len(audio_bytes))
    except Exception:
        await _send_error(ws, "Format base64 audio tidak valid.")
        return

    # ── STT dengan timeout global ──────────────────────────────────────
    try:
        transcript = await asyncio.wait_for(
            asyncio.to_thread(transcriber.transcribe, audio_bytes),
            timeout=40.0  # 40 detik: ffmpeg(15) + Whisper(25) max
        )
    except asyncio.TimeoutError:
        logger.warning("[ws] STT timeout — paksa berhenti")
        await _send_json(ws, "response",
            "Maaf Rofi, tadi aku tidak berhasil mendengar dengan baik. "
            "Bisa kamu ulangi?"
        )
        return
    except Exception as e:
        logger.error("[ws] STT error: %s", e)
        await _send_error(ws, "Gagal transkripsi audio.")
        return

    # Tangani sinyal TIMEOUT dari transcriber
    if not transcript or transcript.strip() == "" or transcript == "TIMEOUT":
        if transcript == "TIMEOUT":
            await _send_json(ws, "response",
                "Maaf Rofi, audio tadi agak susah aku proses. Bisa diulang?"
            )
        else:
            await _send_error(ws, "Tidak ada suara yang terdeteksi.")
        return

    await _send_json(ws, "transcript", transcript)
    await _handle_text(ws, transcript)

async def _handle_text(ws: WebSocket, text: str) -> None:
    if not text:
        return

    if scheduler:
        scheduler.notify_conversation_active()

    # Cek apakah ini jawaban untuk pertanyaan curiosity
    pending_id = memory.get_temp("curiosity_pending")
    if pending_id:
        verdict = await curiosity.handle_response(pending_id, text)
        memory.delete_temp("curiosity_pending")
        if verdict != "unclear":
            ack = "Oke, aku catat." if verdict == "confirmed" else "Oke, aku koreksi."
            await _send_json(ws, "response", ack)
            return

    # Ambil history percakapan untuk konteks
    history = memory.get_recent_messages(limit=20)

    # Kirim ke brain
    # Kirim ke brain — dengan timeout
    try:
        resp: BrainResponse = await asyncio.wait_for(
            brain.think(text, history=history),
            timeout=30.0  # Groq biasanya < 5 detik, 30 detik sangat aman
        )
    except asyncio.TimeoutError:
        logger.warning("[ws] Brain timeout untuk input: %s", text[:50])
        await _send_json(ws, "response",
            "Maaf Rofi, aku butuh waktu lebih dari biasanya. Bisa kamu tanya lagi?"
        )
        return
    except Exception as e:
        logger.error("[ws] Brain error: %s", e)
        await _send_error(ws, "Otto sedang ada masalah.")
        return

    reply = resp.text

    # Catat ke growth tracker
    if tracker:
        tracker.record_interaction(text_length=len(text))

    # TTS
    audio_b64 = ""
    try:
        audio_bytes = await asyncio.to_thread(speaker.synthesize, reply)
        if audio_bytes:
            audio_b64 = base64.b64encode(audio_bytes).decode()
    except Exception as e:
        logger.warning("[ws] TTS gagal, kirim teks saja: %s", e)

    await ws.send_json({
        "type":  "response",
        "data":  reply,
        "audio": audio_b64,
        "meta": {"intent": "curiosity"}
    })


# ─────────────────────────── Helper ──────────────────────────────────────────

async def _send_json(ws: WebSocket, msg_type: str, data: str) -> None:
    await ws.send_json({"type": msg_type, "data": data})


async def _send_error(ws: WebSocket, msg: str) -> None:
    await ws.send_json({"type": "error", "data": msg})


async def _broadcast_curiosity(question: str) -> None:
    try:
        await asyncio.to_thread(speaker.speak_local, question)
        logger.info("[curiosity] Diputar di laptop: %s", question)
    except Exception as e:
        logger.warning("[curiosity] TTS lokal gagal: %s", e)

    if active_ws:
        try:
            await active_ws.send_json({
                "type":  "response",
                "data":  question,
                "audio": "",
                "meta":  {"intent": "curiosity", "skill": "curiosity"},
            })
        except Exception as e:
            logger.warning("[curiosity] Gagal kirim teks ke ws: %s", e)


async def _broadcast_curiosity_with_pending(question: str, hyp_id: str) -> None:
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
