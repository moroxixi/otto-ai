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
import signal
import subprocess

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
from intelligence.pending_state import pending_state

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



CRASH_LOG = PATHS["base"] / "data" / "crash.log"
 
def _write_crash_log(reason: str) -> None:
    """Tulis detail crash ke file sebelum mati."""
    try:
        CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
        timestamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(CRASH_LOG, "a") as f:
            f.write(f"[{timestamp}] {reason}\n")
    except Exception:
        pass  # jangan raise lagi saat crash handler
 
 
def _setup_signal_handlers() -> None:
    """
    Tangkap SIGTERM (dari systemd saat restart) dan SIGINT (Ctrl+C).
    Pastikan shutdown berjalan bersih.
    """
    def _handle_sigterm(signum, frame):
        logger.info("SIGTERM diterima — Otto shutdown bersih.")
        # FastAPI/uvicorn akan handle sisanya via lifespan shutdown
        raise SystemExit(0)
 
    signal.signal(signal.SIGTERM, _handle_sigterm)
 


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
    def _handle_exception(loop, context):
        exc = context.get("exception")
        msg = context.get("message", "Unknown async error")
        logger.critical("[FATAL] Unhandled async exception: %s | %s", msg, exc)
        _write_crash_log(f"Unhandled async exception: {msg} — {exc}")
        # Biarkan Otto tetap jalan — hanya log, tidak kill process
        # Kalau ingin kill: loop.stop()

    loop = asyncio.get_event_loop()
    loop.set_exception_handler(_handle_exception)
    _setup_signal_handlers()
 

    logger.info("── Otto starting up ──")

    transcriber = get_transcriber()
    logger.info("✓ Transcriber siap")

    speaker = Speaker()
    logger.info("✓ Speaker siap")

    watcher   = init_watcher()
    profiler  = init_profiler(watcher)
    curiosity = init_curiosity(profiler, memory)
    scheduler = init_scheduler(watcher, profiler, curiosity, memory=memory)
    tracker   = init_tracker()

    brain = Brain(memory, profiler=profiler)
    logger.info("✓ Brain siap")

    scheduler.set_question_callback(_broadcast_curiosity_with_pending)
    await scheduler.start()
    logger.info("✓ Intelligence layer siap")

    logger.info("── Otto siap. WebSocket: wss://%s:%d/ws ──", SERVER["host"], SERVER["port"])

    yield

    logger.info("Otto shutting down…")

    # 1. Hentikan komponen Otto dulu
    if scheduler:
        await scheduler.stop()
    if watcher:
        await watcher.flush()
    if speaker:
        speaker.shutdown()
    if brain:
        await brain.close()

    # 2. Cancel semua asyncio task yang masih jalan
    #    Ini mencegah uvloop crash saat loop ditutup paksa
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        logger.info("Membersihkan %d task yang masih jalan...", len(tasks))
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Semua task selesai dibersihkan.")


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
        logger.info("[ws] Audio header bytes: %s", audio_bytes[:12].hex())
    except Exception:
        await _send_error(ws, "Format base64 audio tidak valid.")
        return

    await _send_json(ws, "status", "Sedang mendengarkan...")

    try:
        transcript = await asyncio.wait_for(
            asyncio.to_thread(transcriber.transcribe, audio_bytes),
            timeout=90.0  # naik dari 40 → 90 detik untuk Whisper medium
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
    # ── 1. Notify scheduler Rofi sedang aktif ────────────────────────────────
    if scheduler:
        scheduler.notify_conversation_active()

    # ── 2. Cek apakah ini jawaban untuk pertanyaan curiosity ─────────────────
    pending_id = pending_state.get()
    if pending_id:
        verdict = await curiosity.handle_response(pending_id, text)
        pending_state.clear()
        if verdict != "unclear":
            # Log dengan intent yang benar — ini bukan chat biasa
            if watcher:
                asyncio.create_task(
                    watcher.log(text, intent="curiosity_response", skill="curiosity")
                )
            ack = "Oke, aku catat." if verdict == "confirmed" else "Oke, aku koreksi."
            await _send_json(ws, "response", ack)
            return

    # ── 3. Ambil history & kirim ke brain ────────────────────────────────────
    history = memory.get_recent_messages(limit=20)

    try:
        resp: BrainResponse = await asyncio.wait_for(
            brain.think(text, history=history),
            timeout=30.0
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

    # ── 4. Catat interaksi ke growth tracker ─────────────────────────────────
    if tracker:
        tracker.record_interaction(text_length=len(text))

    # ── 5. Inject pertanyaan curiosity di akhir reply (jika waktunya tepat) ──
    #
    # Filosofi: Otto tidak interupsi. Dia jawab dulu, baru selip satu pertanyaan
    # di akhir — terasa natural, seperti teman yang ngobrol sambil penasaran.
    # Pertanyaan hanya muncul jika:
    #   - profiler sudah analyze malam ini (ada hipotesis segar)
    #   - curiosity.try_ask() memutuskan waktu aman
    #   - tidak ada pertanyaan yang sedang menggantung
    #
    injected_hyp_id = None
    if curiosity and not pending_state.get():
        try:
            question, hyp_id = await curiosity.try_ask()
            if question and hyp_id:
                # Selipkan dengan pemisah natural — bukan kalimat baru yang kaku
                reply = reply + f"\n\nOmong-omong — {question}"
                injected_hyp_id = hyp_id
                logger.info("[ws] Curiosity inject pertanyaan: %s", hyp_id)
        except Exception as e:
            logger.warning("[ws] Curiosity inject gagal (tidak kritis): %s", e)

    # Simpan hyp_id ke disk supaya survive restart
    if injected_hyp_id:
        pending_state.set(injected_hyp_id)
    # ── 6.
    if watcher:
        skill_tag = "curiosity" if injected_hyp_id else ""
        asyncio.create_task(
            watcher.log(text, intent="chat", skill=skill_tag)
        )

    # ── 7. Synthesize TTS & kirim balik ──────────────────────────────────────
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
        "meta":  {
            "intent": "chat",
            "skill":  "curiosity" if injected_hyp_id else "",
        },
    })


# ─────────────────────────── Helper ──────────────────────────────────────────

async def _send_json(ws: WebSocket, msg_type: str, data: str) -> None:
    try:
        await ws.send_json({"type": msg_type, "data": data})
    except (WebSocketDisconnect, Exception):
        logger.warning("[ws] Gagal kirim '%s' — client sudah disconnect", msg_type)


async def _send_error(ws: WebSocket, msg: str) -> None:
    await ws.send_json({"type": "error", "data": msg})


async def _broadcast_curiosity(question: str) -> None:
    """Kirim pertanyaan curiosity dari scheduler (bukan dari alur normal)."""
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
    pending_state.set(hyp_id)
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
        loop        = "uvloop",
    )
