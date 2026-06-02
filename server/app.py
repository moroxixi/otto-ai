"""
server/app.py — Entry Point Otto
==================================
"""

from __future__ import annotations

import asyncio
import base64
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
import signal

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import SERVER, DEBUG, PATHS, WHISPER
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

brain:       Brain       | None = None
speaker:     Speaker     | None = None
watcher   = None
profiler  = None
curiosity = None
scheduler = None
tracker   = None
transcriber: Transcriber | None = None
active_ws: WebSocket | None = None

STT_TIMEOUT = WHISPER.get("stt_timeout", 300)

CRASH_LOG = PATHS["base"] / "data" / "crash.log"


def _write_crash_log(reason: str) -> None:
    try:
        CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
        timestamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(CRASH_LOG, "a") as f:
            f.write(f"[{timestamp}] {reason}\n")
    except Exception:
        pass


def _setup_signal_handlers() -> None:
    def _handle_sigterm(signum, frame):
        logger.info("SIGTERM diterima — Otto shutdown bersih.")
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _handle_sigterm)


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

    loop = asyncio.get_event_loop()
    loop.set_exception_handler(_handle_exception)
    _setup_signal_handlers()

    logger.info("── Otto starting up ──")

    transcriber = get_transcriber()
    logger.info("✓ Transcriber siap (dual-mode: tiny/medium)")

    speaker = Speaker()
    logger.info("✓ Speaker siap (Kokoro + Piper streaming)")

    watcher   = init_watcher()
    profiler  = init_profiler(watcher)
    curiosity = init_curiosity(profiler, memory)
    scheduler = init_scheduler(watcher, profiler, curiosity, memory=memory)
    tracker   = init_tracker()

    brain = Brain(memory, profiler=profiler)
    logger.info("✓ Brain siap")

    # FIX BUG 5: sync curiosity._pending_hypothesis_id dengan pending_state saat startup
    # Jika server restart saat ada pending, curiosity harus tau
    _recovered = pending_state.get()
    if _recovered and curiosity:
        curiosity._pending_hypothesis_id = _recovered
        logger.info("[startup] Recovered pending hypothesis: %s", _recovered)

    scheduler.set_question_callback(_broadcast_curiosity_with_pending)
    await scheduler.start()
    logger.info("✓ Intelligence layer siap")
    logger.info("── Otto siap. WebSocket: wss://%s:%d/ws ──", SERVER["host"], SERVER["port"])

    yield

    logger.info("Otto shutting down…")
    if scheduler: await scheduler.stop()
    if watcher:   await watcher.flush()
    if speaker:   speaker.shutdown()
    if brain:     await brain.close()

    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        logger.info("Membersihkan %d task yang masih jalan...", len(tasks))
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Semua task selesai dibersihkan.")


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


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global active_ws
    await ws.accept()

    if active_ws is not None and active_ws is not ws:
        try:
            await active_ws.close(code=1001, reason="Replaced by new connection")
            logger.info("[ws] Koneksi lama ditutup — diganti koneksi baru.")
        except Exception:
            pass

    active_ws = ws
    client = ws.client.host if ws.client else "unknown"
    logger.info("[ws] Client terhubung: %s", client)

    await _send_json(ws, "response", "Otto aktif. Hei, apa yang sedang kamu lakukan Rofi?")
    try:
        await speaker.stream_to_ws(ws, "Otto aktif. Hei, ada yang bisa aku bantu?")
    except Exception as e:
        logger.warning("[ws] Stream salam gagal: %s", e)

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
        if active_ws is ws:
            active_ws = None
        logger.info("[ws] Client disconnect: %s", client)
    except Exception as e:
        logger.error("[ws] Error tak terduga: %s", e, exc_info=True)
        try:
            await _send_error(ws, "Terjadi error internal.")
        except Exception:
            pass
        if active_ws is ws:
            active_ws = None


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

    await _send_json(ws, "status", "Sedang memproses suara...")

    try:
        transcript = await asyncio.wait_for(
            asyncio.to_thread(transcriber.transcribe, audio_bytes),
            timeout=STT_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("[ws] STT timeout setelah %ds", STT_TIMEOUT)
        await _send_json(ws, "response",
            "Maaf Rofi, tadi aku tidak berhasil mendengar dengan baik. Bisa kamu ulangi?"
        )
        await websocket.send_json({"type": "timeout", "data": "STT timeout"})
        return
    except asyncio.CancelledError:
        logger.info("[ws] STT dibatalkan (shutdown).")
        raise
    except Exception as e:
        logger.error("[ws] STT error: %s", e)
        await _send_error(ws, "Gagal transkripsi audio.")
        return

    if not transcript or transcript.strip() == "":
        await _send_error(ws, "Tidak ada suara yang terdeteksi.")
        return

    await _send_json(ws, "transcript", transcript)
    await _handle_text(ws, transcript)


async def _handle_text(ws: WebSocket, text: str) -> None:
    if not text:
        return

    if scheduler:
        scheduler.notify_conversation_active()

    # FIX BUG 3: pending_id diambil SEKALI di awal, dan clear() dijamin
    # dipanggil di finally block agar tidak bocor meski brain error
    pending_id = pending_state.get()
    if pending_id:
        try:
            verdict = await curiosity.handle_response(pending_id, text)
        except Exception as e:
            logger.error("[ws] handle_response error: %s", e)
            verdict = "unclear"
        finally:
            # Selalu clear — baik verdict jelas maupun error/unclear
            # "unclear" boleh tanya ulang nanti lewat scheduler
            pending_state.clear()

        if verdict != "unclear":
            if watcher:
                asyncio.create_task(
                    watcher.log(text, intent="curiosity_response", skill="curiosity")
                )
            ack = "Oke, aku catat." if verdict == "confirmed" else "Oke, aku koreksi."
            await _send_json(ws, "response", ack)
            try:
                await speaker.stream_to_ws(ws, ack)
            except Exception as e:
                # FIX BUG 2: clear active_ws jika stream gagal (client disconnect)
                logger.warning("[ws] Stream ack gagal: %s", e)
                if active_ws is ws:
                    active_ws = None
            return

    history = memory.get_recent_messages(limit=20)

    try:
        resp: BrainResponse = await asyncio.wait_for(
            brain.think(text, history=history),
            timeout=30.0
        )
    except asyncio.TimeoutError:
        logger.warning("[ws] Brain timeout: %s", text[:50])
        reply_timeout = "Maaf Rofi, aku butuh waktu lebih dari biasanya. Bisa kamu tanya lagi?"
        await _send_json(ws, "response", reply_timeout)
        try:
            await speaker.stream_to_ws(ws, reply_timeout)
        except Exception:
            if active_ws is ws:
                active_ws = None
        return
    except asyncio.CancelledError:
        logger.info("[ws] Brain dibatalkan (shutdown).")
        raise
    except Exception as e:
        logger.error("[ws] Brain error: %s", e)
        await _send_error(ws, "Otto sedang ada masalah.")
        return

    reply = resp.text

    if tracker:
        tracker.record_interaction(text_length=len(text))

    injected_hyp_id = None
    if curiosity and not pending_state.get():
        try:
            question, hyp_id = await curiosity.try_ask()
            if question and hyp_id:
                reply = reply + f"\n\nOmong-omong — {question}"
                injected_hyp_id = hyp_id
                logger.info("[ws] Curiosity inject: %s", hyp_id)
        except Exception as e:
            logger.warning("[ws] Curiosity inject gagal: %s", e)

    if injected_hyp_id:
        pending_state.set(injected_hyp_id)

    if watcher:
        skill_tag = "curiosity" if injected_hyp_id else ""
        asyncio.create_task(
            watcher.log(text, intent="chat", skill=skill_tag)
        )

    await ws.send_json({
        "type": "response",
        "data": reply,
        "meta": {
            "intent": "chat",
            "skill":  "curiosity" if injected_hyp_id else "",
        },
    })

    try:
        await speaker.stream_to_ws(ws, reply)
    except Exception as e:
        # FIX BUG 2: clear active_ws jika stream gagal (client disconnect di tengah TTS)
        logger.warning("[ws] Stream TTS gagal: %s", e)
        if active_ws is ws:
            active_ws = None


async def _send_json(ws: WebSocket, msg_type: str, data: str) -> None:
    try:
        await ws.send_json({"type": msg_type, "data": data})
    except Exception:
        logger.warning("[ws] Gagal kirim '%s' — client sudah disconnect", msg_type)


async def _send_error(ws: WebSocket, msg: str) -> None:
    try:
        await ws.send_json({"type": "error", "data": msg})
    except Exception:
        pass


async def _broadcast_curiosity(question: str) -> None:
    try:
        await speaker.ucapkan_laptop_async(question)
        logger.info("[curiosity] Diputar di laptop.")
    except Exception as e:
        logger.warning("[curiosity] TTS lokal gagal: %s", e)

    if active_ws:
        try:
            await active_ws.send_json({
                "type": "response",
                "data": question,
                "meta": {"intent": "curiosity", "skill": "curiosity"},
            })
            await speaker.stream_to_ws(active_ws, question)
        except Exception as e:
            logger.warning("[curiosity] Gagal stream ke ws: %s", e)


async def _broadcast_curiosity_with_pending(question: str, hyp_id: str) -> None:
    pending_state.set(hyp_id)
    await _broadcast_curiosity(question)


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
