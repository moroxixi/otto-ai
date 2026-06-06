/* ══════════════════════════════════════════════════════
   write.js — Otto Write Mode
   WebSocket + OttoOrb + Chat + History
   ══════════════════════════════════════════════════════ */

'use strict';

// ── Config ────────────────────────────────────────────
const WS_RETRY_DELAY = 5000;
const WS_MAX_RETRY   = 8;
const CHAT_KEY       = 'otto_write_history';
const MAX_AGE_MS     = 7 * 24 * 60 * 60 * 1000; // 7 hari

// ── DOM refs ──────────────────────────────────────────
const wsStatusEl   = document.getElementById('ws-status');
const wsStatusTxt  = wsStatusEl.querySelector('span');
const orbCanvas    = document.getElementById('orb-canvas');
const orbLabel     = document.getElementById('orb-label');
const chatArea     = document.getElementById('chat-area');
const chatEmpty    = document.getElementById('chat-empty');
const msgInput     = document.getElementById('msg-input');
const btnSend      = document.getElementById('btn-send');
const inputStatus  = document.getElementById('input-status');

// ── WebSocket state ───────────────────────────────────
let ws           = null;
let wsRetry      = 0;
let wsIntentClose= false;

// ── Orb instance ──────────────────────────────────────
let orb = null;

// ══════════════════════════════════════════════════════
//  OTTO ORB — Canvas 2D Animation
// ══════════════════════════════════════════════════════
class OttoOrb {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx    = canvas.getContext('2d');
    this.state  = 'idle'; // idle | thinking | responding
    this.t      = 0;
    this.burstProgress = 0;
    this.isBursting    = false;

    this.particles = Array.from({ length: 8 }, (_, i) => ({
      angle : (i / 8) * Math.PI * 2,
      radius: 38 + Math.random() * 8,
      speed : 0.007 + Math.random() * 0.004,
      size  : 1.5 + Math.random() * 1.8,
      phase : Math.random() * Math.PI * 2,
    }));

    // Resolusi HiDPI
    const dpr = window.devicePixelRatio || 1;
    const size = 100;
    canvas.width  = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width  = size + 'px';
    canvas.style.height = size + 'px';
    this.ctx.scale(dpr, dpr);
    this.SIZE = size;
  }

  setState(state) {
    if (this.state === state) return;
    const prev = this.state;
    this.state = state;

    if (state === 'responding' && prev === 'thinking') {
      this.isBursting    = true;
      this.burstProgress = 0;
    }

    // Update label
    if (state === 'idle') {
      orbLabel.textContent = '';
      orbLabel.className   = '';
    } else if (state === 'thinking') {
      orbLabel.textContent = 'MEMPROSES';
      orbLabel.className   = 'thinking';
    } else if (state === 'responding') {
      orbLabel.textContent = 'OTTO';
      orbLabel.className   = 'responding';
    }
  }

  // Identical drawBlob dengan index.html (same wobble formula)
  drawBlob(cx, cy, r, points, color) {
    const ctx = this.ctx;
    ctx.beginPath();
    const step = (Math.PI * 2) / points;
    for (let i = 0; i <= points; i++) {
      const angle  = i * step;
      const wobble = 1
        + 0.22 * Math.sin(i * 2.3 + this.t * 1.7)
        + 0.1  * Math.cos(i * 3.7 + this.t * 2.1);
      const x = cx + Math.cos(angle) * r * wobble;
      const y = cy + Math.sin(angle) * r * wobble;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  }

  draw() {
    const ctx = this.ctx;
    const S   = this.SIZE;
    const CX  = S / 2, CY = S / 2;
    ctx.clearRect(0, 0, S, S);
    this.t += 0.016;

    // Burst animation
    if (this.isBursting) {
      this.burstProgress += 0.06;
      if (this.burstProgress >= 1) this.isBursting = false;
    }

    // Warna per state
    const colors = {
      idle      : { r: 192, g: 132, b: 252 },
      thinking  : { r: 251, g: 146, b: 60  },
      responding: { r: 192, g: 132, b: 252 },
    };
    const c = colors[this.state];
    const rgb = `${c.r},${c.g},${c.b}`;

    // Radius orb utama
    let orbR;
    if (this.state === 'thinking') {
      orbR = 22 + Math.sin(this.t * 8) * 4;
    } else if (this.isBursting) {
      const burst = Math.sin(this.burstProgress * Math.PI);
      orbR = 22 + burst * 8;
    } else {
      orbR = 22 + Math.sin(this.t * 1.5) * 3.5;
    }

    // Glow halo
    const glowSize   = orbR * 2.4;
    const glowAlpha  = this.state === 'thinking' ? 0.12 : 0.09;
    const grad = ctx.createRadialGradient(CX, CY, 0, CX, CY, glowSize);
    grad.addColorStop(0,   `rgba(${rgb},${glowAlpha})`);
    grad.addColorStop(0.5, `rgba(${rgb},${glowAlpha * 0.4})`);
    grad.addColorStop(1,   'transparent');
    ctx.beginPath();
    ctx.arc(CX, CY, glowSize, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();

    // Orb utama
    this.drawBlob(CX, CY, orbR,      10, `rgba(${rgb},0.55)`);
    this.drawBlob(CX, CY, orbR * 0.65, 8, `rgba(${rgb},0.28)`);

    // Partikel orbit
    this.particles.forEach((p, i) => {
      let speed;
      if (this.state === 'thinking')   speed = p.speed * 4.5;
      else if (this.state === 'responding') speed = p.speed * 2;
      else speed = p.speed;

      p.angle += speed;

      // Burst: partikel melompat keluar lalu kembali
      let r = p.radius;
      if (this.isBursting) {
        const burst = Math.sin(this.burstProgress * Math.PI);
        r = p.radius + burst * 18;
      }

      const x = CX + Math.cos(p.angle) * r;
      const y = CY + Math.sin(p.angle) * r * 0.6; // orbit ellips

      const alpha = this.state === 'idle' ? 0.45 : 0.7;
      const size  = this.isBursting
        ? p.size * (1 + Math.sin(this.burstProgress * Math.PI) * 0.6)
        : p.size;

      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${rgb},${alpha})`;
      ctx.fill();
    });
  }

  loop() {
    this.draw();
    requestAnimationFrame(() => this.loop());
  }
}

// ══════════════════════════════════════════════════════
//  NOTIF TONE — Web Audio API
// ══════════════════════════════════════════════════════
let _notifCtx = null;

function playNotifTone(type) {
  try {
    if (!_notifCtx) _notifCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (_notifCtx.state === 'suspended') _notifCtx.resume();

    const osc  = _notifCtx.createOscillator();
    const gain = _notifCtx.createGain();
    osc.connect(gain);
    gain.connect(_notifCtx.destination);

    if (type === 'otto') {
      // Do-Mi naik — Otto jawab
      osc.frequency.setValueAtTime(523, _notifCtx.currentTime);
      osc.frequency.setValueAtTime(659, _notifCtx.currentTime + 0.12);
      gain.gain.setValueAtTime(0.12, _notifCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, _notifCtx.currentTime + 0.45);
      osc.start(_notifCtx.currentTime);
      osc.stop(_notifCtx.currentTime + 0.45);
    } else {
      // La — Rofi kirim
      osc.frequency.setValueAtTime(440, _notifCtx.currentTime);
      gain.gain.setValueAtTime(0.08, _notifCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, _notifCtx.currentTime + 0.25);
      osc.start(_notifCtx.currentTime);
      osc.stop(_notifCtx.currentTime + 0.25);
    }
  } catch (e) {
    // AudioContext blocked — tidak fatal
  }
}

// ══════════════════════════════════════════════════════
//  CHAT HISTORY — localStorage 7 hari
// ══════════════════════════════════════════════════════
function saveMessage(role, text) {
  const history = loadHistory();
  history.push({ role, text, ts: Date.now() });
  try { localStorage.setItem(CHAT_KEY, JSON.stringify(history)); } catch {}
}

function loadHistory() {
  try {
    const raw = localStorage.getItem(CHAT_KEY);
    if (!raw) return [];
    const all    = JSON.parse(raw);
    const cutoff = Date.now() - MAX_AGE_MS;
    const filtered = all.filter(m => m.ts > cutoff);
    if (filtered.length !== all.length) {
      localStorage.setItem(CHAT_KEY, JSON.stringify(filtered));
    }
    return filtered;
  } catch { return []; }
}

function renderHistory(history) {
  history.forEach(m => {
    if (m.role === 'user') appendRofiMessage(m.text, false);
    else                   appendOttoMessage(m.text, false);
  });
}

// ══════════════════════════════════════════════════════
//  TYPEWRITER — kata per kata
// ══════════════════════════════════════════════════════
function typewriterWords(element, text, onDone) {
  const words = text.split(' ');
  let i = 0;
  element.textContent = '';
  element.classList.add('typing');

  const interval = setInterval(() => {
    if (i < words.length) {
      element.textContent += (i === 0 ? '' : ' ') + words[i];
      i++;
      chatArea.scrollTop = chatArea.scrollHeight;
    } else {
      clearInterval(interval);
      element.classList.remove('typing');
      if (onDone) onDone();
    }
  }, 60);

  return interval; // bisa di-clearInterval dari luar jika perlu
}

// ══════════════════════════════════════════════════════
//  CHAT RENDER
// ══════════════════════════════════════════════════════
function getTimeStr() {
  const now = new Date();
  return now.toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });
}

function hideChatEmpty() {
  if (chatEmpty && chatEmpty.parentNode === chatArea) {
    chatArea.removeChild(chatEmpty);
  }
}

function appendRofiMessage(text, animate = true) {
  hideChatEmpty();
  const row = document.createElement('div');
  row.className = 'msg-row rofi';

  const dot = document.createElement('div');
  dot.className = 'msg-dot';
  dot.textContent = 'R';

  const bubble = document.createElement('div');
  bubble.className = 'bubble rofi';
  bubble.textContent = text;

  const time = document.createElement('div');
  time.className = 'msg-time';
  time.textContent = getTimeStr();

  row.appendChild(dot);
  row.appendChild(bubble);
  row.appendChild(time);
  chatArea.appendChild(row);
  chatArea.scrollTop = chatArea.scrollHeight;

  if (animate) playNotifTone('rofi');
  return bubble;
}

function appendOttoMessage(text, animate = true) {
  hideChatEmpty();
  const row = document.createElement('div');
  row.className = 'msg-row otto';

  const dot = document.createElement('div');
  dot.className = 'msg-dot';
  dot.textContent = 'O';

  const bubble = document.createElement('div');
  bubble.className = 'bubble otto';

  const time = document.createElement('div');
  time.className = 'msg-time';
  time.textContent = getTimeStr();

  row.appendChild(dot);
  row.appendChild(bubble);
  row.appendChild(time);
  chatArea.appendChild(row);
  chatArea.scrollTop = chatArea.scrollHeight;

  if (animate) {
    playNotifTone('otto');
    typewriterWords(bubble, text, () => {
      orb.setState('idle');
    });
  } else {
    bubble.textContent = text;
  }

  return bubble;
}

// ══════════════════════════════════════════════════════
//  KIRIM PESAN
// ══════════════════════════════════════════════════════
function sendMessage() {
  const text = msgInput.value.trim();
  if (!text) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    setInputStatus('error', '⚠ BELUM TERHUBUNG');
    return;
  }

  appendRofiMessage(text);
  saveMessage('user', text);

  msgInput.value = '';
  msgInput.style.height = 'auto';
  btnSend.classList.remove('has-text');

  orb.setState('thinking');
  setInputStatus('thinking', 'MENUNGGU OTTO...');

  ws.send(JSON.stringify({
    type: 'text',
    data: text,
    mode: 'write',   // ← flag skip TTS di server
  }));
}

// ── Input helpers ─────────────────────────────────────
function setInputStatus(cls, text) {
  inputStatus.className = cls || '';
  inputStatus.textContent = text;
}

// ── Auto-resize textarea ──────────────────────────────
msgInput.addEventListener('input', () => {
  msgInput.style.height = 'auto';
  msgInput.style.height = Math.min(msgInput.scrollHeight, 120) + 'px';
  const hasText = msgInput.value.trim().length > 0;
  btnSend.classList.toggle('has-text', hasText);
});

msgInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

btnSend.addEventListener('click', sendMessage);

// ══════════════════════════════════════════════════════
//  WebSocket
// ══════════════════════════════════════════════════════
function setWsStatus(cls, text) {
  wsStatusEl.className = cls;
  wsStatusTxt.textContent = text;
}

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss://' : 'ws://';
  ws = new WebSocket(`${proto}${location.host}/ws`);

  let pingInterval = null;

  ws.onopen = () => {
    wsRetry = 0;
    setWsStatus('connected', 'online');
    setInputStatus('', '');
    if (pingInterval) clearInterval(pingInterval);
    pingInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'ping' }));
    }, 20000);
  };

  ws.onclose = () => {
    setWsStatus('disconnected', 'offline');
    if (pingInterval) { clearInterval(pingInterval); pingInterval = null; }
    if (wsIntentClose) return;
    if (wsRetry >= WS_MAX_RETRY) {
      setWsStatus('disconnected', 'gagal terhubung');
      setInputStatus('error', 'KONEKSI GAGAL');
      return;
    }
    wsRetry++;
    setWsStatus('reconnecting', `reconnect ${wsRetry}...`);
    setInputStatus('', 'MENGHUBUNGKAN ULANG...');
    setTimeout(connectWS, WS_RETRY_DELAY);
  };

  ws.onerror = () => {};

  ws.onmessage = (event) => {
    let msg;
    try { msg = JSON.parse(event.data); } catch { return; }

    switch (msg.type) {

      case 'pong':
        break;

      // ── Respons teks Otto dari write mode ─────────
      case 'text_response':
        orb.setState('responding');
        setInputStatus('', '');
        appendOttoMessage(msg.text || msg.data || '');
        saveMessage('otto', msg.text || msg.data || '');
        break;

      // ── Fallback: server kirim type "response" biasa ──
      case 'response': {
        const text = msg.data || '';
        if (!text) break;
        orb.setState('responding');
        setInputStatus('', '');
        appendOttoMessage(text);
        saveMessage('otto', text);
        // Abaikan audio jika ada — write mode tidak putar TTS
        break;
      }

      case 'status':
        orb.setState('thinking');
        setInputStatus('thinking', (msg.data || 'MEMPROSES').toUpperCase().slice(0, 30));
        break;

      case 'transcript':
        // Write mode tidak tampilkan transcript (tidak ada audio)
        break;

      case 'error':
        orb.setState('idle');
        setInputStatus('error', '⚠ ' + (msg.data || 'ERROR').toUpperCase().slice(0, 40));
        setTimeout(() => setInputStatus('', ''), 4000);
        break;

      case 'timeout':
        orb.setState('idle');
        setInputStatus('error', '⏱ TIMEOUT — COBA LAGI');
        setTimeout(() => setInputStatus('', ''), 4000);
        break;
    }
  };
}

// ══════════════════════════════════════════════════════
//  BOOT
// ══════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  // Init orb
  orb = new OttoOrb(orbCanvas);
  orb.loop();

  // Render history
  const history = loadHistory();
  if (history.length > 0) {
    renderHistory(history);
  }

  // Fokus input
  msgInput.focus();

  // Sambung WS
  connectWS();
});
