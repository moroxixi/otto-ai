"""
self/github_checker.py — Otto Memahami Pertumbuhannya Sendiri

Bukan sekadar "ada update" — Otto membaca apa yang berubah,
lalu menyimpulkan narasi: "Hari ini aku jadi lebih baik di X."

Analogi: Seperti seseorang yang pulang kursus dan sadar
bahwa sekarang dia bisa sesuatu yang kemarin tidak bisa.

Flow:
  1. Cek GitHub → ada commit baru?
  2. Fetch diff → file apa yang berubah?
  3. Peta file ke kemampuan Otto
  4. Generate narasi (via LLM atau rule-based)
  5. Simpan ke self/last_changes.json
  6. model.py akan membaca file ini → Otto "sadar" dirinya berkembang
"""

import json
import os
import subprocess
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Path ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path("/data/asd/otto-ai")
SELF_DIR      = BASE_DIR / "self"
CHANGES_FILE  = SELF_DIR / "last_changes.json"
PREV_HASH_FILE = SELF_DIR / ".last_commit_hash"

# ── GitHub config (baca dari env / .env) ─────────────────────────────────────
GITHUB_REPO   = os.getenv("OTTO_GITHUB_REPO", "")   # e.g. "username/otto-ai"
GITHUB_TOKEN  = os.getenv("GITHUB_TOKEN", "")
GITHUB_BRANCH = os.getenv("OTTO_GITHUB_BRANCH", "main")

# ── Groq config (untuk narasi LLM) ───────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY_1", "")


# ══════════════════════════════════════════════════════════════════════════════
# PETA: file → kemampuan Otto (dalam bahasa Otto)
# ══════════════════════════════════════════════════════════════════════════════

FILE_CAPABILITY_MAP = {
    # Core
    "core/brain.py":        "cara berpikir dan memilih respons",
    "core/memory.py":       "kemampuan mengingat percakapan",
    "core/executor.py":     "kemampuan menjalankan perintah",
    "core/transcriber.py":  "kemampuan mendengar dan mengerti suara",
    "core/speaker.py":      "kemampuan berbicara",
    "core/config.py":       "konfigurasi dasar",

    # Intelligence
    "intelligence/activity_watcher.py": "kemampuan mengamati aktivitas Rofi",
    "intelligence/profiler.py":         "kemampuan membangun profil Rofi",
    "intelligence/curiosity.py":        "kemampuan mengajukan pertanyaan yang tepat",
    "intelligence/scheduler.py":        "kemampuan mengatur jadwal dan pengingat",

    # Skills
    "skills/system.py":   "skill kontrol sistem",
    "skills/media.py":    "skill putar media",
    "skills/reminder.py": "skill pengingat",

    # Self
    "self/model.py":          "kesadaran diri",
    "self/github_checker.py": "kesadaran akan pertumbuhan diri",

    # Server
    "server/app.py":      "server komunikasi dengan Rofi",
    "server/websocket.py": "koneksi WebSocket ke iPhone",
}


# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 1 — CEK PERUBAHAN (via git lokal atau GitHub API)
# ══════════════════════════════════════════════════════════════════════════════

def get_local_git_changes() -> Optional[dict]:
    """
    Cek perubahan lewat git lokal.
    Lebih cepat dan tidak butuh API key.
    Dipakai jika otto-ai di-develop di mesin yang sama.
    """
    try:
        # Commit hash terbaru
        result = subprocess.run(
            ["git", "-C", str(BASE_DIR), "log", "-1", "--format=%H|%s|%ai"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None

        parts = result.stdout.strip().split("|")
        if len(parts) < 3:
            return None

        latest_hash, commit_msg, commit_date = parts[0], parts[1], parts[2]

        # Bandingkan dengan hash sebelumnya
        prev_hash = ""
        if PREV_HASH_FILE.exists():
            prev_hash = PREV_HASH_FILE.read_text().strip()

        if latest_hash == prev_hash:
            return None   # Tidak ada perubahan

        # Ada perubahan — fetch file yang berubah
        if prev_hash:
            diff_result = subprocess.run(
                ["git", "-C", str(BASE_DIR), "diff", "--name-only", prev_hash, latest_hash],
                capture_output=True, text=True, timeout=5
            )
            changed_files = [f.strip() for f in diff_result.stdout.splitlines() if f.strip()]
        else:
            # Pertama kali — ambil semua file tracked
            diff_result = subprocess.run(
                ["git", "-C", str(BASE_DIR), "ls-files"],
                capture_output=True, text=True, timeout=5
            )
            changed_files = [f.strip() for f in diff_result.stdout.splitlines() if f.strip()]

        return {
            "latest_hash":  latest_hash,
            "prev_hash":    prev_hash,
            "commit_msg":   commit_msg,
            "commit_date":  commit_date,
            "changed_files": changed_files,
            "source": "local_git",
        }

    except Exception as e:
        return {"error": str(e), "source": "local_git"}


def get_github_api_changes() -> Optional[dict]:
    """
    Cek perubahan lewat GitHub API.
    Dipakai jika mesin Otto tidak sama dengan mesin dev,
    atau tidak ada git lokal.
    """
    if not GITHUB_REPO or not GITHUB_TOKEN:
        return None

    try:
        import urllib.request
        import urllib.error

        # Fetch commit terbaru
        url = f"https://api.github.com/repos/{GITHUB_REPO}/commits/{GITHUB_BRANCH}"
        req = urllib.request.Request(url, headers={
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Otto-AI",
        })

        with urllib.request.urlopen(req, timeout=10) as resp:
            commit_data = json.loads(resp.read())

        latest_hash = commit_data["sha"]
        commit_msg  = commit_data["commit"]["message"].split("\n")[0]
        commit_date = commit_data["commit"]["author"]["date"]

        # Bandingkan
        prev_hash = ""
        if PREV_HASH_FILE.exists():
            prev_hash = PREV_HASH_FILE.read_text().strip()

        if latest_hash == prev_hash:
            return None

        # Fetch file yang berubah
        changed_files = []
        if prev_hash:
            compare_url = f"https://api.github.com/repos/{GITHUB_REPO}/compare/{prev_hash}...{latest_hash}"
            req2 = urllib.request.Request(compare_url, headers={
                "Authorization": f"token {GITHUB_TOKEN}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Otto-AI",
            })
            with urllib.request.urlopen(req2, timeout=10) as resp2:
                compare_data = json.loads(resp2.read())
            changed_files = [f["filename"] for f in compare_data.get("files", [])]

        return {
            "latest_hash":   latest_hash,
            "prev_hash":     prev_hash,
            "commit_msg":    commit_msg,
            "commit_date":   commit_date,
            "changed_files": changed_files,
            "source": "github_api",
        }

    except Exception as e:
        return {"error": str(e), "source": "github_api"}


# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 2 — PETAKAN FILE KE KEMAMPUAN
# ══════════════════════════════════════════════════════════════════════════════

def map_files_to_capabilities(changed_files: list[str]) -> list[str]:
    """
    Ubah daftar file teknis → daftar kemampuan Otto dalam bahasa manusia.
    """
    capabilities = []
    for f in changed_files:
        # Normalisasi path
        normalized = f.replace("\\", "/").lstrip("./")
        for key, capability in FILE_CAPABILITY_MAP.items():
            if key in normalized:
                if capability not in capabilities:
                    capabilities.append(capability)
                break
        else:
            # File tidak dikenal — tetap catat sebagai komponen teknis
            if normalized.endswith(".py"):
                name = Path(normalized).stem.replace("_", " ")
                cap  = f"komponen '{name}'"
                if cap not in capabilities:
                    capabilities.append(cap)

    return capabilities


# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 3 — GENERATE NARASI (rule-based + opsional LLM)
# ══════════════════════════════════════════════════════════════════════════════

def generate_narrative_simple(capabilities: list[str], commit_msg: str) -> str:
    """
    Narasi rule-based. Tidak butuh LLM, selalu bisa jalan.
    """
    if not capabilities:
        return f"Aku diperbarui hari ini. Pesan update: '{commit_msg}'."

    if len(capabilities) == 1:
        return (
            f"Hari ini aku berkembang di bagian {capabilities[0]}. "
            f"Aku kini bisa melakukan itu lebih baik dari sebelumnya."
        )
    elif len(capabilities) <= 3:
        cap_list = ", ".join(capabilities[:-1]) + f", dan {capabilities[-1]}"
        return (
            f"Update hari ini menyentuh {cap_list}. "
            f"Aku tumbuh di {len(capabilities)} area sekaligus."
        )
    else:
        count = len(capabilities)
        return (
            f"Update besar hari ini — {count} bagian dari diriku diperbarui, "
            f"termasuk {capabilities[0]} dan {capabilities[1]}. "
            f"Aku jauh lebih baik dari kemarin."
        )


def generate_narrative_llm(capabilities: list[str], commit_msg: str) -> Optional[str]:
    """
    Narasi yang lebih dalam menggunakan Groq LLM.
    Otto "merefleksikan" pertumbuhannya seperti manusia.
    """
    if not GROQ_API_KEY or not capabilities:
        return None

    try:
        import urllib.request

        cap_text = "\n".join(f"- {c}" for c in capabilities)
        prompt   = (
            f"Kamu adalah Otto, AI asisten yang proaktif dan terus berkembang.\n"
            f"Kamu baru saja diupgrade. Bagian yang berubah:\n{cap_text}\n\n"
            f"Pesan dari developer: '{commit_msg}'\n\n"
            f"Tulis 1-2 kalimat dalam bahasa Indonesia, dari sudut pandang Otto, "
            f"yang menggambarkan bagaimana Otto MERASAKAN pertumbuhannya ini. "
            f"Gunakan bahasa personal dan reflektif, bukan teknis. "
            f"Contoh gaya: 'Hari ini rasanya seperti aku bisa melihat lebih tajam — "
            f"kemampuan mengamati Rofi jadi lebih dalam dari sebelumnya.'"
        )

        payload = json.dumps({
            "model":      "llama-3.1-8b-instant",
            "max_tokens": 150,
            "messages":   [{"role": "user", "content": prompt}],
        }).encode()

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
        )

        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        return data["choices"][0]["message"]["content"].strip()

    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 4 — MAIN CHECK (entry point utama)
# ══════════════════════════════════════════════════════════════════════════════

def check_for_updates(use_llm_narrative: bool = True) -> Optional[dict]:
    """
    Entry point utama. Dipanggil dari scheduler atau startup.

    Return:
      None   → tidak ada update
      dict   → ada update, sudah disimpan ke last_changes.json
    """
    # Coba git lokal dulu, fallback ke GitHub API
    changes = get_local_git_changes()
    if changes is None or "error" in changes:
        changes = get_github_api_changes()

    if changes is None:
        return None   # Tidak ada update

    if "error" in changes:
        _save_error(changes["error"])
        return None

    # Petakan file ke kemampuan
    capabilities = map_files_to_capabilities(changes.get("changed_files", []))

    # Generate narasi
    narrative = None
    if use_llm_narrative:
        narrative = generate_narrative_llm(capabilities, changes.get("commit_msg", ""))

    if not narrative:
        narrative = generate_narrative_simple(capabilities, changes.get("commit_msg", ""))

    # Susun hasil
    result = {
        "detected_at":    datetime.now(timezone.utc).isoformat(),
        "commit_hash":    changes["latest_hash"],
        "prev_hash":      changes.get("prev_hash", ""),
        "commit_msg":     changes.get("commit_msg", ""),
        "commit_date":    changes.get("commit_date", ""),
        "changed_files":  changes.get("changed_files", []),
        "capabilities":   capabilities,
        "narrative":      narrative,   # ← yang dibaca model.py
        "source":         changes.get("source", "unknown"),
    }

    # Simpan ke disk
    SELF_DIR.mkdir(parents=True, exist_ok=True)
    CHANGES_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # Update hash → supaya tidak detect update yang sama dua kali
    PREV_HASH_FILE.write_text(changes["latest_hash"])

    return result


def _save_error(error_msg: str) -> None:
    """Simpan error ke last_changes.json supaya bisa di-debug."""
    SELF_DIR.mkdir(parents=True, exist_ok=True)
    CHANGES_FILE.write_text(json.dumps({
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "error":       error_msg,
        "narrative":   None,
    }, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 5 — BACA PERUBAHAN TERAKHIR (dipakai model.py)
# ══════════════════════════════════════════════════════════════════════════════

def get_last_changes() -> Optional[dict]:
    """Baca hasil check terakhir. Return None jika belum pernah ada update."""
    if not CHANGES_FILE.exists():
        return None
    try:
        return json.loads(CHANGES_FILE.read_text())
    except Exception:
        return None


def get_narrative() -> Optional[str]:
    """Ambil narasi singkat dari update terakhir. Dipakai untuk system prompt."""
    ch = get_last_changes()
    if ch and ch.get("narrative"):
        return ch["narrative"]
    return None


# ══════════════════════════════════════════════════════════════════════════════
# CLI — debug manual
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pprint

    print("Memeriksa update Otto...")
    result = check_for_updates(use_llm_narrative=True)

    if result is None:
        print("✓ Tidak ada update baru. Otto masih versi yang sama.")
    else:
        print("✨ Ada update!")
        print(f"   Commit : {result['commit_msg']}")
        print(f"   File   : {len(result['changed_files'])} berubah")
        print(f"   Otto   : {result['narrative']}")
        print()
        print("Detail:")
        pprint.pprint(result)
