import json
from pathlib import Path

path = Path("/data/asd/otto-ai/data/hypotheses.json")
data = json.loads(path.read_text())
cleaned = [h for h in data if h["id"] != "6a2aa743"]
path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2))
print(f"Selesai. {len(data) - len(cleaned)} entry dihapus.")
