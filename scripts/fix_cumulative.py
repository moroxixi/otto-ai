# script sekali jalan: fix_cumulative.py
import json
from pathlib import Path

f = Path("/data/asd/otto-ai/data/growth/history.json")
weeks = json.loads(f.read_text())
weeks.sort(key=lambda w: (w["year"], w["week_number"]))

running = 0
for w in weeks:
    running += w.get("score_total", 0)
    w["cumulative_total"] = running

f.write_text(json.dumps(weeks, indent=2, ensure_ascii=False))
print("Done:", [(w["week_number"], w["cumulative_total"]) for w in weeks])
