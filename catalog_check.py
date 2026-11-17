"""Tool removal through the server's own loadout path (the function behind /api/loadout), not a direct graph read.
For held-out tasks, take a tool the live loadout contains, remove it from the catalog, and ask again: immediately and
after 6 s (past the old 5 s catalog cache). The removed tool must never appear, in Deja's column or tool search's.
Then put the tools back and check the server sees them again.   uv run python catalog_check.py   (needs HydraDB up)"""
import json, time
import server as S
from hydra_store import remove_tool, write_catalog

def names_in(r): return {x["name"] for col in ("deja", "vector") for x in r[col]}

cases, used = [], set()
for i in S.held:
    if len(cases) == 20: break
    r = S.loadout(S.tasks[i]["q"], S.Q[i])
    A = next((x["name"] for x in r["deja"] if x["name"] not in used), None) if r["mode"] == "deja" else None
    if A: cases.append({"task": i, "tool": A}); used.add(A)
for c in cases: remove_tool(S.L.db, c["tool"])                       # the catalog changes; the server's cache is warm
for when in ("immediately", "after_6s"):
    if when == "after_6s": time.sleep(6)
    for c in cases: c[when] = c["tool"] not in names_in(S.loadout(S.tasks[c["task"]]["q"], S.Q[c["task"]]))
write_catalog(S.L.db, [c["tool"] for c in cases])                    # the tools come back
back = sum(c["tool"] in S.catalog()[0] for c in cases)
out = {"cases": len(cases), "absent_immediately": sum(c["immediately"] for c in cases), "absent_after_6s": sum(c["after_6s"] for c in cases),
       "back_in_catalog_after_readd": back, "catalog_size": len(S.catalog()[0])}
json.dump(out, open("data/catalog_check.json", "w"), indent=1); print(out)
assert out["absent_immediately"] == out["absent_after_6s"] == out["back_in_catalog_after_readd"] == len(cases) > 0
