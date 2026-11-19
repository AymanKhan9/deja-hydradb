"""Live Deja: bulk-load the learned graph into HydraDB, then serve the page + /api/loadout (Deja vs tool search, side by side).
The last HOLDOUT tasks of the shuffled stream are never learned from; they are the page's presets.

Served model: learns only from tool calls that worked; reads the tool catalog from the graph ((:Catalog)-[:LISTS]->(:Tool));
abstains when nothing fits (policy fit by abstain.py). The cold start is handled by retr's memory ramp, not a fallback."""
import json, os, random, threading, time, http.server, urllib.parse
import numpy as np
import retr
from hydra_store import HydraLearned, hid, twin, write_catalog, read_catalog, catalog_version

HOLDOUT, BUDGET, PORT = 200, 2000, int(os.environ.get("PORT", 8090))
retr.Learned.OUTCOME = True
cat, tasks = retr.build(); names = list(cat); ix = {n: k for k, n in enumerate(names)}
TE = retr.embed(None, "data/tool_emb.npy"); Q = retr.embed(None, "data/q_emb.npy")
order = list(range(len(tasks))); random.Random(1).shuffle(order)
train, held = [tasks[i] for i in order[:-HOLDOUT]], order[-HOLDOUT:]
retr.Learned.Q = Q; HydraLearned.tool_names = {hid("tool", n): n for n in cat}
from neo4j.exceptions import ServiceUnavailable, NotALeader
for attempt in range(60):   # after a reboot HydraDB holds the old write lease for ~30 s; wait instead of crash-looping
    try:
        L = HydraLearned(); t0 = time.time(); nt, nl, nc = L.bulk_load(train); write_catalog(L.db, names); break
    except (ServiceUnavailable, NotALeader) as e:
        print(f"HydraDB not writable yet ({type(e).__name__}), retrying", flush=True); time.sleep(5)
else: raise SystemExit("HydraDB never became writable")
print(f"graph loaded: {nt} task nodes, {nl} term links, {nc} co-use links, {len(names)} catalog tools in {time.time()-t0:.0f}s", flush=True)
presets = [{"id": i, "q": tasks[i]["q"], "multi": len(tasks[i]["gold_servers"]) > 1} for i in held]; EXAMPLES = []

AB = json.load(open("data/abstain.json")); POL = AB["policies"]["similarity only"]; PRM = POL["params"]
THR = POL["threshold"]
def confidence(s_mem, s_tool):
    z = (np.array([s_mem, s_tool]) - np.array(PRM["mu"])) / np.array(PRM["sd"])
    return float(1 / (1 + np.exp(-(z @ np.array(PRM["w"]) + PRM["b"]))))

calls, fails = {}, {}
for t in tasks:
    for g in t["gold"]: calls[g] = calls.get(g, 0) + 1
    for g in t["failed"]: fails[g] = fails.get(g, 0) + 1
BROKEN = {g for g in calls if calls[g] >= 10 and fails.get(g, 0) / calls[g] >= 0.8}   # fail in >= 80% of >= 10 tasks

_cat = {"v": "unread", "set": None, "ix": None}
def catalog():
    """Current catalog from HydraDB. Each call reads the Catalog node's version (one small read); the tool list itself is
    re-read only when the version changed, so a removed tool is never served after its removal returns."""
    v = catalog_version(L.db)
    if v != _cat["v"]:
        s = read_catalog(L.db); _cat.update(v=v, set=s, ix=np.array(sorted(ix[n] for n in s)))
    return _cat["set"], _cat["ix"]

from fastembed import TextEmbedding
EMB = retr.EMB = TextEmbedding("BAAI/bge-small-en-v1.5", threads=2)      # shared with retr's clause embedding
def tool(n, **kw): return {"name": n, "server": cat[n]["server"], "tokens": cat[n]["tokens"], "broken": n in BROKEN, **kw}
LOCK = threading.Lock()  # ponytail: one Bolt session + 'last rows' shared state; per-request sessions if traffic matters

def loadout(q, qvec, budget=BUDGET):
    with LOCK: return _loadout(q, qvec, budget)

def _loadout(q, qvec, budget, store=None):
    st = store or L; t = time.perf_counter(); active, aix = catalog(); aix_all = aix   # tool search has no outcome memory
    held = st.quarantined()                                  # graph (Quarantine-HOLDS->Tool) or, for the twin, the record
    if held: active = active - held; aix = np.array([k for k in aix if names[k] in active])
    sim = TE @ qvec; V = [names[aix[k]] for k in np.argsort(-sim[aix], kind="stable")[:400]]
    nb = st.neighbours(qvec); s_mem = nb[0][1] if nb else 0.0; s_tool = float(sim[aix].max()); conf = confidence(s_mem, s_tool)
    mode = "abstain" if conf < THR else "deja"
    ranked, parts = retr.deja_rank(st, q, qvec, sim, V, ix, active=active) if mode == "deja" else ([], {"knn": {}, "terms": {}, "co": {}, "clause": {}})
    graph_ms = (time.perf_counter() - t) * 1000
    picked = [] if mode == "abstain" else retr.pack(ranked, cat, budget)                                   # comparison column: hybrid tool search
    Vall = [names[aix_all[k]] for k in np.argsort(-sim[aix_all], kind="stable")[:400]]                   # over the full live catalog
    if store is not None: return mode, picked
    nbd = dict(nb); why_k = {}
    if mode == "deja":
        for i, used in getattr(L, "last_votes", {}).items():
            for g in used * (i in nbd):                      # whole-task neighbours only (the read also covers the steps)
                why_k.setdefault(g, []).append([tasks[i]["q"][:90], round(nbd.get(i, 0), 2)])
    why_t, why_c = {}, {}
    if mode == "deja":
        for r in getattr(L, "last_term_rows", []): why_t.setdefault(r["tool"], []).append([r["term"], r["n"]])
        why_c = {r["b"]: [r["a"], r["n"]] for r in getattr(L, "last_co_rows", [])}
    reason = {"abstain": f"Nothing in the catalog looks like it fits: the closest past task is {s_mem:.2f} similar and the best tool {s_tool:.2f}. "
                         "Deja loads nothing and would ask the user instead of guessing.",
              "deja": ""}[mode]
    return {"mode": mode, "reason": reason, "confidence": round(conf, 3), "s_mem": round(s_mem, 3), "s_tool": round(s_tool, 3),
            "deja": [tool(n, sim=round(float(sim[ix[n]]), 3), knn=round(parts["knn"].get(n, 0), 3), term=round(parts["terms"].get(n, 0), 3),
                          co=round(parts["co"].get(n, 0), 3), clause=round(parts["clause"].get(n, 0), 3), why_knn=sorted(why_k.get(n, []), key=lambda x: -x[1])[:2],
                          why_terms=sorted(why_t.get(n, []), key=lambda x: -x[1])[:4], why_co=why_c.get(n)) for n in picked],
            "vector": [tool(n, sim=round(float(sim[ix[n]]), 3)) for n in retr.pack(retr.search_rank(q, sim, Vall, ix), cat, budget)],
            "graph_ms": round(graph_ms, 1), "budget": budget, "catalog": len(active), "memory": len(L.past), "quarantined": sorted(held)[:20],
            "n_quarantined": len(held)}

class H(http.server.SimpleHTTPRequestHandler):
    def __init__(s, *a, **k): super().__init__(*a, directory="web", **k)
    def j(s, obj, code=200):
        b = json.dumps(obj).encode(); s.send_response(code)
        s.send_header("Content-Type", "application/json"); s.send_header("Content-Length", str(len(b))); s.end_headers(); s.wfile.write(b)
    def do_GET(s):
        u = urllib.parse.urlparse(s.path); p = urllib.parse.parse_qs(u.query)
        if u.path == "/api/presets": return s.j({"presets": presets, "examples": EXAMPLES})
        if u.path == "/api/loadout":
            try:
                if "id" in p:
                    i = int(p["id"][0])
                    if i not in held: return s.j({"error": "unknown preset"}, 400)
                    r = loadout(tasks[i]["q"], Q[i]); r.update(gold=sorted(tasks[i]["gold_ok"]), failed=sorted(tasks[i]["failed"]), q=tasks[i]["q"]); return s.j(r)
                q = (p.get("q") or [""])[0].strip()[:500]           # untrusted input: bounded, only ever a Cypher *parameter*
                if not q: return s.j({"error": "empty task"}, 400)
                v = np.array(next(iter(EMB.embed([q]))), dtype=np.float32); v /= np.linalg.norm(v)
                r = loadout(q, v); r["q"] = q; return s.j(r)
            except Exception as e:
                return s.j({"error": type(e).__name__}, 500)
        return super().do_GET()
    def log_message(s, *a): pass

def holdout_check():
    """The 200 never-learned tasks (all answerable), read through HydraDB and checked against the in-memory twin.
    Success = every tool that WORKED in the recorded run is in the loadout (abstaining counts as a miss)."""
    T = twin(L); n = len(held); same = 0; lat = []; wins = []; c = {"deja": 0, "vector": 0, "deja_called": 0, "deja_broken": 0, "vector_broken": 0}
    modes = {"deja": 0, "abstain": 0}; scored = 0
    for i in held:
        t = tasks[i]; r = _loadout(t["q"], Q[i], BUDGET); lat.append(r["graph_ms"]); modes[r["mode"]] += 1
        lo, lv = {x["name"] for x in r["deja"]}, {x["name"] for x in r["vector"]}
        same += _loadout(t["q"], Q[i], BUDGET, store=T) == (r["mode"], [x["name"] for x in r["deja"]])
        c["deja_broken"] += bool(lo & BROKEN); c["vector_broken"] += bool(lv & BROKEN); c["deja_called"] += t["gold"] <= lo
        if not t["gold_ok"]: continue
        scored += 1; hd, hv = t["gold_ok"] <= lo, t["gold_ok"] <= lv; c["deja"] += hd; c["vector"] += hv
        if hd and not hv: wins.append((len(t["gold_servers"]) > 1, len(t["gold_ok"]), i))
    lat.sort()
    out = {"n": n, "scored": scored, "deja": c["deja"] / scored, "vector": c["vector"] / scored, "deja_called": c["deja_called"] / n,
           "deja_broken": c["deja_broken"] / n, "vector_broken": c["vector_broken"] / n, "modes": modes,
           "matches_oracle": same == n, "latency_ms": {"p50": lat[n // 2], "p95": lat[int(n * .95)]}}
    for multi, _, i in sorted(wins, reverse=True)[:4]:    # verified examples: Deja loads every tool that worked, tool search doesn't
        q = tasks[i]["q"]; EXAMPLES.append({"id": i, "label": ("[2 servers] " if multi else "") + (q[:72] + "…" if len(q) > 72 else q)})
    print("held-out via HydraDB:", out, f"| {len(wins)} held-out tasks where Deja wins outright", flush=True); return out

if __name__ == "__main__":
    R = json.load(open(f"data/results_v2_{BUDGET}.json"))
    R["budgets"] = {b: json.load(open(f"data/results_v2_{b}.json"))["orders"]["1"]["overall"] for b in (1000, 2000, 4000)
                    if os.path.exists(f"data/results_v2_{b}.json")}
    for k in ("outcome", "lifecycle", "load", "analysis_v5", "catalog_check"):
        if os.path.exists(f"data/{k}.json"): R[k.split("_")[0]] = json.load(open(f"data/{k}.json"))
    R.get("lifecycle", {}).pop("scenario_cases", None)
    if os.path.exists("data/bench.json"): R["bench"] = json.load(open("data/bench.json"))
    if os.path.exists("data/agent_check.json"): R["agent"] = {k: v for k, v in json.load(open("data/agent_check.json")).items() if k != "rows"}
    R["abstain"] = {k: v for k, v in AB.items() if k != "setup"} | {"setup": AB["setup"], "served_threshold": THR}
    R["hydra"] = holdout_check()
    json.dump(R, open("web/results.json", "w"))
    print(f"serving on :{PORT}", flush=True)
    http.server.ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
