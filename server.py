"""Live Deja: bulk-load the learned graph into HydraDB, then serve the page + /api/loadout (Deja vs vector, side by side).
The last HOLDOUT tasks of the shuffled stream are never learned from; they are the page's presets."""
import json, os, random, threading, time, http.server, urllib.parse
import numpy as np
import retr
from hydra_store import HydraLearned, hid, twin

HOLDOUT, BUDGET, PORT = 200, 2000, int(os.environ.get("PORT", 8090))
cat, tasks = retr.build(); names = list(cat); ix = {n: k for k, n in enumerate(names)}
TE = retr.embed(None, "data/tool_emb.npy"); Q = retr.embed(None, "data/q_emb.npy")
order = list(range(len(tasks))); random.Random(1).shuffle(order)
train, held = [tasks[i] for i in order[:-HOLDOUT]], order[-HOLDOUT:]
retr.Learned.Q = Q; HydraLearned.tool_names = {hid("tool", n): n for n in cat}
from neo4j.exceptions import ServiceUnavailable, NotALeader
for attempt in range(60):   # after a reboot HydraDB holds the old write lease for ~30 s; wait instead of crash-looping
    try:
        L = HydraLearned(); t0 = time.time(); nt, nl, nc = L.bulk_load(train); break
    except (ServiceUnavailable, NotALeader) as e:
        print(f"HydraDB not writable yet ({type(e).__name__}), retrying", flush=True); time.sleep(5)
else: raise SystemExit("HydraDB never became writable")
print(f"graph loaded: {nt} task nodes, {nl} term links, {nc} co-use links in {time.time()-t0:.0f}s", flush=True)
presets = [{"id": i, "q": tasks[i]["q"], "multi": len(tasks[i]["gold_servers"]) > 1} for i in held]; EXAMPLES = []
from fastembed import TextEmbedding
EMB = TextEmbedding("BAAI/bge-small-en-v1.5", threads=2)

def tool(n, **kw): return {"name": n, "server": cat[n]["server"], "tokens": cat[n]["tokens"], **kw}

LOCK = threading.Lock()  # ponytail: one Bolt session + 'last rows' shared state; per-request sessions if traffic matters

def loadout(q, qvec, budget=BUDGET):
    with LOCK: return _loadout(q, qvec, budget)

def _loadout(q, qvec, budget, store=None):
    st = store or L; t = time.perf_counter()
    sim = TE @ qvec; V = [names[k] for k in np.argsort(-sim, kind="stable")[:400]]
    ranked, parts = retr.deja_rank(st, q, qvec, sim, V, ix); graph_ms = (time.perf_counter() - t) * 1000
    if store is not None: return retr.pack(ranked, cat, budget)
    nb = dict(L.neighbours(qvec)); why_k = {}
    for i, used in getattr(L, "last_votes", {}).items():
        for g in used: why_k.setdefault(g, []).append([tasks[i]["q"][:90], round(nb.get(i, 0), 2)])
    why_t = {}
    for r in getattr(L, "last_term_rows", []): why_t.setdefault(r["tool"], []).append([r["term"], r["n"]])
    why_c = {r["b"]: [r["a"], r["n"]] for r in getattr(L, "last_co_rows", [])}
    return {"deja": [tool(n, sim=round(float(sim[ix[n]]), 3), knn=round(parts["knn"].get(n, 0), 3), term=round(parts["terms"].get(n, 0), 3),
                          co=round(parts["co"].get(n, 0), 3), why_knn=sorted(why_k.get(n, []), key=lambda x: -x[1])[:2],
                          why_terms=sorted(why_t.get(n, []), key=lambda x: -x[1])[:4], why_co=why_c.get(n)) for n in retr.pack(ranked, cat, budget)],
            "vector": [tool(n, sim=round(float(sim[ix[n]]), 3)) for n in retr.pack(V, cat, budget)],
            "graph_ms": round(graph_ms, 1), "budget": budget, "catalog": len(cat), "memory": len(L.past)}

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
                    r = loadout(tasks[i]["q"], Q[i]); r["gold"] = sorted(tasks[i]["gold"]); r["q"] = tasks[i]["q"]; return s.j(r)
                q = (p.get("q") or [""])[0].strip()[:500]           # untrusted input: bounded, only ever a Cypher *parameter*
                if not q: return s.j({"error": "empty task"}, 400)
                v = np.array(next(iter(EMB.embed([q]))), dtype=np.float32); v /= np.linalg.norm(v)
                r = loadout(q, v); r["q"] = q; return s.j(r)
            except Exception as e:
                return s.j({"error": type(e).__name__}, 500)
        return super().do_GET()
    def log_message(s, *a): pass

def holdout_check():
    """The 200 never-learned tasks, scored with reads going through HydraDB, checked against the in-memory twin."""
    ok = {"deja": 0, "vector": 0, "knn": 0}; same = 0; lat = []; T = twin(L); wins = []
    for i in held:
        t = tasks[i]; r = _loadout(t["q"], Q[i], BUDGET); lat.append(r["graph_ms"])
        hit = {m: t["gold"] <= {x["name"] for x in r[m]} for m in ("deja", "vector")}
        for m in hit: ok[m] += hit[m]
        if hit["deja"] and not hit["vector"]: wins.append((len(t["gold_servers"]) > 1, len(t["gold"]), i))
        same += _loadout(t["q"], Q[i], BUDGET, store=T) == [x["name"] for x in r["deja"]]
        sim = TE @ Q[i]; V = [names[k] for k in np.argsort(-sim, kind="stable")[:400]]; kn = T.knn_scores(Q[i])
        ok["knn"] += t["gold"] <= set(retr.pack(retr.rank(set(V[:200]) | set(kn), lambda n: sim[ix[n]] + retr.LAM * kn.get(n, 0)), cat, BUDGET))
    lat.sort(); n = len(held)
    out = {"n": n, **{m: v / n for m, v in ok.items()}, "matches_oracle": same == n,
           "latency_ms": {"p50": lat[n // 2], "p95": lat[int(n * .95)]}}
    for multi, _, i in sorted(wins, reverse=True)[:4]:    # verified examples: Deja loads every needed tool, tool search doesn't
        q = tasks[i]["q"]; EXAMPLES.append({"id": i, "label": ("[2 servers] " if multi else "") + (q[:72] + "…" if len(q) > 72 else q)})
    print("held-out via HydraDB:", out, f"| {len(wins)} held-out tasks where Deja wins outright", flush=True); return out

if __name__ == "__main__":
    R = json.load(open(f"data/results_v2_{BUDGET}.json"))
    R["budgets"] = {b: json.load(open(f"data/results_v2_{b}.json"))["orders"]["1"]["overall"] for b in (1000, 2000, 4000)
                    if os.path.exists(f"data/results_v2_{b}.json")}
    R["hydra"] = holdout_check()
    if os.path.exists("data/load.json"): R["load"] = json.load(open("data/load.json"))
    json.dump(R, open("web/results.json", "w"))
    print(f"serving on :{PORT}", flush=True)
    http.server.ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
