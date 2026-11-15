"""Live Deja: bulk-load the learned graph into HydraDB, then serve the page + /api/loadout (Deja vs vector, side by side).
The last HOLDOUT tasks of the shuffled stream are never learned from; they are the page's presets."""
import json, os, random, threading, time, http.server, urllib.parse
import numpy as np
import retr
from hydra_store import HydraLearned

HOLDOUT, BUDGET, PORT = 200, 2000, int(os.environ.get("PORT", 8090))
cat, tasks = retr.build(); names = list(cat); ix = {n: k for k, n in enumerate(names)}
TE = retr.embed(None, "data/tool_emb.npy"); Q = retr.embed(None, "data/q_emb.npy")
order = list(range(len(tasks))); random.Random(1).shuffle(order)
train, held = [tasks[i] for i in order[:-HOLDOUT]], order[-HOLDOUT:]
L = HydraLearned(); t0 = time.time(); nl, nc = L.bulk_load(train)
print(f"graph loaded: {len(train)} tasks, {nl} term links, {nc} co-use links in {time.time()-t0:.0f}s", flush=True)
presets = [{"id": i, "q": tasks[i]["q"], "multi": len(tasks[i]["gold_servers"]) > 1} for i in held]
from fastembed import TextEmbedding
EMB = TextEmbedding("BAAI/bge-small-en-v1.5", threads=2)

def tool(n, **kw): return {"name": n, "server": cat[n]["server"], "tokens": cat[n]["tokens"], **kw}

LOCK = threading.Lock()  # ponytail: one Bolt session + 'last rows' shared state; per-request sessions if traffic matters

def loadout(q, qvec, budget=BUDGET):
    with LOCK: return _loadout(q, qvec, budget)

def _loadout(q, qvec, budget):
    t = time.perf_counter()
    sim = TE @ qvec; o = np.argsort(-sim); V = [names[k] for k in o[:400]]
    ts = L.term_scores(q); cs = L.co_scores(V[:5]); graph_ms = (time.perf_counter() - t) * 1000
    cand = set(V[:200]) | set(ts) | set(cs)
    D = sorted(cand, key=lambda n: (-round(float(sim[ix[n]] + 0.5 * ts.get(n, 0) + 0.2 * cs.get(n, 0)), 9), n))
    why_t = {}
    for r in getattr(L, "last_term_rows", []): why_t.setdefault(r["tool"], []).append([r["term"], r["n"]])
    why_c = {r["b"]: [r["a"], r["n"]] for r in getattr(L, "last_co_rows", [])}
    return {"deja": [tool(n, sim=round(float(sim[ix[n]]), 3), term=round(ts.get(n, 0), 3), co=round(cs.get(n, 0), 3),
                          why_terms=sorted(why_t.get(n, []), key=lambda x: -x[1])[:4], why_co=why_c.get(n)) for n in retr.pack(D, cat, budget)],
            "vector": [tool(n, sim=round(float(sim[ix[n]]), 3)) for n in retr.pack(V, cat, budget)],
            "graph_ms": round(graph_ms, 1), "budget": budget, "catalog": len(cat)}

class H(http.server.SimpleHTTPRequestHandler):
    def __init__(s, *a, **k): super().__init__(*a, directory="web", **k)
    def j(s, obj, code=200):
        b = json.dumps(obj).encode(); s.send_response(code)
        s.send_header("Content-Type", "application/json"); s.send_header("Content-Length", str(len(b))); s.end_headers(); s.wfile.write(b)
    def do_GET(s):
        u = urllib.parse.urlparse(s.path); p = urllib.parse.parse_qs(u.query)
        if u.path == "/api/presets": return s.j(presets)
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
    """The 200 never-learned tasks, scored with reads going through HydraDB, checked against the in-memory mirror."""
    ok = {"deja": 0, "vector": 0}; same = 0; lat = []
    for i in held:
        t = tasks[i]; r = _loadout(t["q"], Q[i], BUDGET); lat.append(r["graph_ms"])
        for m in ok: ok[m] += t["gold"] <= {x["name"] for x in r[m]}
        sim = TE @ Q[i]; V = [names[k] for k in np.argsort(-sim)[:400]]
        ts, cs = retr.Learned.term_scores(L, t["q"]), retr.Learned.co_scores(L, V[:5])
        D = sorted(set(V[:200]) | set(ts) | set(cs), key=lambda n: (-round(float(sim[ix[n]] + 0.5 * ts.get(n, 0) + 0.2 * cs.get(n, 0)), 9), n))
        same += retr.pack(D, cat, BUDGET) == [x["name"] for x in r["deja"]]
    lat.sort(); n = len(held)
    out = {"n": n, "deja": ok["deja"] / n, "vector": ok["vector"] / n, "matches_oracle": same == n,
           "latency_ms": {"p50": lat[n // 2], "p95": lat[int(n * .95)]}}
    print("held-out via HydraDB:", out, flush=True); return out

if __name__ == "__main__":
    R = json.load(open(f"data/results_{BUDGET}_k{retr.Learned.K}_n{retr.Learned.MIN_N}.json"))
    R["hydra"] = holdout_check()
    if os.path.exists("data/load.json"): R["load"] = json.load(open("data/load.json"))
    json.dump(R, open("web/results.json", "w"))
    print(f"serving on :{PORT}", flush=True)
    http.server.ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
