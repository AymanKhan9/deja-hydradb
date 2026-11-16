"""Deja's learned graph in HydraDB.

  (:Term {key, df}) -[:LED]-> (:Link {n}) -[:TO]-> (:Tool {key, uses})      task word -> tool that solved it
  (:Tool) -[:CO]-> (:CoLink {n}) -[:TO]-> (:Tool)                          tools used together in one task
  (:Meta {n_tasks})

Counts sit on reified Link nodes because HydraDB's batch writes can set node properties but not edge properties.
Readers (agents) only query HydraDB. The single learner keeps an in-memory mirror of the counts so each write is one
absolute-value batch instead of read-modify-write; the mirror is also the oracle the graph reads are checked against.
"""
import hashlib, math, collections, sys
from neo4j import GraphDatabase
from retr import Learned, terms

def hid(kind, key):  # stable non-negative vertex id (< 2^52 so it survives JSON)
    return int(hashlib.md5(f"{kind}\x1f{key}".encode()).hexdigest()[:13], 16)

META = hid("meta", "deja")

class Hydra:
    def __init__(s, token_file=".hydradb/auth-token", url="bolt://127.0.0.1:7687"):
        s.drv = GraphDatabase.driver(url, auth=("neo4j", open(token_file).read().strip()))
        s.ses = s.drv.session(database="default")
    def q(s, cypher, **p): return [r.data() for r in s.ses.run(cypher, **p)]
    def nodes(s, label, rows, props):
        sets = ", ".join(f"n.{k} = row.{k}" for k in props)
        for i in range(0, len(rows), 1000):
            s.q(f"UNWIND $rows AS row MERGE (n {{id: row.id}}) SET n:{label}, {sets}", rows=rows[i:i + 1000])
    def edges(s, rel, pairs):
        rows = [{"src": a, "dst": b} for a, b in pairs]
        for i in range(0, len(rows), 1000):
            s.q(f"UNWIND $rows AS row CREATE (a {{id: row.src}})-[:{rel}]->(b {{id: row.dst}})", rows=rows[i:i + 1000])
    def any_of(s, var, keys):  # scalar params only (list params are rejected outside UNWIND)
        return " OR ".join(f"{var}.key = $k{i}" for i in range(len(keys))), {f"k{i}": k for i, k in enumerate(keys)}

class HydraLearned(Learned):
    """Same interface as the in-memory Learned; the three reads (task votes, term links, co-use links) are HydraDB
    traversals. Picking the K nearest past tasks is plain vector search (numpy) — the graph supplies what they used."""
    tool_names = {}                                   # vertex id -> tool name (decoding only; set from the catalog)
    def __init__(s, db=None):
        super().__init__(); s.db = db or Hydra()
    def _write(s, tasks, lk, co, tools):
        db = s.db
        db.nodes("Task", [{"id": hid("task", t["i"]), "i": t["i"]} for t in tasks], ["i"])
        db.nodes("Term", [{"id": hid("term", w), "key": w, "df": s.df[w]} for w in {w for w, _ in lk}], ["key", "df"])
        db.nodes("Tool", [{"id": hid("tool", g), "key": g, "uses": s.uses[g]} for g in tools], ["key", "uses"])
        db.nodes("Link", [{"id": hid("link", w + "\x1f" + g), "n": s.led[w][g]} for w, g in lk], ["n"])
        db.nodes("CoLink", [{"id": hid("co", a + "\x1f" + b), "n": s.co[a][b]} for a, b in co], ["n"])
        db.nodes("Meta", [{"id": META, "n_tasks": s.N}], ["n_tasks"])
        db.edges("USED", [(hid("task", t["i"]), hid("tool", g)) for t in tasks for g in s.view(t)["gold"]])
        if s.OUTCOME:   # failures are recorded too (edge type = outcome); votes read USED only
            db.edges("FAILED", [(hid("task", t["i"]), hid("tool", g)) for t in tasks for g in t.get("failed", ())])
        db.edges("LED", [(hid("term", w), hid("link", w + "\x1f" + g)) for w, g in lk])
        db.edges("TO", [(hid("link", w + "\x1f" + g), hid("tool", g)) for w, g in lk] + [(hid("co", a + "\x1f" + b), hid("tool", b)) for a, b in co])
        db.edges("CO", [(hid("tool", a), hid("co", a + "\x1f" + b)) for a, b in co])
    def ingest(s, task):
        """Incremental write after each task. Task memory is append-only; counts rewrite the touched terms' current top-K
        links. A link that later drops out of a top K stays in the graph, and term_scores cuts it at read time."""
        super().ingest(task)
        lk = [(w, g) for w in terms(task["q"]) for g, _ in s.links(w)]
        v = s.view(task)
        co = {(a, b) for a in v["gold"] for b in v["gold"] if a != b} | {tuple(f) for f in v["flows"]}
        s._write([task], lk, co, set(v["gold"]) | {b for _, b in co})
    def bulk_load(s, tasks):
        """Boot path: learn from many tasks in memory, then write the whole graph in 1000-row batches (labels first)."""
        for t in tasks: Learned.ingest(s, t)
        lk = [(w, g) for w in s.led for g, _ in s.links(w)]; co = [(a, b) for a, c in s.co.items() for b in c]
        s._write(tasks, lk, co, set(s.uses) | {b for _, b in co} | {a for a, _ in co})
        return len(tasks), len(lk), len(co)
    def votes(s, nb):
        """Task -USED-> Tool for the neighbour tasks: one id-anchored batched read, no property hydration."""
        if not nb: return {}
        rows = s.db.q("UNWIND $rows AS row MATCH (a {id: row.id})-[:USED]->(b) RETURN row.id AS src, b.id AS dst",
                      rows=[{"id": hid("task", i)} for i, _ in nb])
        back = {hid("task", i): i for i, _ in nb}; out = collections.defaultdict(list)
        for r in rows: out[back[r["src"]]].append(s.tool_names[r["dst"]])
        s.last_votes = {i: sorted(v) for i, v in out.items()}
        return s.last_votes
    def term_scores(s, q):
        ts = sorted(terms(q))
        if not ts: return {}
        where, p = s.db.any_of("t", ts)
        dfs = {r["term"]: r["df"] for r in s.db.q(f"MATCH (t:Term) WHERE {where} RETURN t.key AS term, t.df AS df", **p)}
        top = sorted(dfs, key=lambda w: (dfs[w], w))[:s.TOPT]          # highest idf first; only these get expanded
        s.last_term_rows = []
        if not top: return {}
        where, p = s.db.any_of("t", top)
        rows = s.db.q(f"MATCH (t:Term)-[:LED]->(l:Link)-[:TO]->(x:Tool) WHERE {where} RETURN t.key AS term, t.df AS df, x.key AS tool, l.n AS n", **p)
        # Same cut as the twin: per term keep the top K by (-n, tool) with n >= MIN_N. Incremental ingest can leave a link
        # that fell out of a term's top K in the graph; its stored count never exceeds its true count and every current
        # top-K link is rewritten whenever the term is touched, so this cut reproduces the twin exactly.
        byterm = collections.defaultdict(list)
        for r in rows: byterm[r["term"]].append(r)
        rows = [r for rs in byterm.values() for r in sorted(rs, key=lambda r: (-r["n"], r["tool"]))[:s.K] if r["n"] >= s.MIN_N]
        N = (s.db.q("MATCH (m:Meta {id: $id}) RETURN m.n_tasks AS n", id=META) or [{"n": 0}])[0]["n"]
        out, idf = collections.Counter(), {}
        s.last_term_rows = rows                      # kept for the UI's "why" view
        for r in rows:
            idf[r["term"]] = math.log((N + 1) / (r["df"] + 0.5))
            out[r["tool"]] += idf[r["term"]] * r["n"] / r["df"]
        z = sum(idf.values())
        return {g: v / z for g, v in out.items()} if z else {}
    def co_scores(s, seeds):
        if not seeds: return {}
        where, p = s.db.any_of("a", seeds)
        rows = s.db.q(f"MATCH (a:Tool)-[:CO]->(l:CoLink)-[:TO]->(b:Tool) WHERE {where} RETURN a.key AS a, a.uses AS uses, b.key AS b, l.n AS n", **p)
        rank = {a: r for r, a in enumerate(seeds)}; out = collections.Counter(); s.last_co_rows = rows
        for r in rows: out[r["b"]] = max(out[r["b"]], r["n"] / r["uses"] / (1 + rank[r["a"]]))
        return out

CATALOG = hid("catalog", "deja")

def write_catalog(db, tool_names):
    """The live catalog is graph state: (:Catalog)-[:LISTS]->(:Tool). Labels first, then edges."""
    db.nodes("Catalog", [{"id": CATALOG, "key": "catalog"}], ["key"])
    db.nodes("Tool", [{"id": hid("tool", n), "key": n} for n in tool_names], ["key"])
    db.edges("LISTS", [(CATALOG, hid("tool", n)) for n in tool_names])

def remove_tool(db, name):
    """tools/list changed: the tool leaves the catalog. Its USED/CO/LED history stays; reads just stop returning it."""
    db.q("MATCH (c {id: $c})-[r:LISTS]->(t {id: $t}) DELETE r", c=CATALOG, t=hid("tool", name))

def read_catalog(db):
    """Current catalog, one id-only batched read."""
    rows = db.q("UNWIND $rows AS row MATCH (a {id: row.id})-[:LISTS]->(b) RETURN row.id AS src, b.id AS dst", rows=[{"id": CATALOG}])
    return {HydraLearned.tool_names[r["dst"]] for r in rows}

def twin(h):
    """The same learned state, read from memory instead of HydraDB (shares the mirror; nothing is copied)."""
    import copy
    t = copy.copy(h); t.__class__ = Learned; return t

if __name__ == "__main__":
    # consistency check: graph reads must equal the in-memory twin (all three reads, then the final loadout)
    import retr, numpy as np
    cat, tasks = retr.build(); names = list(cat); nix = {n: k for k, n in enumerate(names)}
    TE, Q = retr.embed(None, "data/tool_emb.npy"), retr.embed(None, "data/q_emb.npy")
    Learned.Q = Q; HydraLearned.tool_names = {hid("tool", n): n for n in cat}
    Learned.OUTCOME = "--all-calls" not in sys.argv          # check the served mode by default
    H = HydraLearned(); H.bulk_load(tasks[:300])
    for t in tasks[300:400]: H.ingest(t)                 # the per-task learning path, one write batch per task
    T = twin(H); bad = 0
    for t in tasks[400:500]:
        sim = TE @ Q[t["i"]]; V = [names[k] for k in np.argsort(-sim, kind="stable")[:400]]
        a = retr.deja_rank(H, t["q"], Q[t["i"]], sim, V, nix)[0]; b = retr.deja_rank(T, t["q"], Q[t["i"]], sim, V, nix)[0]
        bad += retr.pack(a, cat, 2000) != retr.pack(b, cat, 2000)
    print("bulk_load 300 + ingest 100 one at a time; graph vs twin loadout mismatches on the next 100:", bad); assert bad == 0
