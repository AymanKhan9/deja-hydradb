"""Deja's learned graph in HydraDB.

  (:Term {key, df}) -[:LED]-> (:Link {n}) -[:TO]-> (:Tool {key, uses})      task word -> tool that solved it
  (:Tool) -[:CO]-> (:CoLink {n}) -[:TO]-> (:Tool)                          tools used together / data flowed between
  (:Meta {n_tasks})

Counts sit on reified Link nodes because HydraDB's batch writes can set node properties but not edge properties.
Readers (agents) only query HydraDB. The single learner keeps an in-memory mirror of the counts so each write is one
absolute-value batch instead of read-modify-write; the mirror is also the oracle the graph reads are checked against.
"""
import hashlib, math, collections
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
    """Same interface as the in-memory Learned, but term/co-use scores are read from HydraDB by graph traversal."""
    def __init__(s, db=None):
        super().__init__(); s.db = db or Hydra()
    def ingest(s, task):
        super().ingest(task)
        ts, gold = terms(task["q"]), task["gold"]
        db = s.db
        db.nodes("Term", [{"id": hid("term", w), "key": w, "df": s.df[w]} for w in ts], ["key", "df"])
        db.nodes("Tool", [{"id": hid("tool", g), "key": g, "uses": s.uses[g]} for g in gold], ["key", "uses"])
        db.nodes("Link", [{"id": hid("link", w + "\x1f" + g), "n": s.led[w][g]} for w in ts for g in gold], ["n"])
        pairs = {(a, b) for a in gold for b in gold if a != b} | {tuple(f) for f in task["flows"]}
        db.nodes("CoLink", [{"id": hid("co", a + "\x1f" + b), "n": s.co[a][b]} for a, b in pairs], ["n"])
        db.nodes("Meta", [{"id": META, "n_tasks": s.N}], ["n_tasks"])
        db.edges("LED", [(hid("term", w), hid("link", w + "\x1f" + g)) for w in ts for g in gold])
        db.edges("TO", [(hid("link", w + "\x1f" + g), hid("tool", g)) for w in ts for g in gold]
                       + [(hid("co", a + "\x1f" + b), hid("tool", b)) for a, b in pairs])
        db.edges("CO", [(hid("tool", a), hid("co", a + "\x1f" + b)) for a, b in pairs])
    def bulk_load(s, tasks):
        """Boot path: learn from many tasks in memory, then write the whole graph in 1000-row batches (labels first)."""
        for t in tasks: Learned.ingest(s, t)
        db = s.db; lk = [(w, g) for w in s.led for g, _ in s.links(w)]; co = [(a, b) for a, c in s.co.items() for b in c]
        tools = set(s.uses) | {b for _, b in co} | {a for a, _ in co}
        db.nodes("Term", [{"id": hid("term", w), "key": w, "df": s.df[w]} for w in {w for w, _ in lk}], ["key", "df"])
        db.nodes("Tool", [{"id": hid("tool", g), "key": g, "uses": s.uses[g]} for g in tools], ["key", "uses"])
        db.nodes("Link", [{"id": hid("link", w + "\x1f" + g), "n": s.led[w][g]} for w, g in lk], ["n"])
        db.nodes("CoLink", [{"id": hid("co", a + "\x1f" + b), "n": s.co[a][b]} for a, b in co], ["n"])
        db.nodes("Meta", [{"id": META, "n_tasks": s.N}], ["n_tasks"])
        db.edges("LED", [(hid("term", w), hid("link", w + "\x1f" + g)) for w, g in lk])
        db.edges("TO", [(hid("link", w + "\x1f" + g), hid("tool", g)) for w, g in lk] + [(hid("co", a + "\x1f" + b), hid("tool", b)) for a, b in co])
        db.edges("CO", [(hid("tool", a), hid("co", a + "\x1f" + b)) for a, b in co])
        return len(lk), len(co)
    def graph_term_scores(s, q):
        ts = sorted(terms(q))
        if not ts: return {}
        where, p = s.db.any_of("t", ts)
        rows = s.db.q(f"MATCH (t:Term)-[:LED]->(l:Link)-[:TO]->(x:Tool) WHERE {where} RETURN t.key AS term, t.df AS df, x.key AS tool, l.n AS n", **p)
        N = (s.db.q("MATCH (m:Meta {id: $id}) RETURN m.n_tasks AS n", id=META) or [{"n": 0}])[0]["n"]
        out, idf = collections.Counter(), {}
        s.last_term_rows = rows                      # kept for the UI's "why" view
        for r in rows:
            idf[r["term"]] = math.log((N + 1) / (r["df"] + 0.5))
            out[r["tool"]] += idf[r["term"]] * r["n"] / r["df"]
        z = sum(idf.values())
        return {g: v / z for g, v in out.items()} if z else {}
    def graph_co_scores(s, seeds):
        if not seeds: return {}
        where, p = s.db.any_of("a", seeds)
        rows = s.db.q(f"MATCH (a:Tool)-[:CO]->(l:CoLink)-[:TO]->(b:Tool) WHERE {where} RETURN a.key AS a, a.uses AS uses, b.key AS b, l.n AS n", **p)
        rank = {a: r for r, a in enumerate(seeds)}; out = collections.Counter(); s.last_co_rows = rows
        for r in rows: out[r["b"]] = max(out[r["b"]], r["n"] / r["uses"] / (1 + rank[r["a"]]))
        return out
    # the eval loop calls these two; route them through the graph
    term_scores = graph_term_scores
    co_scores = graph_co_scores

if __name__ == "__main__":
    # consistency check: graph reads must equal the in-memory oracle after ingesting a few hundred tasks
    import retr
    cat, tasks = retr.build()
    H = HydraLearned()
    for t in tasks[:300]: H.ingest(t)
    bad = 0
    for t in tasks[300:400]:
        a, b = H.graph_term_scores(t["q"]), Learned.term_scores(H, t["q"])
        if a.keys() != b.keys() or any(abs(a[k] - b[k]) > 1e-9 for k in a): bad += 1
        seeds = sorted(t["gold"])
        c, d = H.graph_co_scores(seeds), Learned.co_scores(H, seeds)
        if {k for k, v in c.items() if v} != {k for k, v in d.items() if v}: bad += 1
    print("graph vs oracle mismatches:", bad); assert bad == 0
