"""Deja as an MCP proxy: one MCP server in front of your real MCP servers, learning from real agent traffic.

The agent sees two tools, so it works with any MCP client (no tools/list_changed support needed):
  deja_load_tools(task, more=[])  -> the loadout for this task: full schemas for the tools Deja picks within the token
                                     budget, plus one-line listings of the next few; `more` loads listed tools too
  deja_call(tool, arguments)      -> forwards the call to the upstream server that owns the tool
Each deja_load_tools call with a new task ends the previous one. Deja then learns what it saw: the tools whose calls
worked (`isError` false) become (:Task)-[:USED]->(:Tool) in HydraDB, the ones that only failed become FAILED edges (and
feed quarantine), a task where nothing worked is logged as unserved, and a tool the agent had to ask for via `more` is
logged as a loadout miss. The catalog is re-read from every upstream's tools/list at the start of each task and synced
to (:Catalog)-[:LISTS]->(:Tool), so a tool an upstream drops is never loaded again.

  BUCKET=deja-proxy ./hydra.sh                         # HydraDB for the proxy's graph
  uv run python proxy.py upstreams.json                # stdio MCP server; upstreams in the usual mcpServers format
State: the graph in HydraDB, plus data/proxy_tasks.jsonl (the task log the in-memory mirror is rebuilt from on start).
"""
import contextlib, fcntl, hashlib, json, math, os, re, sys, time
import anyio
import numpy as np
from mcp import Client, StdioServerParameters
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DEJA_UPSTREAMS", "upstreams.json"))
os.chdir(ROOT)                                                  # data/, .hydradb/ are relative to the repo
import retr
from hydra_store import HydraLearned, hid, write_catalog, remove_tool, META

LOG = os.environ.get("DEJA_LOG", "data/proxy_tasks.jsonl")
PENDING = LOG + ".pending"                                         # the open task, saved after every call: clients may kill the proxy
VECS = "data/proxy_tool_vecs.npz"                                 # tool embeddings by text hash: restarts skip re-embedding
BUDGET, LISTED = int(os.environ.get("DEJA_BUDGET", 2000)), int(os.environ.get("DEJA_LISTED", 10))
retr.Learned.OUTCOME = True
log = lambda *a: print(*a, file=sys.stderr, flush=True)       # stdout is the MCP channel

def api_name(server, tool):
    """Namespaced `server-tool`, valid for every client (^[a-zA-Z0-9_-]{1,64}$)."""
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{server}-{tool}")
    return s if len(s) <= 64 else s[:55] + "_" + hashlib.md5(s.encode()).hexdigest()[:8]

class Deja:
    def __init__(s):
        from fastembed import TextEmbedding
        s.emb_model = retr.EMB = TextEmbedding("BAAI/bge-small-en-v1.5", threads=2)   # shared with retr's step embeddings
        s.ups, s.cat, s.names, s.TE, s.cur = {}, {}, [], np.zeros((0, 384), np.float32), None
        s.vec_cache = dict(np.load(VECS)) if os.path.exists(VECS) else {}; s.up = anyio.Event()
        s.L = HydraLearned(); s.qrows = []; s.lock = open(LOG + ".lock", "w")
        try: fcntl.flock(s.lock, fcntl.LOCK_EX | fcntl.LOCK_NB); s.learns = True
        except BlockingIOError: s.learns = False                          # another session's proxy is learning: serve, don't write
        s.tasks = [json.loads(x) for x in open(LOG)] if os.path.exists(LOG) else []
        mem = [t for t in s.tasks if not t.get("unserved")]
        s.qrows = list(s.cached([t["q"] for t in mem])); retr.Learned.Q = np.array(s.qrows).reshape(-1, 384)
        for t in mem: t.update(gold=set(t["gold"]), gold_ok=set(t["gold_ok"]), failed=set(t["failed"]))
        HydraLearned.tool_names = {hid("tool", g): g for t in mem for g in t["gold"]}   # memory can name tools that left the catalog
        n = (s.L.db.q("MATCH (m:Meta {id: $id}) RETURN m.n_tasks AS n", id=META) or [{"n": 0}])[0]["n"]
        if n == len(mem) or (not s.learns and n > len(mem)):                                                       # graph already holds them: rebuild the mirror only
            for t in s.tasks:
                if t.get("unserved"): [s.L.recent[g].append(0) for g in t["failed"]]
                else: retr.Learned.ingest(s.L, t)
        elif n == 0 and s.learns: s.L.bulk_load(mem)            # fresh graph: write the log into it (ponytail: unserved failures not replayed here)
        else: raise SystemExit(f"graph has {n} tasks, log has {len(mem)}: start HydraDB on a fresh BUCKET")
        if s.learns and os.path.exists(PENDING):                       # killed mid-task last time: learn that task now
            c = json.load(open(PENDING)); HydraLearned.tool_names.update({hid("tool", g): g for g in c["calls"]})
            s.cur = dict(c, qvec=s.embed([c["q"]])[0], more=set(c["more"])); s.finish()
        log(f"deja proxy: {len(s.qrows)} past tasks in memory ({len(s.tasks) - len(mem)} unserved logged)"
            + ("" if s.learns else "; another Deja proxy is learning, this one only serves"))

    def embed(s, texts):
        E = np.array(list(s.emb_model.embed(texts)), dtype=np.float32)
        return E / np.linalg.norm(E, axis=1, keepdims=True)

    def cached(s, texts):
        """Embeddings by text hash, kept in VECS across restarts."""
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]; new = sorted({k: t for k, t in zip(keys, texts) if k not in s.vec_cache}.items())
        if new: s.vec_cache.update(zip((k for k, _ in new), s.embed([t for _, t in new]))); np.savez(VECS, **s.vec_cache)
        return np.array([s.vec_cache[k] for k in keys], np.float32).reshape(-1, 384)

    async def refresh(s):
        """Re-read every upstream's tools/list and sync the graph's catalog to it."""
        cat = {}
        for srv, c in s.ups.items():
            cursor = None
            while True:
                r = await c.list_tools(cursor=cursor, cache_mode="bypass")
                for t in r.tools:
                    n = api_name(srv, t.name); d = t.description or ""
                    cat[n] = {"name": n, "server": srv, "tool": t.name, "description": d, "schema": t.input_schema,
                              "text": f"{n}: {d}", "tokens": max(1, len(json.dumps(t.input_schema)) // 4 + len(d) // 4 + 8)}
                cursor = r.next_cursor
                if not cursor: break
        rows = s.L.db.q("MATCH (c:Catalog {id: $id})-[:LISTS]->(t:Tool) RETURN t.key AS k", id=hid("catalog", "deja"))
        before = {r["k"] for r in rows}
        if set(cat) - before: write_catalog(s.L.db, sorted(set(cat) - before))
        for n in sorted(before - set(cat)): remove_tool(s.L.db, n)
        if set(cat) != set(s.cat) or any(cat[n]["text"] != s.cat[n]["text"] for n in cat):
            s.cat = cat; s.names = sorted(cat); s.nix = {n: k for k, n in enumerate(s.names)}
            s.TE = await anyio.to_thread.run_sync(s.cached, [cat[n]["text"] for n in s.names])   # off the loop: long descriptions are slow on CPU
            retr._BM = retr.BM25(cat, s.names); retr._LT = {n: math.log(cat[n]["tokens"]) for n in cat}
            HydraLearned.tool_names.update({hid("tool", n): n for n in cat})
            log(f"catalog: {len(cat)} tools from {len(s.ups)} servers")

    def finish(s):
        """End the current task and learn from it."""
        c, s.cur = s.cur, None
        if not s.learns or not c or not c["calls"] or c["qvec"] is None: return        # calls made without a task teach nothing
        ok = {n for n, v in c["calls"].items() if v}
        t = {"i": len(s.qrows), "q": c["q"], "gold": set(c["calls"]), "gold_ok": ok, "failed": set(c["calls"]) - ok, "flows": [],
             "loaded": c["loaded"], "asked_for_more": sorted(c["more"])}
        if ok:
            s.qrows.append(c["qvec"]); retr.Learned.Q = np.array(s.qrows); s.L.ingest(t)
        else: t["unserved"] = True; s.L.record_failures(t)            # nothing worked: logged, teaches no tool, failures still count
        s.tasks.append(t)
        with open(LOG, "a") as f: f.write(json.dumps({k: sorted(v) if isinstance(v, set) else v for k, v in t.items()}) + "\n")
        if os.path.exists(PENDING): os.remove(PENDING)

    def full(s, n): return {"name": n, "description": s.cat[n]["description"], "input_schema": s.cat[n]["schema"]}

    async def load(s, task, more):
        await s.up.wait()                                               # upstreams still starting: wait, don't fail
        if more and s.cur and task.strip() in ("", s.cur["q"]):          # load listed tools into the current task
            got = [n for n in more if n in s.cat]; s.cur["more"] |= set(got) - set(s.cur["loaded"])
            return {"tools": [s.full(n) for n in got], "unknown": [n for n in more if n not in s.cat]}
        s.finish(); await s.refresh()
        if not s.names: return {"tools": [], "note": "no upstream tools"}
        qv = s.embed([task])[0]; sim = s.TE @ qv; held = s.L.quarantined(); active = set(s.names) - held
        V = [s.names[k] for k in np.argsort(-sim, kind="stable") if s.names[k] in active][:400]
        ranked = retr.deja_rank(s.L, task, qv, sim, V, s.nix, active=active)[0]
        picked = retr.pack(ranked, s.cat, BUDGET); listed = [n for n in ranked if n not in picked][:LISTED]
        s.cur = {"q": task, "qvec": qv, "calls": {}, "loaded": picked, "more": set()}
        return {"tools": [s.full(n) for n in picked],
                "also_available": [{"name": n, "description": s.cat[n]["description"][:100]} for n in listed],
                "memory": {"past_tasks": len(s.qrows), "quarantined": sorted(held)},
                "how": "Call a tool with deja_call(tool, arguments). To use a listed tool, call deja_load_tools with the same task and more=[names]."}

    async def call(s, tool, arguments):
        await s.up.wait()
        if tool not in s.cat: raise ToolError(f"unknown tool {tool!r}: call deja_load_tools first")
        if s.cur is None: s.cur = {"q": "", "qvec": None, "calls": {}, "loaded": [], "more": set()}
        c = s.cat[tool]
        try: r = await s.ups[c["server"]].call_tool(c["tool"], arguments or {})
        except Exception as e:                                         # protocol-level failure counts as a failed call
            s.cur["calls"][tool] = s.cur["calls"].get(tool, False); raise ToolError(f"{type(e).__name__}: {e}")
        s.cur["calls"][tool] = s.cur["calls"].get(tool, False) or not r.is_error
        if s.learns and s.cur["q"]:
            json.dump({k: sorted(v) if isinstance(v, set) else v for k, v in s.cur.items() if k != "qvec"}, open(PENDING, "w"))
        text = "\n".join(getattr(b, "text", None) or f"[{b.type} content]" for b in r.content) or json.dumps(r.structured_content)
        if r.is_error: raise ToolError(text)
        return text

D = None

async def hold(name, spec, ready, stop):
    """Connect one upstream and keep it open until shutdown. A server that fails to start is skipped, not fatal."""
    target = spec["url"] if "url" in spec else StdioServerParameters(command=spec["command"], args=spec.get("args", []), env=spec.get("env"))
    try:
        async with Client(target) as c:
            D.ups[name] = c; ready.set(); await stop.wait()
    except Exception as e: log(f"upstream {name} failed: {type(e).__name__}: {e}")
    finally: D.ups.pop(name, None); ready.set()

@contextlib.asynccontextmanager
async def lifespan(server):
    global D
    D = Deja(); t0 = time.time(); stop = anyio.Event(); specs = json.load(open(CONFIG))["mcpServers"]
    ready = {n: anyio.Event() for n in specs}
    async def start():                         # in the background: the client's initialize is answered at once (clients time out)
        for e in ready.values(): await e.wait()
        await D.refresh(); D.up.set(); log(f"{len(D.ups)}/{len(specs)} upstreams up in {time.time() - t0:.1f}s")
    async with anyio.create_task_group() as tg:                        # upstreams start concurrently (npx/uvx take ~7 s each)
        for n, spec in specs.items(): tg.start_soon(hold, n, spec, ready[n], stop)
        tg.start_soon(start)
        try: yield None
        finally: D.finish(); stop.set()                                # the last task is learned on shutdown

SERVERS = ", ".join(json.load(open(CONFIG))["mcpServers"])        # named up front: clients that defer MCP tools only show this text
mcp = MCPServer("deja", lifespan=lifespan, instructions=(
    f"Deja holds the tools of these MCP servers: {SERVERS}. When a task could use one of them, call deja_load_tools with the "
    "user's task first: it returns the tools that fit. Then call them with deja_call."))

@mcp.tool(description=f"""Get the tools for a task from these MCP servers: {SERVERS}. Call this first with the user's task, in their
words. Returns full definitions of the tools to use and a short list of other relevant tools; pass the same task with
more=[names] to load listed ones.""")
async def deja_load_tools(task: str, more: list[str] | None = None) -> str:
    return json.dumps(await D.load(task, more or []))   # ponytail: HydraDB and embedding calls block the loop; one client at a time

@mcp.tool()
async def deja_call(tool: str, arguments: dict | None = None) -> str:
    """Call a tool returned by deja_load_tools, with arguments matching its input_schema."""
    return await D.call(tool, arguments)

if __name__ == "__main__":
    mcp.run("stdio")
