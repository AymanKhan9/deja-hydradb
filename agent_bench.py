"""Agent benchmark in Claude Code: held-out tasks, three ways of giving Claude the tools, scored on what it calls.

Arms (same held-out tasks, same model, same system prompt):
  deja    Deja's loadout (served model: learns from calls that worked, quarantine), tools given directly
  memory  memory-only loadout (past-task votes, no graph hops, no quarantine), tools given directly
  search  all 2,236 tools behind one MCP server; Claude Code defers them and Claude finds them with its own tool search
Tools aren't live: a stub MCP server plays back the outputs recorded in the Toucan trace (the next recorded output of
that tool, else "no result"). Scored per task: is the first tool call one that worked in the recorded run; were all
of those called by the end; tool calls, turns, tokens. Runs `claude -p` on the logged-in account (no API credit).

  uv run python agent_bench.py prep             # held-out tasks + recorded outputs + loadouts -> data/bench_tasks.json
  uv run python agent_bench.py run 20           # first 20 tasks x 3 arms -> data/bench_runs.jsonl (resumable)
  uv run python agent_bench.py report           # -> data/bench.json
"""
import concurrent.futures, glob, hashlib, json, os, random, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
TASKS, RUNS = "data/bench_tasks.json", "data/bench_runs.jsonl"
SYSTEM = ("You are an agent completing the user's task with the tools provided. Do the task now with the tools; don't ask "
          "the user questions. When you are done, give a brief answer.")
ARMS = ("deja", "memory", "search")

def api_name(n):
    """Claude Code names MCP tools mcp__t__<name>; the API allows 64 chars, so <name> gets at most 56."""
    import re
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", n)
    return s if s == n and len(s) <= 56 else s[:47] + "_" + hashlib.md5(n.encode()).hexdigest()[:8]

def prep():
    import numpy as np, pyarrow.parquet as pq
    import retr, toucan
    retr.Learned.OUTCOME = True
    cat, tasks = retr.build(); names = list(cat); nix = {n: k for k, n in enumerate(names)}
    TE, Q = retr.embed(None, "data/tool_emb.npy"), retr.embed(None, "data/q_emb.npy"); retr.Learned.Q = Q
    order = list(range(len(tasks))); random.Random(1).shuffle(order); train, held = order[:-200], order[-200:]   # as server.py
    T = json.load(open("data/traces_slim_ok.json")); seen, D = set(), []
    for t in T:
        if t["prompt_id"] not in seen: seen.add(t["prompt_id"]); D.append(t)
    D = [t for t in D if any(c["tool"] in cat for c in t["calls"])]
    assert len(D) == len(tasks) and all(d["question"] == t["q"] for d, t in zip(D, tasks))
    want = {D[i]["uuid"]: i for i in held if tasks[i]["gold_ok"]}; rec = {}
    for f in sorted(glob.glob("data/raw/kimi_*.parquet")):
        for b in pq.ParquetFile(f).iter_batches(batch_size=500, columns=["uuid", "question", "messages", "available_tools", "metadata"]):
            for row in b.to_pylist():
                if row["uuid"] in want and want[row["uuid"]] not in rec:
                    rec[want[row["uuid"]]] = [{"tool": c["tool"], "out": (c["out"] or "")[:4000]} for c in toucan.parse(row)["calls"] if c["tool"] in cat]
    L = retr.Learned()
    for i in train: L.ingest(tasks[i])
    out = []
    for i in held:
        if i not in rec: continue
        t = tasks[i]; sim = TE @ Q[i]; held_q = L.quarantined(); active = set(names) - held_q
        V = [names[k] for k in np.argsort(-sim, kind="stable") if names[k] in active][:400]
        Vall = [names[k] for k in np.argsort(-sim, kind="stable")[:400]]
        out.append({"i": i, "q": t["q"], "gold_ok": sorted(t["gold_ok"]), "gold": sorted(t["gold"]), "recorded": rec[i],
                    "multi": len(t["gold_servers"]) > 1,
                    "deja": retr.pack(retr.deja_rank(L, t["q"], Q[i], sim, V, nix, active=active)[0], cat, 2000),
                    "memory": retr.pack(retr.memory_rank(L, t["q"], Q[i], sim, Vall, nix), cat, 2000)})
    json.dump(out, open(TASKS, "w")); print(f"{len(out)} held-out tasks with recorded outputs -> {TASKS}")

def serve(k, arm, log):
    """Stub MCP server for task k: lists the arm's tools, plays back recorded outputs, logs every call."""
    import anyio, mcp_types as mt
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server
    task = json.load(open(TASKS))[k]; S = json.load(open("data/tool_schemas.json"))
    names = sorted(S) if arm == "search" else task[arm]; back = {api_name(n): n for n in names}
    left = {}
    for c in task["recorded"]: left.setdefault(c["tool"], []).append(c["out"])
    def schema(p):
        p = dict(p) if isinstance(p, dict) else {}; p.pop("$schema", None); p["type"] = "object"
        if not isinstance(p.get("properties"), dict): p["properties"] = {}
        return p
    tools = [mt.Tool(name=api_name(n), description=S[n]["description"] or n, inputSchema=schema(S[n]["parameters"])) for n in names]
    async def on_list(ctx, params): return mt.ListToolsResult(tools=tools)
    async def on_call(ctx, p):
        n = back.get(p.name, p.name); outs = left.get(n) or []
        text = outs.pop(0) if outs else "No result."
        with open(log, "a") as f: f.write(json.dumps({"tool": n, "args": p.arguments, "recorded": text != "No result."}) + "\n")
        return mt.CallToolResult(content=[mt.TextContent(type="text", text=text)])
    srv = Server("t", on_list_tools=on_list, on_call_tool=on_call)
    async def main():
        async with stdio_server() as (r, w): await srv.run(r, w, srv.create_initialization_options())
    anyio.run(main)

def one(k, arm, model=None):
    task = json.load(open(TASKS))[k]
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "calls.jsonl"); cfg = os.path.join(d, "mcp.json")
        json.dump({"mcpServers": {"t": {"command": sys.executable, "args": [os.path.join(HERE, "agent_bench.py"), "serve", str(k), arm, log]}}}, open(cfg, "w"))
        cmd = ["claude", "-p", task["q"], "--mcp-config", cfg, "--strict-mcp-config", "--setting-sources", "project",
               "--disable-slash-commands", "--no-session-persistence", "--system-prompt", SYSTEM, "--tools", "ToolSearch" if arm == "search" else "",
               "--allowedTools", "mcp__t", "--max-turns", "15", "--output-format", "stream-json", "--verbose"] + (["--model", model] if model else [])
        env = {**os.environ, "ENABLE_TOOL_SEARCH": "true" if arm == "search" else "false"}
        t0 = time.time(); p = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True, timeout=900)
        ev = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
        calls = [json.loads(l) for l in open(log)] if os.path.exists(log) else []
    init = next((e for e in ev if e.get("type") == "system" and e.get("subtype") == "init"), {})
    res = next((e for e in ev if e.get("type") == "result"), {})
    uses = [b["name"] for e in ev if e.get("type") == "assistant" for b in e["message"]["content"] if b.get("type") == "tool_use"]
    called = [c["tool"] for c in calls]; ok = set(task["gold_ok"])
    return {"k": k, "i": task["i"], "arm": arm, "model": init.get("model"), "tools_listed": sum(t.startswith("mcp__t__") for t in init.get("tools", [])),
            "first_ok": bool(called) and called[0] in ok, "all_ok_called": ok <= set(called), "n_calls": len(called),
            "n_toolsearch": sum(u == "ToolSearch" for u in uses), "turns": res.get("num_turns"), "usage": res.get("usage"),
            "cost_usd_notional": res.get("total_cost_usd"), "subtype": res.get("subtype"), "secs": round(time.time() - t0, 1),
            "called": called, "multi": task["multi"],
            "error": (None if ev else p.stderr[-500:]) or (str(res.get("result"))[:300] if res.get("is_error") and res.get("subtype") == "success" else None)}

def run(n, arms=ARMS, par=3, model=None):
    done = {(r["k"], r["arm"]) for r in map(json.loads, open(RUNS))} if os.path.exists(RUNS) else set()
    jobs = [(k, a) for k in range(n) for a in arms if (k, a) not in done]
    with concurrent.futures.ThreadPoolExecutor(par) as ex:
        for r in ex.map(lambda j: one(*j, model=model), jobs):
            if r["error"] or r["subtype"] is None: print("FAILED", r["k"], r["arm"], r["error"], flush=True); continue   # e.g. usage limit: rerun resumes
            with open(RUNS, "a") as f: f.write(json.dumps(r) + "\n")
            print(r["k"], r["arm"], r["model"], "first_ok", r["first_ok"], "all_ok", r["all_ok_called"], "calls", r["n_calls"],
                  "turns", r["turns"], "$", r["cost_usd_notional"], f"{r['secs']}s", flush=True)

def report():
    """Main score counts a call to a near-twin (tool-tool cosine >= 0.80, as in subst_check.py) as covering a needed tool:
    Toucan has the same tool on several servers, and an exact-name match marks a different server's copy as a miss."""
    import numpy as np
    import retr
    names = list(retr.build()[0]); TE = np.load("data/tool_emb.npy"); nix = {n: k for k, n in enumerate(names)}; T = json.load(open(TASKS))
    cover = lambda g, called: any(c == g or (c in nix and float(TE[nix[g]] @ TE[nix[c]]) >= 0.80) for c in called)
    R = [json.loads(l) for l in open(RUNS)]
    for r in R:
        ok = T[r["k"]]["gold_ok"]
        r["all_ok_twin"] = all(cover(g, r["called"]) for g in ok); r["first_ok_twin"] = bool(r["called"]) and any(cover(g, r["called"][:1]) for g in ok)
    ks = sorted({r["k"] for r in R if all((r["k"], a) in {(x["k"], x["arm"]) for x in R} for a in ARMS)})
    out = {"tasks": len(ks), "model": sorted({r["model"] for r in R if r["model"]}), "arms": {}}
    for a in ARMS:
        S = [r for r in R if r["arm"] == a and r["k"] in ks]
        if not S: continue
        tok = [sum((r["usage"] or {}).get(x, 0) or 0 for x in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")) for r in S]
        out["arms"][a] = {"all_worked_tools_called": sum(r["all_ok_twin"] for r in S) / len(S), "first_call_worked": sum(r["first_ok_twin"] for r in S) / len(S),
                          "all_worked_tools_called_exact_name": sum(r["all_ok_called"] for r in S) / len(S),
                          "tool_calls": sum(r["n_calls"] for r in S) / len(S), "toolsearch_calls": sum(r["n_toolsearch"] for r in S) / len(S),
                          "turns": sum(r["turns"] or 0 for r in S) / len(S), "input_tokens": sum(tok) / len(S),
                          "output_tokens": sum((r["usage"] or {}).get("output_tokens", 0) for r in S) / len(S),
                          "cost_usd_notional": sum(r["cost_usd_notional"] or 0 for r in S) / len(S), "secs": sum(r["secs"] for r in S) / len(S)}
    json.dump(out, open("data/bench.json", "w"), indent=1); print(json.dumps(out, indent=1))

if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "prep": prep()
    elif a[0] == "serve": serve(int(a[1]), a[2], a[3])
    elif a[0] == "one": print(json.dumps(one(int(a[1]), a[2]), indent=1))
    elif a[0] == "run": run(int(a[1]), tuple(a[2].split(",")) if len(a) > 2 else ARMS)
    elif a[0] == "report": report()
