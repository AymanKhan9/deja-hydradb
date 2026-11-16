"""Parse Toucan-1.5M traces into calls, value flows (tool output -> later tool arg) and entity mentions."""
import ast, json, re, unicodedata, glob
import pyarrow.parquet as pq

def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s)).strip()

# A tool call failed if its response opens with an error. Looks only at the first 300 chars: a successful result can
# mention "error" deep inside its payload. Precision was hand-checked on a sample (see README).
ERR = re.compile(r"(^\s*\{?\s*\"?error\"?\s*[:=]|^\W*error\b|exception|traceback|failed to|not found|invalid (argument|param|input)|"
                 r"unauthori[sz]ed|forbidden|rate limit|timed? ?out|\b(4\d\d|5\d\d)\b.*(error|status)|is not a function|unknown tool|"
                 r"no such tool|missing required|^.{0,40}\berror\b|\"status\"\s*:\s*\"(not_found|error|failed)\")", re.I)
def call_ok(out): return bool(out and out.strip()) and not ERR.search(out[:300])

def idlike(v):
    # ponytail: heuristic ID filter; plain words/phrases are search terms, not resolved entities
    v = v.strip()
    if len(v) < 5 or v.lower() in ("true", "false", "none", "null") or len(set(v)) < 4: return False
    if v.replace(".", "").replace("-", "").isdigit() and len(v) < 6: return False
    return bool(re.search(r"[\d_/:]", v)) or not re.fullmatch(r"[A-Za-z ]+", v)

def leaves(x, k=""):
    if isinstance(x, dict):
        for kk, v in x.items(): yield from leaves(v, kk)
    elif isinstance(x, list):
        for v in x: yield from leaves(v, k)
    elif x is not None: yield k, str(x)

def _sibling_strings(obj, v, out):
    """Short string values sitting in the same JSON object as v (e.g. the 'name' next to an 'id')."""
    if isinstance(obj, dict):
        if any(str(x).strip() == v for x in obj.values() if not isinstance(x, (dict, list))):
            out.extend(str(x) for x in obj.values() if isinstance(x, str) and 3 <= len(x) <= 60 and x.strip() != v)
        for x in obj.values(): _sibling_strings(x, v, out)
    elif isinstance(obj, list):
        for x in obj: _sibling_strings(x, v, out)

def mentions_for(call, v):
    m = [s for _, s in leaves(call["args"]) if 3 <= len(s) <= 60 and s.strip() != v]
    try: _sibling_strings(json.loads(call["out"]), v, m)
    except Exception: pass
    return {n for n in (norm(x) for x in m) if len(n) >= 3 and not n.isdigit()}

def parse(row):
    msgs = json.loads(row["messages"]); calls = []; user = row.get("question") or ""; prose = ""
    names = {t["function"]["name"] for t in json.loads(row.get("available_tools") or "[]") if "function" in t}
    for m in msgs:
        if m["role"] == "user": user += " " + str(m.get("content"))
        fc = None
        if m["role"] == "assistant" and m.get("function_call"): fc = m["function_call"]
        elif m["role"] == "tool_call":
            try: fc = ast.literal_eval(m["content"])
            except Exception: fc = None
        elif m["role"] == "assistant": prose += " " + str(m.get("content") or "")
        if fc:
            a = fc.get("arguments")
            try: a = json.loads(a) if isinstance(a, str) else a
            except Exception: a = {}
            name = str(fc.get("name", "")).strip().strip('"')
            if names and name not in names: continue          # hallucinated/garbled tool name
            calls.append({"tool": name, "args": a if isinstance(a, dict) else {}, "out": None, "prose": prose})
        elif m["role"] in ("function", "tool_response") and calls:
            for c in calls:
                if c["out"] is None: c["out"] = str(m.get("content")); break
    flows = []  # (src_call, dst_call, param, value)
    for j, c in enumerate(calls):
        for k, v in leaves(c["args"]):
            v = v.strip()
            if not idlike(v) or v in user: continue                      # user-given, not a lookup result
            for i in range(j):
                if calls[i]["out"] and calls[i]["tool"] != c["tool"] and v in calls[i]["out"]:
                    if v not in calls[i]["prose"]: flows.append((i, j, k, v))  # model knew it before the lookup -> not a flow
                    break
    meta = json.loads(row.get("metadata") or "{}")
    tool_server = {}
    for s in meta.get("mcp_servers", []):
        for rt in (s.get("remote_server_response") or {}).get("tools", []):
            for n in names:
                if n == rt.get("name") or n.endswith("-" + str(rt.get("name"))): tool_server.setdefault(n, s["server_name"])
    return {"uuid": row["uuid"], "prompt_id": meta.get("prompt_id"), "question": row.get("question") or "",
            "servers": tuple(sorted(s["server_name"] for s in meta.get("mcp_servers", []))), "tool_server": tool_server,
            "tools": json.loads(row.get("available_tools") or "[]"), "calls": calls, "flows": flows}

def load(pattern="data/raw/kimi_*.parquet", slim=False):
    """Stream shards in small batches (full shards as Python dicts OOM a 16 GB box). slim drops message text/outputs."""
    out = []
    for f in sorted(glob.glob(pattern)):
        for b in pq.ParquetFile(f).iter_batches(batch_size=500, columns=["uuid", "question", "messages", "available_tools", "metadata"]):
            for row in b.to_pylist():
                p = parse(row)
                if not p["calls"]: continue
                if slim:
                    p["tools"] = [{"name": t["function"]["name"], "desc": t["function"].get("description", ""), "tokens": len(json.dumps(t)) // 4}
                                  for t in p["tools"] if "function" in t]
                    p["calls"] = [{"tool": c["tool"], "ok": call_ok(c["out"])} for c in p["calls"]]
                out.append(p)
    return out
