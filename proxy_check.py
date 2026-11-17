"""End-to-end check of proxy.py over real stdio: an MCP client -> Deja proxy -> three fixture MCP servers (subprocesses).
  BUCKET=proxy-check-$(date +%s) ./hydra.sh && uv run python proxy_check.py
Checks: the agent sees 2 tools; calls are forwarded; a tool the agent needed but Deja didn't load is learned and loaded
next time; a tool that keeps failing is quarantined; a tool an upstream drops is never served; memory survives a restart.
`python proxy_check.py serve <name>` runs a fixture server."""
import asyncio, json, os, sys, tempfile, threading, time

if sys.argv[1:2] == ["serve"]:                               # fixture upstream servers
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    srv = MCPServer(sys.argv[2])
    if sys.argv[2] == "weather":
        @srv.tool()
        def get_forecast(city: str) -> str:
            """Get the weather forecast for a city for the next days."""
            return f"{city}: sunny, 24C"
        @srv.tool()
        def get_alerts(region: str) -> str:
            """Get severe weather alerts for a region."""
            raise ToolError("upstream API key missing")       # chronically broken
        @srv.tool()
        def get_air_quality(city: str) -> str:
            """Get the current air quality index for a city."""
            return f"{city}: AQI 31"
    if sys.argv[2] == "notes":
        @srv.tool()
        def add_note(text: str) -> str:
            """Add a note to the user's notebook."""
            return "saved"
        @srv.tool()
        def delete_note(note_id: str) -> str:
            """Delete a note from the user's notebook by id."""
            return "deleted"
        @srv.tool()
        def search_notes(query: str) -> str:
            """Search the user's notes by keyword."""
            return "no matches"
        def drop():                                            # the upstream drops search_notes once the flag file appears
            while not os.path.exists(os.environ["DROP_FLAG"]): time.sleep(0.1)
            srv.remove_tool("search_notes")
        threading.Thread(target=drop, daemon=True).start()
    if sys.argv[2] == "calendar":
        @srv.tool()
        def create_event(title: str, date: str) -> str:
            """Create a calendar event on a date."""
            return "created"
        @srv.tool()
        def list_events(date: str) -> str:
            """List calendar events on a date."""
            return "[]"
        @srv.tool()
        def convert_currency(amount: float, src: str, dst: str) -> str:
            """Convert an amount of money between currencies."""
            return str(amount * 1.1)
    srv.run("stdio"); sys.exit()

from mcp import Client, StdioServerParameters
HERE = os.path.dirname(os.path.abspath(__file__)); TMP = tempfile.mkdtemp(); FLAG = os.path.join(TMP, "drop")
CFG = os.path.join(TMP, "upstreams.json"); LOGF = os.path.join(TMP, "tasks.jsonl")
json.dump({"mcpServers": {n: {"command": sys.executable, "args": [__file__, "serve", n], "env": {"DROP_FLAG": FLAG, "PATH": os.environ["PATH"]}}
                          for n in ("weather", "notes", "calendar")}}, open(CFG, "w"))
PROXY = StdioServerParameters(command=sys.executable, args=[os.path.join(HERE, "proxy.py"), CFG], cwd=HERE,
                              env={**os.environ, "DEJA_LOG": LOGF, "DEJA_BUDGET": "110", "DEJA_LISTED": "3", "DEJA_RAMP": "3"})

async def load(c, task, more=None):
    r = await c.call_tool("deja_load_tools", {"task": task, **({"more": more} if more else {})})
    return json.loads(r.content[0].text)

async def call(c, tool, args):
    r = await c.call_tool("deja_call", {"tool": tool, "arguments": args}); return not r.is_error, r.content[0].text

async def main():
    out = {}
    async with Client(PROXY) as c:
        out["agent_sees"] = sorted(t.name for t in (await c.list_tools()).tools)
        assert out["agent_sees"] == ["deja_call", "deja_load_tools"]
        first = await load(c, "Organise a picnic in Paris this weekend")
        out["picnic_loadout_before_learning"] = [t["name"] for t in first["tools"]]
        for city in ("Paris", "Lyon", "Nice"):                 # the agent needs the forecast and a note, every time
            if city != "Paris": await load(c, f"Organise a picnic in {city} this weekend")
            assert (await call(c, "weather-get_forecast", {"city": city}))[0]
            assert (await call(c, "notes-add_note", {"text": f"picnic in {city}: bring water"}))[0]
        after = await load(c, "Organise a picnic in Marseille this weekend")
        out["picnic_loadout_after_3_tasks"] = [t["name"] for t in after["tools"]]
        assert "notes-add_note" in out["picnic_loadout_after_3_tasks"], out
        for k in range(5):                                     # a tool that keeps failing
            await load(c, f"Are there severe weather alerts in region {k}?")
            ok, msg = await call(c, "weather-get_alerts", {"region": str(k)}); assert not ok
            await call(c, "weather-get_forecast", {"city": str(k)})
        alerts = await load(c, "Any severe weather alerts in Provence?")
        out["quarantined"] = alerts["memory"]["quarantined"]; out["alerts_loadout"] = [t["name"] for t in alerts["tools"]]
        assert "weather-get_alerts" in out["quarantined"] and "weather-get_alerts" not in out["alerts_loadout"]
        s1 = await load(c, "Search my notes for the picnic list")
        out["search_before_drop"] = [t["name"] for t in s1["tools"] + s1["also_available"]]       # loaded or listed
        open(FLAG, "w").close(); await asyncio.sleep(0.5)       # the notes server drops search_notes
        s2 = await load(c, "Search my notes for the picnic list")   # (new task: the catalog is re-read)
        out["search_after_drop"] = [t["name"] for t in s2["tools"] + s2["also_available"]]
        assert "notes-search_notes" in out["search_before_drop"] and "notes-search_notes" not in out["search_after_drop"], out
        ok, msg = await call(c, "notes-search_notes", {"query": "x"}); assert not ok and "unknown tool" in msg
        await load(c, "Weather alerts for Corsica please")      # a task where nothing works: logged as unserved, not learned
        assert not (await call(c, "weather-get_alerts", {"region": "Corsica"}))[0]
        s3 = await load(c, "Search my notes for the picnic list")
        out["memory_before_restart"] = s3["memory"]["past_tasks"]; assert out["memory_before_restart"] == s2["memory"]["past_tasks"]
    async with Client(PROXY) as c:                             # restart: memory and quarantine come back from the log + graph
        r = await load(c, "Organise a picnic in Bordeaux this weekend")
        out["memory_after_restart"] = r["memory"]["past_tasks"]; out["picnic_loadout_after_restart"] = [t["name"] for t in r["tools"]]
        assert out["memory_after_restart"] == out["memory_before_restart"] and "notes-add_note" in out["picnic_loadout_after_restart"]
    log = [json.loads(x) for x in open(LOGF)]
    out["log"] = {"tasks": len(log), "unserved": sum(bool(t.get("unserved")) for t in log)}; assert out["log"]["unserved"] == 1
    json.dump(out, open(os.path.join(HERE, "data/proxy_check.json"), "w"), indent=1); print(json.dumps(out, indent=1))

asyncio.run(main())
