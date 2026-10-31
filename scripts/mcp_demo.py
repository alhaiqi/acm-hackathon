"""A7 demo: GoTrain as an MCP server, driven by a plain MCP client (what GoPilot would be).

    python scripts/mcp_demo.py                 # starts `python -m gotrain.mcp_server` over stdio

Lists the tools the server exposes, then calls a few read-only ones.
"""
import asyncio
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent


async def main():
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    server = StdioServerParameters(command=sys.executable, args=["-m", "gotrain.mcp_server"], env=env)
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            tools = (await s.list_tools()).tools
            print(f"{len(tools)} tools over MCP:", ", ".join(t.name for t in tools))
            for name, args in [("gpu_status", {}), ("list_tasks", {})]:
                res = await s.call_tool(name, args)
                text = " ".join(getattr(c, "text", "") for c in res.content)
                print(f"\n--- {name} ---\n{text[:1500]}")
    print("\nMCP OK")


if __name__ == "__main__":
    asyncio.run(main())
