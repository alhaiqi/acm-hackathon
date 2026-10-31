"""GoTrain as an MCP server, so GoPilot (or any MCP client) can drive the same tools.

    python -m gotrain.mcp_server            # stdio transport
"""
try:                                        # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:                         # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from .tools import ALL_TOOLS

mcp = _Server("gotrain")
for fn in ALL_TOOLS:
    mcp.tool()(fn)

if __name__ == "__main__":
    mcp.run()
