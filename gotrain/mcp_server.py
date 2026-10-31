"""GoTrain as an MCP server, so GoPilot (or any MCP client) can drive the same tools.

    python -m gotrain.mcp_server            # stdio transport

Exposes the training agent's tools (ALL_TOOLS) plus the extras that only GoPilot gets (EXTRA_TOOLS:
data advisor, reproduce receipts, GeoTIFF inference). The training agent never sees the extras.
"""
try:                                        # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:                         # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from .tools import ALL_TOOLS


def extra_tools() -> list:
    from . import advisor, inference, receipt
    return list(advisor.MCP_TOOLS) + list(receipt.MCP_TOOLS) + list(inference.MCP_TOOLS)


EXTRA_TOOLS = extra_tools()

mcp = _Server("gotrain")
for fn in ALL_TOOLS + EXTRA_TOOLS:
    mcp.tool()(fn)

if __name__ == "__main__":
    mcp.run()
