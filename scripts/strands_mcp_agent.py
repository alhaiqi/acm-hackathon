"""GoPilot-style integration: a separate Strands agent that knows GoTrain ONLY through MCP.

It imports no GoTrain code. Strands' MCPClient starts `python -m gotrain.mcp_server` over stdio, lists its
tools, and hands a read-only subset to a Strands Agent on Bedrock, which then answers a question
about the real state (tasks, test-split status, model versions). This is how GoPilot would drive GoTrain.

    python scripts/strands_mcp_agent.py ["question"]
"""
import os
import sys
from pathlib import Path

from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient

ROOT = Path(__file__).resolve().parent.parent
READ_ONLY = {"list_tasks", "leaderboard", "model_versions", "read_journal", "gpu_status", "time_remaining"}
QUESTION = (sys.argv[1] if len(sys.argv) > 1 else
            "Which GEO-Bench tasks are downloaded, which already used their test split, and what model "
            "versions exist for m-eurosat (stage, val and test score)? Answer in a short table.")

env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
gotrain = MCPClient(lambda: stdio_client(StdioServerParameters(command=sys.executable,
                                                               args=["-m", "gotrain.mcp_server"], env=env)))
with gotrain:
    every = gotrain.list_tools_sync()
    allowed = [t for t in every if t.tool_name in READ_ONLY]
    print(f"MCP server exposes {len(every)} tools; this agent gets the read-only {len(allowed)}: "
          f"{sorted(t.tool_name for t in allowed)}\n")
    model = BedrockModel(model_id=os.environ.get("GOTRAIN_MODEL_ID", "eu.anthropic.claude-sonnet-5-5"),
                         region_name=os.environ.get("GOTRAIN_REGION", "eu-central-1"), max_tokens=4000)
    agent = Agent(model=model, tools=allowed,
                  system_prompt="You are GoPilot. You reach the GoTrain training service only through its MCP tools.")
    agent(QUESTION)
print("\nSTRANDS MCP OK")
