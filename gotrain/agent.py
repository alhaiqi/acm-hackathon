"""The GoTrain agent: Claude on Bedrock + Strands ReAct loop + GoTrain tools.

    python -m gotrain.agent --tasks m-eurosat m-pv4ger-seg --budget-min 90

One fresh agent per task (clean context, independent budget). Every tool call, its arguments,
result and duration is written to logs/<session>/tool_calls.jsonl, and the full conversation to
logs/<session>/<task>_messages.json: this is how every score can be traced back.
"""
from __future__ import annotations

import argparse
import json
import time

from . import config, tools
from .prompts import SYSTEM_PROMPT, task_prompt


class ToolLogger:
    def __init__(self, path):
        self.path = path
        self.t = {}

    def before(self, event):
        self.t[event.tool_use["toolUseId"]] = time.time()

    def after(self, event):
        res = event.result or {}
        text = " ".join(c.get("text", "") for c in res.get("content", []) if isinstance(c, dict))
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "task": tools.BUDGET["task"],
               "tool": event.tool_use["name"], "input": event.tool_use.get("input"),
               "status": res.get("status"), "seconds": round(time.time() - self.t.pop(event.tool_use["toolUseId"], time.time()), 1),
               "result": text[:4000]}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def make_agent(logger: ToolLogger | None = None, model=None):
    from strands import Agent, tool
    from strands.agent.conversation_manager import SlidingWindowConversationManager
    from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent
    from strands.models import BedrockModel

    if model is None:
        model = BedrockModel(model_id=config.MODEL_ID, region_name=config.REGION, max_tokens=8000)
    agent = Agent(model=model, system_prompt=SYSTEM_PROMPT, tools=[tool(f) for f in tools.ALL_TOOLS],
                  conversation_manager=SlidingWindowConversationManager(window_size=80))
    if logger:
        agent.hooks.add_callback(BeforeToolCallEvent, logger.before)
        agent.hooks.add_callback(AfterToolCallEvent, logger.after)
    return agent


def run_task(task: str, budget_min: float, session_dir, model=None) -> str:
    tools.BUDGET.update(task=task, deadline=time.time() + 60 * budget_min)
    logger = ToolLogger(session_dir / "tool_calls.jsonl")
    agent = make_agent(logger, model)
    done = lambda: (config.RESULTS_DIR / f"{task}.lock.json").exists()
    left = lambda: (tools.BUDGET["deadline"] - time.time()) / 60
    prompt, result, errors, resumes = task_prompt(task, budget_min), "", 0, 0
    # Keep the agent working until it finalizes or the budget ends. If Bedrock throttles or the
    # model stops early, it is resumed with its own journal and leaderboard as memory.
    while True:
        try:
            result = agent(prompt)
        except Exception as e:
            errors += 1
            print(f"[gotrain] agent error ({type(e).__name__}: {e}); resuming", flush=True)
            if errors > 5:
                break
            time.sleep(min(60, 10 * errors))
            agent = make_agent(logger, model)   # fresh context; state lives on disk
        resumes += 1
        if done() or left() < 4 or resumes > 25:
            break
        prompt = (f"You have {left():.0f} minutes left for {task} and have not finalized. "
                  "Read read_journal and leaderboard, then continue the search; finalize before time runs out.")
    # safety net: if the model ended without finalizing, finalize the best VALID run
    if not (config.RESULTS_DIR / f"{task}.lock.json").exists():
        from .finalize import finalize
        fin = finalize(task, None, "auto-finalize: agent ended without calling finalize_task")
        (session_dir / f"{task}_autofinalize.json").write_text(json.dumps(fin, indent=1, default=str))
    (session_dir / f"{task}_messages.json").write_text(json.dumps(agent.messages, indent=1, default=str))
    (session_dir / f"{task}_final_report.md").write_text(str(result))
    return str(result)


def main():
    ap = argparse.ArgumentParser(description="Run the GoTrain agent")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--budget-min", type=float, default=90, help="minutes per task")
    a = ap.parse_args()
    session = config.LOGS_DIR / time.strftime("session_%Y%m%d_%H%M%S")
    session.mkdir(parents=True)
    for t in a.tasks:
        print(f"\n===== {t}: {a.budget_min:.0f} min =====\n", flush=True)
        run_task(t, a.budget_min, session)
    from .report import write_report
    print(write_report())


if __name__ == "__main__":
    main()
