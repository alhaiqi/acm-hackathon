"""Strands hook providers: the agent's rules enforced by the framework, before any tool runs.

The tools keep their own checks (defence in depth); these hooks make the policy visible in one place
and apply it to every tool, including ones added later:

  GoTrainGuards  - one task per session: a tool call naming another task is cancelled
                 - the test split: finalize_task is cancelled if this task's lock file exists
                 - the budget: after the deadline only read/finalize tools may run
  ToolCallLog    - every tool call, its arguments, result and duration -> logs/<session>/tool_calls.jsonl
"""
from __future__ import annotations

import json
import time

from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent, HookProvider, HookRegistry

from . import config, tools

# tools that spend GPU time or extend the search: refused once the budget is used up
SPENDING = {"start_training", "write_custom_model", "submit_shortlist", "race_decision", "search_hf_models",
            "add_hf_candidate", "download_task"}


class GoTrainGuards(HookProvider):
    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeToolCallEvent, self.check)

    def check(self, event: BeforeToolCallEvent) -> None:
        name = event.tool_use["name"]
        args = event.tool_use.get("input") or {}
        task = tools.BUDGET["task"]
        named = args.get("task") or (args.get("run_id") or "").split("__")[0] or None
        if task and named and named != task:
            event.cancel_tool = f"refused by guard: this session works on {task}, not {named}"
        elif name == "finalize_task" and task and (config.RESULTS_DIR / f"{task}.lock.json").exists():
            event.cancel_tool = f"refused by guard: the test split of {task} was already used (lock file exists)"
        elif name in SPENDING and tools.BUDGET["deadline"] and time.time() > tools.BUDGET["deadline"]:
            event.cancel_tool = "refused by guard: the time budget is used up; finalize the best VALID run now"


class ToolCallLog(HookProvider):
    def __init__(self, path):
        self.path = path
        self.t = {}

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeToolCallEvent, self.before)
        registry.add_callback(AfterToolCallEvent, self.after)

    def before(self, event: BeforeToolCallEvent) -> None:
        self.t[event.tool_use["toolUseId"]] = time.time()

    def after(self, event: AfterToolCallEvent) -> None:
        res = event.result or {}
        text = " ".join(c.get("text", "") for c in res.get("content", []) if isinstance(c, dict))
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "task": tools.BUDGET["task"],
               "tool": event.tool_use["name"], "input": event.tool_use.get("input"),
               "status": res.get("status"), "cancelled": bool(getattr(event, "cancel_message", None)),
               "seconds": round(time.time() - self.t.pop(event.tool_use["toolUseId"], time.time()), 1),
               "result": text[:4000]}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
