"""The GoTrain agent: Claude on Bedrock + Strands ReAct loop + GoTrain tools.

    python -m gotrain.agent --tasks m-eurosat m-pv4ger-seg --budget-min 90

One fresh agent per task (clean context, independent budget). Every tool call, its arguments,
result and duration is written to logs/<session>/tool_calls.jsonl, and the full conversation to
logs/<session>/<task>_messages.json: this is how every score can be traced back.

Strands features used:
  Agent + BedrockModel + @tool        the ReAct loop on Claude (Bedrock)
  hooks (guards.py)                   rules enforced before any tool runs; tool-call log
  FileSessionManager                  the conversation is persisted; after a Bedrock error the agent
                                      resumes the same conversation instead of starting blank
  SlidingWindowConversationManager    long searches stay inside the context window
  structured_output_model             the final report is a validated Pydantic object (numbers from disk)
"""
from __future__ import annotations

import argparse
import json
import time

from pydantic import BaseModel, Field

from . import config, tools
from .guards import GoTrainGuards, ToolCallLog
from .prompts import SYSTEM_PROMPT, resume_prompt, task_prompt

ToolLogger = ToolCallLog   # old name


class RunRow(BaseModel):
    run_id: str
    change: str = Field(description="what this run changed versus its parent, in a few words")
    val_score: float | None = None


class FinalReport(BaseModel):
    """The agent's final report. Scores are overwritten from results/<task>.json, never trusted from text."""
    task: str
    chosen_run_ids: list[str] = Field(description="the run(s) evaluated on test")
    why_chosen: str = Field(description="the VALID evidence for this choice")
    runs: list[RunRow] = Field(description="every run of this task, in the order they were trained")
    lessons: list[str] = Field(description="the three most important lessons, one sentence each")
    val_score: float | None = None
    test_score: float | None = None


def make_agent(logger: ToolCallLog | None = None, model=None, session_manager=None):
    from strands import Agent, tool
    from strands.agent.conversation_manager import SlidingWindowConversationManager
    from strands.models import BedrockModel

    if model is None:
        model = BedrockModel(model_id=config.MODEL_ID, region_name=config.REGION, max_tokens=8000)
    hooks = [GoTrainGuards()] + ([logger] if logger else [])
    return Agent(model=model, system_prompt=SYSTEM_PROMPT, tools=[tool(f) for f in tools.ALL_TOOLS],
                 conversation_manager=SlidingWindowConversationManager(window_size=80), hooks=hooks,
                 session_manager=session_manager)


def _session(task: str, session_dir, attempt: int):
    from strands.session import FileSessionManager
    return FileSessionManager(session_id=f"{task}-{attempt}", storage_dir=str(session_dir / "strands_sessions"))


def final_report(agent, task: str, session_dir) -> dict | None:
    """Ask for the report as a validated object; fill the scores from the result file on disk."""
    try:
        res = agent("Write your final report for this task: every run, the chosen run(s) and the three lessons.",
                    structured_output_model=FinalReport)
        rep = res.structured_output.model_dump()
    except Exception as e:
        print(f"[gotrain] structured final report failed ({type(e).__name__}: {e})", flush=True)
        return None
    p = config.RESULTS_DIR / f"{task}.json"
    if p.exists():
        r = json.loads(p.read_text())
        rep.update(task=task, chosen_run_ids=r.get("run_ids", [r["run_id"]]), val_score=r["val_score"],
                   test_score=r["test_score"])
    (session_dir / f"{task}_final_report.json").write_text(json.dumps(rep, indent=1))
    return rep


VALID_ONLY_NOTE = (" EXPERIMENT MODE (validation only): finalize_task is disabled and the test split is not "
                   "available. Search for the best VALID score until the budget ends, then report your best run.")


def run_task(task: str, budget_min: float, session_dir, model=None, valid_only: bool = False) -> str:
    """valid_only: a search experiment (e.g. agent vs random search). The guard cancels finalize_task and
    there is no auto-finalize, so the test split is never read."""
    tools.BUDGET.update(task=task, deadline=time.time() + 60 * budget_min, valid_only=valid_only)
    logger = ToolCallLog(session_dir / "tool_calls.jsonl")
    attempt = 0
    agent = make_agent(logger, model, _session(task, session_dir, attempt))
    done = lambda: not valid_only and (config.RESULTS_DIR / f"{task}.lock.json").exists()
    left = lambda: (tools.BUDGET["deadline"] - time.time()) / 60
    prompt, result, errors, resumes = task_prompt(task, budget_min) + (VALID_ONLY_NOTE if valid_only else ""), "", 0, 0
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
            # resume the same persisted conversation; after two failed resumes start a clean one
            # (the conversation itself may be the problem). Either way the run state lives on disk.
            if errors > 2:
                attempt += 1
            agent = make_agent(logger, model, _session(task, session_dir, attempt))
        resumes += 1
        # normal mode keeps ~4 min for finalize; a valid-only experiment uses the whole budget (as its baseline does)
        if done() or left() < (1 if valid_only else 4) or resumes > 25:
            break
        prompt = resume_prompt(task, left(), valid_only)
    # safety net: if the model ended without finalizing, finalize the best VALID run (never in valid_only mode)
    if not valid_only and not (config.RESULTS_DIR / f"{task}.lock.json").exists():
        from .finalize import finalize
        fin = finalize(task, None, "auto-finalize: agent ended without calling finalize_task")
        (session_dir / f"{task}_autofinalize.json").write_text(json.dumps(fin, indent=1, default=str))
    final_report(agent, task, session_dir)
    (session_dir / f"{task}_messages.json").write_text(json.dumps(agent.messages, indent=1, default=str))
    (session_dir / f"{task}_final_report.md").write_text(str(result))
    return str(result)


def main():
    ap = argparse.ArgumentParser(description="Run the GoTrain agent")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--budget-min", type=float, default=90, help="minutes per task")
    ap.add_argument("--valid-only", action="store_true", help="search experiment: no finalize, test never read")
    a = ap.parse_args()
    session = config.LOGS_DIR / time.strftime("session_%Y%m%d_%H%M%S")
    session.mkdir(parents=True)
    for t in a.tasks:
        print(f"\n===== {t}: {a.budget_min:.0f} min{' (valid only)' if a.valid_only else ''} =====\n", flush=True)
        run_task(t, a.budget_min, session, valid_only=a.valid_only)
    from .report import write_report
    print(write_report())


if __name__ == "__main__":
    main()
