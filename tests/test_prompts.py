"""The prompts tell Claude what to call; a misspelled or deleted tool name is a silent failure (the agent
improvises instead). Checked here, without Bedrock:

- every snake_case name in the system, task and resume prompts is a real tool, a tool argument, a
  training config key or a metrics field;
- every tool named with "(" in the strategy tools' hints ("next", "first") is a real tool;
- the method names the stages in order (data, then papers, then the assumptions, then the shortlist), and
  the noise rule is a hard rule; with the arXiv tool off no literature text is left behind;
- the resume prompt points back to the tournament state and keeps valid-only mode free of finalize.

    python tests/test_prompts.py
"""
import ast
import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SNAKE = r"\b[a-z0-9]+(?:_[a-z0-9]+)+\b"
METRICS_FIELDS = {"top_confusions"}          # written by train.metrics_from_probs, read by the agent


def main():
    from gotrain import prompts, strategy, tools
    from gotrain.train import DEFAULTS
    from gotrain.literature import search_papers       # named by the prompt variant with literature, even if off here
    every = list(tools.ALL_TOOLS) + [search_papers]
    names = {f.__name__ for f in every}
    args = {p for f in every for p in inspect.signature(f).parameters}

    # 1. prompts only name things that exist
    texts = {"system": prompts.build_system_prompt(True), "system_no_arxiv": prompts.build_system_prompt(False),
             "task": prompts.task_prompt("m-eurosat", 90),
             "resume": prompts.resume_prompt("m-eurosat", 30), "resume_valid": prompts.resume_prompt("m-eurosat", 30, True)}
    for k, t in texts.items():
        unknown = set(re.findall(SNAKE, t)) - names - args - set(DEFAULTS) - METRICS_FIELDS
        assert not unknown, (k, sorted(unknown))
        assert "<<" not in t, k                                  # every literature placeholder filled
    off = texts["system_no_arxiv"]
    assert "search_papers" not in off and "paper" not in off.lower()     # GOTRAIN_ARXIV=0: no dangling step

    # 2. the strategy hints only call real tools
    src = Path(inspect.getsourcefile(strategy)).read_text(encoding="utf-8")
    called = {m for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and isinstance(n.value, str)
              for m in re.findall(r"\b([a-z_]+)\(", n.value)}
    assert called and called <= names, sorted(called - names)

    # 3. the method in order, and the noise rule
    s = texts["system"]
    order = ["inspect_dataset", "quick_probe", "USE CASE", "search_papers", "COMPETING ASSUMPTIONS", "Step 2",
             "propose_candidates", "submit_shortlist", "search_hf_models", "Step 3", "race_decision", "Step 5",
             "branch_status", "compare_runs", "Step 6", "finalize_task"]
    method, at = s[s.index("METHOD"):], 0
    for t in order:                         # each step is named after the one before it
        assert t in method[at:], (t, "missing or out of order")
        at = method.index(t, at)
    rules = s[:s.index("METHOD")]
    assert "compare_runs" in rules and "Only a KEEP" in rules and "exactly once" in rules

    # 4. resume: back to the tournament state; valid-only never asks to finalize
    assert "branch_status" in texts["resume"] and "race_decision" in texts["resume"]
    assert "finalize before" in texts["resume"] and "finalize before" not in texts["resume_valid"]
    print(f"prompts name {len(set(re.findall(SNAKE, s)) & names)} of {len(names)} tools; hints call {sorted(called)}")
    print("PROMPTS OK")


if __name__ == "__main__":
    main()
