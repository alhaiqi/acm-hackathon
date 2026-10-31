"""Run the real Strands ReAct loop offline with a scripted model instead of Bedrock.

Checks the plumbing end to end: tool calls are executed, logged to tool_calls.jsonl,
runs are trained, the agent is NOT allowed to finalize twice, and auto-finalize
kicks in for a task the "model" forgot to finalize.

    GEO_BENCH_DIR=... GOTRAIN_HOME=... python tests/test_agent_loop.py
"""
import json
import time

from strands.models.model import Model

from gotrain import config
from gotrain.agent import run_task


class Scripted(Model):
    def __init__(self, script):
        self.script, self.i = script, 0

    def update_config(self, **kw):
        pass

    def get_config(self):
        return {}

    async def structured_output(self, *a, **k):
        raise NotImplementedError

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kw):
        step = self.script[self.i] if self.i < len(self.script) else ("text", "done")
        self.i += 1
        yield {"messageStart": {"role": "assistant"}}
        if step[0] == "tool":
            # let a step's input depend on the last tool result (e.g. use the returned run_id)
            last = json.dumps(messages[-1]["content"]) if messages else ""
            inp = step[2](last) if callable(step[2]) else step[2]
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": f"t{self.i}", "name": step[1]}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(inp)}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockDelta": {"delta": {"text": step[1]}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}


def run_id_from(last):
    import re
    return {"run_id": re.search(r"(m-[\w-]+__r\d{3})", last).group(1), "wait_minutes": 5}


task = "m-eurosat"
script = [
    ("tool", "inspect_dataset", {"task": task}),
    ("tool", "quick_probe", {"task": task}),
    ("tool", "record_insight", {"task": task, "text": "3 balanced classes, 4 bands, 32px: start with resnet18."}),
    ("tool", "start_training", {"task": task, "hypothesis": "baseline",
                                "config_json": json.dumps({"pretrained": False, "epochs": 3, "num_workers": 0})}),
    ("tool", "check_training", run_id_from),
    ("tool", "start_training", {"task": task, "hypothesis": "upsample to 64 px", "parent_run_id": f"{task}__r001",
                                "config_json": json.dumps({"pretrained": False, "epochs": 3, "num_workers": 0, "img_size": 64})}),
    ("tool", "check_training", run_id_from),
    ("tool", "leaderboard", {"task": task}),
    ("tool", "finalize_task", {"task": task, "run_id": f"{task}__r001", "justification": "best val"}),
    ("tool", "finalize_task", {"task": task, "run_id": f"{task}__r002", "justification": "try again (must be refused)"}),
    ("text", "Final report: done."),
]
seg_script = [("tool", "start_training", {"task": "m-pv4ger-seg", "hypothesis": "baseline", "config_json":
               json.dumps({"pretrained": False, "epochs": 2, "num_workers": 0})}),
              ("tool", "check_training", run_id_from), ("text", "forgot to finalize")]

if __name__ == "__main__":
    session = config.LOGS_DIR / time.strftime("test_%H%M%S")
    session.mkdir(parents=True)
    print(run_task(task, 10, session, model=Scripted(script)))
    print(run_task("m-pv4ger-seg", 10, session, model=Scripted(seg_script)))
    calls = [json.loads(l) for l in (session / "tool_calls.jsonl").read_text().splitlines()]
    print(len(calls), "tool calls logged:", [c["tool"] for c in calls])
    second = [c for c in calls if c["tool"] == "finalize_task"][-1]
    assert '"refused": true' in second["result"], second
    assert (session / "m-pv4ger-seg_autofinalize.json").exists()
    assert json.loads((config.RESULTS_DIR / "m-pv4ger-seg.json").read_text())["test_score"] >= 0
    print("AGENT LOOP OK")
