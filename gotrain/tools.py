"""The agent's tools. Plain Python functions returning JSON text.

The same functions are wrapped as Strands tools (agent.py) and as MCP tools (mcp_server.py),
so GoPilot can call exactly what the GoTrain agent calls.
The docstrings ARE the tool descriptions the model reads: keep them precise.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import config, data, registry
from .train import DEFAULTS

# Set by agent.py for each task session; tools refuse to start runs past the deadline.
BUDGET = {"deadline": None, "task": None}


def _j(obj) -> str:
    return json.dumps(obj, indent=1, default=str)


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _running_runs() -> list[str]:
    out = []
    for task_dir in config.RUNS_DIR.iterdir():
        if not task_dir.is_dir():
            continue
        for rd in task_dir.iterdir():
            st = rd / "status.json"
            if st.exists():
                s = json.loads(st.read_text())
                if s.get("state") == "running" and _alive(s.get("pid")):
                    out.append(rd.name)
    return out


# --------------------------------------------------------------------------- tools
def list_tasks() -> str:
    """List the 12 GEO-Bench tasks, whether each is downloaded, and whether its test split is already used.

    Returns JSON with classification and segmentation task names and their state.
    """
    def state(t):
        try:
            present = (config.task_dir(t) / "task_specs.pkl").exists()
        except Exception:
            present = False
        return {"task": t, "downloaded": present, "test_used": (config.RESULTS_DIR / f"{t}.lock.json").exists()}
    return _j({"classification": [state(t) for t in config.CLASSIFICATION_TASKS],
               "segmentation": [state(t) for t in config.SEGMENTATION_TASKS]})


def download_task(task: str) -> str:
    """Download one GEO-Bench task (only that task) from Hugging Face into GEO_BENCH_DIR.

    Args:
        task: task name, e.g. "m-eurosat" or "m-pv4ger-seg".
    """
    return data.download_task(task)


def inspect_dataset(task: str) -> str:
    """Explore a task using TRAIN and VALID only: kind (classification / multilabel / segmentation),
    number of classes and names, bands, input shape, dtype, label distribution, class imbalance,
    per-channel statistics, and known data issues. Builds the fast cache on first call (can take minutes).

    Args:
        task: task name.
    """
    return _j(data.describe(task))


def quick_probe(task: str) -> str:
    """Seconds-long diagnostic BEFORE deep learning: a random forest on per-band statistics and
    normalised band differences (the NDVI family), trained on TRAIN and scored on VALID.
    Returns the score floor, per-band importance and the most useful band pairs. Use it to decide
    whether the problem is spectral (band values) or spatial (shapes/texture), and to plan band ablations.
    It is a diagnostic, not a final model (it cannot be finalized).

    Args:
        task: task name.
    """
    from .probe import quick_probe as qp
    return _j(qp(task))


def training_options() -> str:
    """Show every training config key with its default and what it does, plus suggested backbones.
    Call before writing your first config."""
    return _j({
        "defaults": DEFAULTS,
        "primary_metric": {"classification": "accuracy", "multilabel": "f1_micro", "segmentation": "miou"},
        "classification_backbones (timm names)": ["resnet18", "resnet50", "convnext_tiny", "efficientnet_b0",
                                                  "efficientnet_b3", "vit_small_patch16_224 (needs img_size 224)",
                                                  "swin_tiny_patch4_window7_224 (needs img_size 224)"],
        "segmentation_encoders (smp names)": ["resnet18", "resnet34", "resnet50", "efficientnet-b0", "efficientnet-b4",
                                              "mit_b0", "mit_b2", "tu-convnext_tiny"],
        "segmentation_archs": ["unet", "unetplusplus", "fpn", "deeplabv3plus", "linknet", "pspnet"],
        "tips": [
            "Small images (e.g. 64x64): upsampling with img_size 128-224 often helps ImageNet backbones.",
            "Many bands (Sentinel-2): keep all channels; the first conv is adapted. Try channels=[RGB indices] as an ablation.",
            "Use train_fraction 0.2-0.3 and few epochs to screen ideas cheaply, then confirm on full data.",
            "Pretrained encoders like a lower lr than the head: backbone_lr_mult 0.1 is the default.",
            "Imbalanced classes: class_weights 'balanced' or loss 'focal'; segmentation: loss 'ce+dice'.",
            "tta true is a cheap gain at evaluation time.",
        ],
    })


def start_training(task: str, config_json: str, hypothesis: str, parent_run_id: str = "") -> str:
    """Start ONE training run in the background on the GPU. Trains on TRAIN, picks the epoch by VALID,
    never touches TEST. Only one run at a time. Returns the run_id; then poll with check_training.

    Args:
        task: task name.
        config_json: JSON object overriding training defaults, e.g. '{"backbone": "resnet50", "lr": 0.0005}'.
        hypothesis: one or two sentences: what you change versus the parent run and why you expect it to help.
        parent_run_id: the run this one builds on ("" for a first baseline).
    """
    if BUDGET["deadline"] and time.time() > BUDGET["deadline"]:
        return _j({"refused": True, "why": "time budget for this task is used up: finalize the best run now"})
    busy = _running_runs()
    if busy:
        return _j({"refused": True, "why": f"GPU busy with {busy}; wait with check_training or stop_training"})
    try:
        cfg = json.loads(config_json) if config_json.strip() else {}
    except json.JSONDecodeError as e:
        return _j({"refused": True, "why": f"config_json is not valid JSON: {e}"})
    unknown = [k for k in cfg if k not in DEFAULTS]
    if unknown:
        return _j({"refused": True, "why": f"unknown config keys {unknown}; see training_options"})
    if BUDGET["deadline"]:   # never let one run eat past the deadline
        left = (BUDGET["deadline"] - time.time()) / 60
        cfg["time_limit_min"] = round(min(cfg.get("time_limit_min", DEFAULTS["time_limit_min"]), max(1.0, left - 2)), 1)
    rd = registry.new_run(task, cfg, hypothesis, parent_run_id or None)
    logf = open(rd / "stdout.log", "w")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    p = subprocess.Popen([sys.executable, "-m", "gotrain.train", "--run-dir", str(rd)],
                         stdout=logf, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    (rd / "status.json").write_text(json.dumps({"state": "running", "pid": p.pid, "epoch": 0, "started": time.time()}))
    return _j({"run_id": rd.name, "pid": p.pid, "time_limit_min": cfg.get("time_limit_min", DEFAULTS["time_limit_min"]),
               "next": "call check_training(run_id, wait_minutes) to wait for results"})


def check_training(run_id: str, wait_minutes: float = 5.0) -> str:
    """Wait up to wait_minutes (max 20) for a run to finish, then report its status, the last epochs
    and its best VALID metrics. Returns early as soon as the run is done or failed.

    Args:
        run_id: id returned by start_training.
        wait_minutes: how long to wait before reporting.
    """
    rd = registry.run_dir(run_id)
    end = time.time() + 60 * min(max(wait_minutes, 0), 20)
    while True:
        st = json.loads((rd / "status.json").read_text())
        if st.get("state") == "running" and not _alive(st.get("pid")):
            time.sleep(2)
            st = json.loads((rd / "status.json").read_text())
            if st.get("state") == "running":
                st = {**st, "state": "failed", "error": "process died (out of memory?) - see read_run_log"}
                (rd / "status.json").write_text(json.dumps(st))
        if st.get("state") != "running" or time.time() >= end:
            break
        time.sleep(10)
    hist = []
    if (rd / "history.jsonl").exists():
        hist = [json.loads(l) for l in (rd / "history.jsonl").read_text().splitlines()[-6:]]
    out = {"run_id": run_id, "state": st.get("state"), "epoch": st.get("epoch"), "best_val": st.get("best_val"),
           "last_epochs": hist}
    if st.get("state") == "failed":
        out["error"] = st.get("error")
        out["traceback_tail"] = (st.get("traceback") or "")[-1500:]
    if (rd / "metrics.json").exists():
        out["final"] = json.loads((rd / "metrics.json").read_text())
    return _j(out)


def stop_training(run_id: str) -> str:
    """Kill a running training run (e.g. it is clearly worse than the best run). Its best checkpoint so far is kept.

    Args:
        run_id: the run to stop.
    """
    rd = registry.run_dir(run_id)
    st = json.loads((rd / "status.json").read_text())
    if _alive(st.get("pid")):
        os.killpg(os.getpgid(int(st["pid"])), signal.SIGTERM)
    st["state"] = "stopped"
    (rd / "status.json").write_text(json.dumps(st))
    return _j({"run_id": run_id, "state": "stopped", "best_val": st.get("best_val")})


def leaderboard(task: str) -> str:
    """All runs of a task sorted by VALID score, with their hypotheses, parents and configs.

    Args:
        task: task name.
    """
    return _j(registry.leaderboard(task))


def read_run_log(run_id: str, tail_lines: int = 40) -> str:
    """Last lines of a run's training log (use it to debug failures or inspect learning curves).

    Args:
        run_id: the run.
        tail_lines: number of lines from the end.
    """
    p = registry.run_dir(run_id) / "train.log"
    if not p.exists():
        p = registry.run_dir(run_id) / "stdout.log"
    lines = p.read_text().splitlines() if p.exists() else []
    return "\n".join(lines[-max(1, min(tail_lines, 200)):])


def record_insight(task: str, text: str) -> str:
    """Write to the task journal: what you observed, what you concluded, what you will try next and why.
    The journal is part of the deliverable: judges read it.

    Args:
        task: task name.
        text: a few sentences.
    """
    registry.journal(task, text)
    return "saved"


def read_journal(task: str) -> str:
    """Read the task journal (your earlier reasoning, useful after a long session or a restart).

    Args:
        task: task name.
    """
    return registry.read_journal(task) or "(empty)"


def write_custom_model(name: str, code: str) -> str:
    """Save a custom PyTorch model and smoke-test it. The code must define
    build_model(in_ch: int, n_classes: int, kind: str, cfg: dict) -> torch.nn.Module returning logits of shape
    (B, n_classes) for classification/multilabel or (B, n_classes, H, W) for segmentation (same H, W as input).
    Use it in a run with config {"custom_model": name}. timm and segmentation_models_pytorch are installed.

    Args:
        name: python identifier, e.g. "unet_attention".
        code: full python source.
    """
    if not name.isidentifier():
        return _j({"ok": False, "why": "name must be a python identifier"})
    p = config.CUSTOM_DIR / f"{name}.py"
    p.write_text(code)
    test = ("import importlib.util,torch;s=importlib.util.spec_from_file_location('m',%r);m=importlib.util.module_from_spec(s);"
            "s.loader.exec_module(m)\nfor kind,shape in [('classification',(2,)),('segmentation',(2,3,64,64))]:\n"
            "  net=m.build_model(4,3,kind,{});o=net(torch.randn(2,4,64,64))\n"
            "  assert o.shape[0]==2 and o.shape[1]==3,(kind,tuple(o.shape))\n  print(kind,tuple(o.shape))") % str(p)
    r = subprocess.run([sys.executable, "-c", test], capture_output=True, text=True, timeout=300)
    return _j({"ok": r.returncode == 0, "saved": str(p), "output": (r.stdout + r.stderr)[-2000:]})


def time_remaining() -> str:
    """Minutes left in the budget for the current task, and whether a run is using the GPU."""
    left = None if not BUDGET["deadline"] else round((BUDGET["deadline"] - time.time()) / 60, 1)
    return _j({"task": BUDGET["task"], "minutes_left": left, "running": _running_runs()})


def gpu_status() -> str:
    """GPU name, memory used/total and utilisation (nvidia-smi)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or r.stderr.strip()
    except FileNotFoundError:
        return "no GPU (nvidia-smi not found): training runs on CPU, keep models and epochs small"


def finalize_task(task: str, run_id: str, justification: str) -> str:
    """FINAL STEP, ONCE PER TASK: evaluate the chosen run on the TEST split and publish it as the
    production model version. Refused if the test split of this task was already used.
    Choose run_id by VALID score only (normally the leaderboard's top run).

    Args:
        task: task name.
        run_id: the run to evaluate.
        justification: why this run (VALID evidence only).
    """
    from .finalize import finalize
    return _j(finalize(task, run_id, justification))


def publish_run(run_id: str, stage: str = "candidate") -> str:
    """Upload a run's checkpoint, config, metrics and logs as a new model version
    (S3 if GOTRAIN_S3_BUCKET is set, local store otherwise) and record it in the model registry.

    Args:
        run_id: the run.
        stage: "candidate" (default) or "staging".
    """
    from .store import Store
    return _j(Store().publish_run(run_id, stage=stage))


def push_dataset(task: str) -> str:
    """Upload the task's prepared train/valid cache to the store so another machine can train without re-processing.

    Args:
        task: task name.
    """
    from .store import Store
    return _j(Store().push_dataset(task))


def model_versions(task: str) -> str:
    """List published model versions of a task (version, run, scores, stage, lineage).

    Args:
        task: task name.
    """
    from .store import Store
    return _j(Store().versions(task))


ALL_TOOLS = [list_tasks, download_task, inspect_dataset, quick_probe, training_options, start_training, check_training,
             stop_training, leaderboard, read_run_log, record_insight, read_journal, write_custom_model,
             time_remaining, gpu_status, finalize_task, publish_run, push_dataset, model_versions]
