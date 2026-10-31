"""The model tournament and the branch search, as tools the agent calls. stats.py measures, this decides.

    propose_candidates  data facts + the feasible model menu (code filters only what cannot run)
    submit_shortlist    the agent's reasoned picks: checked against the rules, logged, turned into screen configs
    race_decision       screen -> who races (and with how many seeds); race -> the two branch roots
    compare_runs        the noise rule: KEEP only a gain that is significant AND above training noise
    branch_status       which branch is ahead, which runs next, whether to prune one

Everything is read from the run folders (config.json: train_fraction, seed, parent), so no extra state
is needed to know what a run is:
    screen run  train_fraction < 1
    race run    full data, parent is a screen run (or none); same config except the seed = one group
    branch run  full data, descends from a race run; the race run it descends from is the branch root
runs/<task>/strategy.json only remembers plans (shortlist, race seeds, branch roots, assumptions).
"""
from __future__ import annotations

import json
import math
import time

from . import config, registry, stats
from .train import CANDIDATE_POOL, DEFAULTS

ALPHA = 0.05            # significance level of the paired test
FLOOR_MULT = 1.0        # a kept gain must also exceed this many training-noise SDs (doc: 1, ideally 2)
DROP_SE = 2.0           # drop / prune what is more than this many SEs behind the leader
MIN_TRIES = 2           # a branch gets this many tries before it can be pruned
SCREEN = {"train_fraction": 0.3, "epochs": 10}
RACE_SIZE = 3           # models that go from the screen to the race
SEEDS = (2, 3, 5)       # min, default (no timing known), max seeds per raced model
MAX_PICKS, MIN_FAMILIES = 6, 3
SAFE_DEFAULT = {"classification": "resnet18", "multilabel": "resnet18", "segmentation": "unet-resnet34"}
IDENTITY = ("backbone", "seg_arch", "custom_model")     # what makes a run "the same model"
NOT_RECIPE = ("seed", "time_limit_min")                  # differences that keep runs in one race group


def _j(obj) -> str:
    return json.dumps(obj, indent=1, default=str)


def _name(kind: str, cfg: dict) -> str:
    cfg = {**DEFAULTS, **cfg}
    if cfg.get("custom_model"):
        return f"custom:{cfg['custom_model']}"
    return f"{cfg['seg_arch']}-{cfg['backbone']}" if kind == "segmentation" else cfg["backbone"]


def _pool(kind: str) -> dict:
    return {_name(kind, e["config"]): e for e in CANDIDATE_POOL[kind]}


def _state(task: str) -> dict:
    p = registry.task_runs_dir(task) / "strategy.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save_state(task: str, **kw) -> None:
    st = {**_state(task), **kw}
    (registry.task_runs_dir(task) / "strategy.json").write_text(json.dumps(st, indent=1))


def _minutes_left():
    from .tools import BUDGET     # late import: tools.py imports this module
    return None if not BUDGET["deadline"] else (BUDGET["deadline"] - time.time()) / 60


# --------------------------------------------------------------------------- reading runs
def _runs(task: str) -> list[dict]:
    """Every run of the task with what the strategy needs to know about it."""
    out = []
    for r in registry.all_runs(task):
        st = r["status"] or {}
        has_preds = (registry.run_dir(r["run_id"]) / "val_preds.npz").exists()
        out.append({"id": r["run_id"], "cfg": r["config"], "parent": r["parent"], "state": st.get("state", "created"),
                    "ok": st.get("state") == "done" and has_preds, "minutes": (r["metrics"] or {}).get("minutes") or 0,
                    "screen": r["config"].get("train_fraction", 1.0) < 1, "hypothesis": r["hypothesis"]})
    kind = next((stats.load(r["id"]).kind for r in out if r["ok"]), None)
    pool = _pool(kind) if kind else {}
    for r in out:
        r["kind"] = kind
        r["model"] = _name(kind or "classification", r["cfg"])
        r["family"] = pool[r["model"]]["family"] if r["model"] in pool else r["model"]
        r["score"] = stats.score(r["id"]) if r["ok"] else None
    return out


def _race_runs(runs: list[dict]) -> list[dict]:
    screens = {r["id"] for r in runs if r["screen"]}
    return [r for r in runs if not r["screen"] and (r["parent"] is None or r["parent"] in screens)]


def _gkey(r: dict) -> str:
    return json.dumps({k: v for k, v in sorted(r["cfg"].items()) if k not in NOT_RECIPE})


def _groups(race: list[dict]) -> dict:
    """Race runs grouped by config, ignoring the seed: one group = one model trained several times."""
    g = {}
    for r in race:
        g.setdefault(_gkey(r), []).append(r)
    return g


def _floor(runs: list[dict]) -> dict:
    groups = [[r["id"] for r in g if r["ok"]] for g in _groups(_race_runs(runs)).values()]
    return stats.noise_floor(groups)


# --------------------------------------------------------------------------- 1. the menu
def task_facts(task: str) -> dict:
    """Facts that decide what can run: kind, bands, image size, dataset size."""
    from . import data
    meta = data.task_meta(task)
    Xtr, _ = data.load_split(task, "train")
    Xva, _ = data.load_split(task, "valid")
    return {"kind": meta["kind"], "n_classes": meta["n_classes"], "bands": int(Xtr.shape[1]),
            "image_size": [int(Xtr.shape[2]), int(Xtr.shape[3])], "train_n": int(len(Xtr)), "valid_n": int(len(Xva)),
            "known_issue": meta.get("known_issue")}


def _timing(task: str):
    p = config.HOME / "timing.json"
    t = json.loads(p.read_text()).get(task) if p.exists() else None
    return t["sec_per_epoch"] if t else None


def _menu(task: str, facts: dict) -> list[dict]:
    kind, bands, size, n = facts["kind"], facts["bands"], min(facts["image_size"]), facts["train_n"]
    sec = _timing(task)                         # measured with the default resnet18: scale by size, roughly
    ref = CANDIDATE_POOL[kind][0]["params_M"]
    left = _minutes_left()
    menu = []
    for name, e in _pool(kind).items():
        notes = [e["suits"]]
        if "SSL4EO" in e["weights"]:
            notes.append("pretrained on 13-band Sentinel-2: matches this data" if bands >= 12 else
                         f"pretrained on 13-band Sentinel-2 but this data has {bands} bands: the weights fit less well")
        if e["family"] == "segformer" and bands > 3:
            notes.append(f"RGB-only encoder: a learned {bands}->3 band adapter is added in front")
        if e["params_M"] > 20 and n < 2000:
            notes.append(f"{e['params_M']}M parameters for {n} train images: overfitting risk")
        if e["config"].get("img_size") and size < e["config"]["img_size"]:
            notes.append(f"images are {size}px; this config upsamples to {e['config']['img_size']}px (slower)")
        est = None
        if sec:
            scale = (e["params_M"] / ref) ** 0.7 * (e["config"].get("img_size", size) / size) ** 2
            est = round(sec * scale * SCREEN["epochs"] * SCREEN["train_fraction"] / 60, 1)
        fits = est is None or left is None or est < 0.15 * left       # one screen run <= 15% of what is left
        menu.append({"name": name, "family": e["family"], "weights": e["weights"], "params_M": e["params_M"],
                     "screen_minutes_est": est, "feasible": fits, "notes": notes,
                     **({"why_not": f"one screen run ~{est} min of {left:.0f} left"} if not fits else {})})
    return menu


def propose_candidates(task: str) -> str:
    """Step 1 of the model tournament. Returns the data facts and the menu of candidate models (with
    family, pretraining, size, rough screen time and notes on how each fits THIS data). Nothing is
    chosen here: read it together with inspect_dataset and quick_probe, reason about which models fit
    the data, then call submit_shortlist with your picks and why.

    Args:
        task: task name.
    """
    facts = task_facts(task)
    return _j({"facts": facts, "safe_default": SAFE_DEFAULT[facts["kind"]], "menu": _menu(task, facts),
               "rules": f"pick {MIN_FAMILIES}-{MAX_PICKS} feasible models from at least {MIN_FAMILIES} families; "
                        f"the safe default is always screened",
               "next": "submit_shortlist(task, picks_json, assumptions_json)"})


# --------------------------------------------------------------------------- 2. the agent's picks
def shortlist(task: str, picks: list[dict], assumptions: list[str], facts: dict) -> dict:
    kind = facts["kind"]
    menu = {m["name"]: m for m in _menu(task, facts)}
    if not 1 <= len(picks) <= MAX_PICKS:
        return {"refused": True, "why": f"pick between {MIN_FAMILIES} and {MAX_PICKS} models"}
    bad = [p.get("name") for p in picks if p.get("name") not in menu]
    if bad:
        return {"refused": True, "why": f"not on the menu: {bad}; choose from {sorted(menu)}"}
    no_why = [p["name"] for p in picks if len(str(p.get("why", "")).strip()) < 10]
    if no_why:
        return {"refused": True, "why": f"give a reason ('why': how it fits this data) for {no_why}"}
    slow = [p["name"] for p in picks if not menu[p["name"]]["feasible"]]
    if slow:
        return {"refused": True, "why": f"too slow for the time left: {slow}"}
    picks = list({p["name"]: p for p in picks}.values())
    added = []
    if SAFE_DEFAULT[kind] not in {p["name"] for p in picks}:
        picks.append({"name": SAFE_DEFAULT[kind], "why": "safe default: always screened, so reasoning can only "
                                                         "add options, never remove a reliable one", "risk": ""})
        added.append(SAFE_DEFAULT[kind])
    fams = {menu[p["name"]]["family"] for p in picks}
    all_fams = {m["family"] for m in menu.values() if m["feasible"]}
    if len(fams) < min(MIN_FAMILIES, len(all_fams)):
        return {"refused": True, "why": f"only families {sorted(fams)}: diverse starts need {MIN_FAMILIES}; "
                                        f"available {sorted(all_fams)}"}
    pool = _pool(kind)
    screen = [{"name": p["name"], "config": {**pool[p["name"]]["config"], **SCREEN},
               "hypothesis": f"screen {p['name']}: {p['why']}"} for p in picks]
    _save_state(task, shortlist=picks, assumptions=assumptions)
    lines = ["MODEL SHORTLIST (reasoned before any training)"]
    lines += [f"- {p['name']} ({menu[p['name']]['family']}): {p['why']}" + (f" | risk: {p['risk']}" if p.get("risk") else "")
              for p in picks]
    if assumptions:
        lines += ["Competing assumptions about the data (one per branch):"] + [f"- {a}" for a in assumptions]
    registry.journal(task, "\n".join(lines))
    return {"accepted": [p["name"] for p in picks], "added_safe_default": added, "families": sorted(fams),
            "screen_runs": screen,
            "next": "start_training each screen run (parent_run_id ''), one at a time; then race_decision(task)"}


def submit_shortlist(task: str, picks_json: str, assumptions_json: str = "[]") -> str:
    """Step 2 of the model tournament: submit the models you chose from propose_candidates, each with
    your reasoning. Checked against the rules (on the menu, feasible, at least 3 families; the safe
    default is added if missing), written to the journal, and turned into cheap screen configs.

    Args:
        task: task name.
        picks_json: JSON list, e.g. '[{"name": "ssl4eo_resnet18", "why": "13 Sentinel-2 bands and the probe says
            spectral: weights pretrained on the same sensor", "risk": "slower to converge"}]'.
        assumptions_json: JSON list of the two competing assumptions about the data the branches will test,
            e.g. '["spectral: the band values carry the signal", "spatial: shapes and texture carry it"]'.
    """
    try:
        picks, assumptions = json.loads(picks_json), json.loads(assumptions_json or "[]")
    except json.JSONDecodeError as e:
        return _j({"refused": True, "why": f"not valid JSON: {e}"})
    return _j(shortlist(task, picks, assumptions, task_facts(task)))


# --------------------------------------------------------------------------- 3. screen and race
def _n_seeds(task: str, n_models: int) -> int:
    sec, left = _timing(task), _minutes_left()
    if not sec or not left:
        return SEEDS[1]
    t_full = sec * DEFAULTS["epochs"] / 60                    # upper bound: early stopping usually ends sooner
    return max(SEEDS[0], min(SEEDS[2], math.floor(0.25 * left / (n_models * t_full))))


def screen_result(task: str, runs: list[dict]) -> dict:
    done = [r for r in runs if r["screen"] and r["ok"]]
    if len(done) < 2:
        return {"stage": "screen", "waiting": True, "why": "fewer than 2 finished screen runs"}
    # A screen run is one cheap run: its gaps hold val-image noise but not training noise (no seeds yet),
    # so significance alone would drop models that are only unlucky. The screen therefore only ranks:
    # the top RACE_SIZE race (at least two families); the rest are reported with their gap.
    ranked = sorted(done, key=lambda r: -r["score"])
    lead = ranked[0]
    race = ranked[:RACE_SIZE]
    if len({r["family"] for r in race}) < 2:   # the race must hold at least two families
        other = next((r for r in ranked[RACE_SIZE:] if r["family"] != lead["family"]), None)
        if other:
            race = race[:RACE_SIZE - 1] + [other]
    rows = []
    for r in ranked:
        row = {"run": r["id"], "model": r["model"], "family": r["family"], "score": r["score"]}
        if r is not lead:
            t = stats.paired_test(r["id"], lead["id"])
            row.update(gap=t["diff"], p=t["p"], clearly_behind=bool(t["p"] < ALPHA and -t["diff"] > DROP_SE * t["se"]))
        row["verdict"] = "race" if r in race else "not raced"
        rows.append(row)
    k = _n_seeds(task, len(race))
    pool = _pool(runs[0]["kind"])
    plan = [{"name": r["model"], "parent_run_id": r["id"], "seed": s,
             "config": {**{kk: v for kk, v in r["cfg"].items() if kk not in SCREEN}, "seed": s},
             "hypothesis": f"race {r['model']}, seed {s}: full data, repeated to separate skill from luck"}
            for r in race for s in range(k)]
    _save_state(task, race={"models": [r["model"] for r in race], "seeds": k})
    return {"stage": "screen", "results": rows, "race_models": [r["model"] for r in race],
            "families": {r["model"]: pool.get(r["model"], {}).get("family", r["model"]) for r in race},
            "seeds": k, "race_runs": plan,
            "next": "start_training every race run (full data), one at a time; then race_decision(task) again"}


def race_result(task: str, runs: list[dict]) -> dict:
    race = _race_runs(runs)
    if any(r["state"] == "running" for r in race):
        return {"stage": "race", "waiting": True, "why": "race runs still training: wait with check_training"}
    groups = [g for g in _groups(race).values() if any(r["ok"] for r in g)]
    if len(groups) < 2:
        return {"stage": "race", "waiting": True, "why": "fewer than 2 raced models have finished runs"}
    rows = []
    for g in groups:
        ok = [r for r in g if r["ok"]]
        gs = stats.group_score([r["id"] for r in ok])
        rows.append({"model": ok[0]["model"], "family": ok[0]["family"], "mean": gs["mean"], "se": gs["se"],
                     "n_runs": gs["n_runs"], "failed": len(g) - len(ok), "runs": [r["id"] for r in ok],
                     "best_run": max(ok, key=lambda r: r["score"])["id"]})
    rows.sort(key=lambda x: -x["mean"])
    floor = _floor(runs)
    a = rows[0]
    b = next((x for x in rows[1:] if x["family"] != a["family"]), None)
    skipped = [x["model"] for x in rows[1:rows.index(b)]] if b else []
    for x in rows:
        x["verdict"] = "branch A" if x is a else "branch B" if x is b else (
            "skipped: same family as the leader" if x["model"] in skipped else "out")
    roots = [a["best_run"]] + ([b["best_run"]] if b else [])
    st = _state(task)
    _save_state(task, roots=roots)
    return {"stage": "race", "ranking": rows, "noise_floor": floor,
            "branches": {"A": {"root": a["best_run"], "model": a["model"]},
                         **({"B": {"root": b["best_run"], "model": b["model"]}} if b else {})},
            "assumptions": st.get("assumptions", []),
            "next": "give each branch the assumption it suits; start each branch's runs with parent_run_id = its root "
                    "(or a kept run of that branch); after every run call compare_runs, between runs branch_status"}


def race_decision(task: str) -> str:
    """Steps 3-4 of the model tournament. After the screen runs: which models are clearly behind
    (dropped) and which race, with ready configs and the number of seeds. After the race runs: models
    ranked by their mean over seeds +- SE, the noise floor, and the two winners from DIFFERENT families
    that become the roots of branch A and branch B.

    Args:
        task: task name.
    """
    runs = _runs(task)
    if not any(r["ok"] for r in runs):
        return _j({"waiting": True, "why": "no finished run with predictions yet"})
    return _j(race_result(task, runs) if any(r["ok"] for r in _race_runs(runs)) else screen_result(task, runs))


# --------------------------------------------------------------------------- 4. keep or revert
def compare(task: str, run_id: str, parent_id: str, runs: list[dict]) -> dict:
    byid = {r["id"]: r for r in runs}
    for rid in (run_id, parent_id):
        if rid not in byid:
            return {"refused": True, "why": f"unknown run {rid}"}
        if not byid[rid]["ok"]:
            return {"verdict": "FAILED", "run": rid, "state": byid[rid]["state"],
                    "advice": "no predictions: read_run_log, fix the cause, keep the parent"}
    t = stats.paired_test(run_id, parent_id)
    floor = _floor(runs)
    need = None if floor["diff_sd"] is None else FLOOR_MULT * floor["diff_sd"]
    if t["diff"] > 0 and t["p"] < ALPHA and (need is None or t["diff"] > need):
        verdict, why = "KEEP", "the gain is significant and larger than training noise"
    elif t["diff"] < 0 and t["p"] < ALPHA:
        verdict, why = "REVERT", "clearly worse: build on the parent"
    else:
        verdict, why = "TIE", ("the difference is inside the noise: keep the parent (or the cheaper of the two); "
                               "the idea is not proven, do not stack more changes on it")
    if verdict == "TIE" and t["diff"] > 0 and t["p"] < ALPHA:
        why = f"significant on these images but below training noise ({need:.4f}): " + why
    return {"verdict": verdict, "why": why, "run": run_id, "parent": parent_id, "metric": t["metric"],
            "run_score": byid[run_id]["score"], "parent_score": byid[parent_id]["score"],
            "gain": t["diff"], "gain_95": [t["low"], t["high"]], "p": t["p"], "noise_floor": need,
            **({"a_only_right": t["a_only_right"], "b_only_right": t["b_only_right"]} if "a_only_right" in t else {})}


def compare_runs(run_id: str, parent_run_id: str = "") -> str:
    """The noise rule. Call it after EVERY finished run, before building on it. Compares the run with
    its parent image by image on VALID. KEEP only if the gain is significant AND larger than training
    noise (measured from the race seeds); REVERT if clearly worse; TIE otherwise (keep the parent).

    Args:
        run_id: the new run.
        parent_run_id: the run to compare with ("" = the run's own parent).
    """
    task = run_id.split("__")[0]
    parent = parent_run_id or registry.read(run_id)["parent"]
    if not parent:
        return _j({"refused": True, "why": f"{run_id} has no parent: pass parent_run_id"})
    return _j(compare(task, run_id, parent, _runs(task)))


# --------------------------------------------------------------------------- 5. branches
def branches(task: str, runs: list[dict]) -> dict:
    """A branch = one raced model and every run that descends from its race runs. Keyed by the race
    group, so a branch may grow from any of that model's seeds."""
    byid = {r["id"]: r for r in runs}
    race = {r["id"]: r for r in _race_runs(runs)}
    roots = set(_state(task).get("roots", [])) | {r["parent"] for r in runs if r["parent"] in race and not r["screen"]}
    roots &= set(race)
    if not roots:
        return {"waiting": True, "why": "no branch roots yet: finish the race and call race_decision"}
    branch_of = {}                              # run id -> race group key of its branch
    for r in runs:
        x = r["id"]
        while x is not None and x not in race:
            x = byid[x]["parent"] if x in byid else None
        if x is not None and (x in roots or r["id"] in roots):
            branch_of[r["id"]] = _gkey(race[x])
    groups = _groups(list(race.values()))
    keys = sorted(set(branch_of.values()),      # A = the race winner: order by the race mean
                  key=lambda k: -stats.group_score([r["id"] for r in groups[k] if r["ok"]])["mean"])
    rows = {}
    for i, key in enumerate(keys):
        members = [byid[rid] for rid, k in branch_of.items() if k == key]
        tries = [m for m in members if m["id"] not in race]
        best = max((m for m in members if m["ok"]), key=lambda m: m["score"])
        ci = stats.bootstrap_ci(best["id"])
        rows[chr(ord("A") + i)] = {"model": groups[key][0]["model"], "roots": sorted(m["id"] for m in members if m["id"] in race),
                                   "best_run": best["id"], "best_score": best["score"], "best_95": [ci["low"], ci["high"]],
                                   "tries": len(tries), "failed": sum(m["state"] == "failed" for m in tries),
                                   "running": [m["id"] for m in tries if m["state"] == "running"],
                                   "minutes": round(sum(m["minutes"] for m in tries), 1)}
    out = {"branches": rows, "assumptions": _state(task).get("assumptions", [])}
    if len(rows) < 2:
        return {**out, "next": "A", "why": "only one branch"}
    floor = _floor(runs)
    lead, trail = sorted(rows, key=lambda k: -rows[k]["best_score"])[:2]
    t = stats.paired_test(rows[lead]["best_run"], rows[trail]["best_run"])
    clear = (t["p"] < ALPHA and t["diff"] > DROP_SE * t["se"]
             and (floor["diff_sd"] is None or t["diff"] > FLOOR_MULT * floor["diff_sd"]))
    out["gap"] = {"leader": lead, "gain": t["diff"], "p": t["p"], "clearly_ahead": clear}
    if clear and rows[trail]["tries"] >= MIN_TRIES:
        out.update(prune=trail, next=lead, why=f"branch {lead} is clearly ahead after {rows[trail]['tries']} tries "
                                               f"of branch {trail}: stop {trail}, give its time to {lead}")
    else:
        nxt = min(rows, key=lambda k: (rows[k]["minutes"], rows[k]["tries"]))
        out.update(next=nxt, why=("no branch is clearly ahead yet" if not clear else
                                  f"branch {lead} leads, but {trail} has had only {rows[trail]['tries']} tries")
                   + f": run branch {nxt} next (least GPU time so far)")
    out["ensemble_candidates"] = [rows[k]["best_run"] for k in rows if k != out.get("prune")]
    return out


def branch_status(task: str) -> str:
    """Call between runs once the branches exist. For each branch (A, B, ...): its root model, best run
    and score with 95% interval, tries, failures and GPU minutes. Says which branch to run next (they
    take turns on the GPU) and whether one is clearly behind and should be pruned.

    Args:
        task: task name.
    """
    return _j(branches(task, _runs(task)))


ALL_STRATEGY_TOOLS = [propose_candidates, submit_shortlist, race_decision, compare_runs, branch_status]
