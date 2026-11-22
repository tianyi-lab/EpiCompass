"""Run a method on NewtonBench and score the submitted laws with NewtonBench's own evaluator.

    python scripts/run_newtonbench.py --newtonbench_dir /path/to/NewtonBench --model gemini-3.1-flash-lite \
        --method epicompass --judge <judge model known to NewtonBench> --out results/newtonbench

One JSON per trial: <out>/<method>/<model>/<module>/<difficulty>/<version>/<system>/trial<k>.json (existing trials are
skipped, so an interrupted run can be resumed). The agent's and the component's LLM calls go through EPIC_BASE_URL;
the symbolic-equivalence judge goes through NewtonBench's own LLM configuration.
"""
import argparse
import importlib
import json
import os
import sys
import traceback
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYSTEMS = ["vanilla_equation", "simple_system", "complex_system"]


def one(job):
    a, module_name, diff, system, ver, trial = job
    out_dir = os.path.join(a.out, a.method, a.model.replace("/", "_"), module_name, diff, ver, system)
    path = os.path.join(out_dir, f"trial{trial}.json")
    if os.path.exists(path):
        return path, "skip"
    os.makedirs(out_dir, exist_ok=True)
    try:
        from epicompass.adapters.newtonbench import run_episode
        module = importlib.import_module(f"modules.{module_name}")
        res = run_episode(module, a.model, a.noise, diff, system, ver, max_turns=a.max_turns, method=a.method)
        ev = module.evaluate_law(res["submitted_law"], param_description=module.PARAM_DESCRIPTION, difficulty=diff,
                                 law_version=ver, judge_model_name=a.judge,
                                 trial_info={"trial_id": trial, "trial_dir": out_dir})
        rec = {"module_name": module_name, "equation_difficulty": diff, "model_system": system, "law_version": ver,
               "trial_id": trial, "noise_level": a.noise, "model_name": a.model, "method": a.method, **res,
               "evaluation": ev}
        with open(path + ".tmp", "w") as f:
            json.dump(rec, f, indent=1, default=str)
        os.replace(path + ".tmp", path)
        return path, f"symbolic={ev.get('symbolic_equivalent')} rmsle={ev.get('rmsle')}"
    except Exception:  # noqa: BLE001
        with open(path + ".err", "w") as f:
            f.write(traceback.format_exc())
        return path, "ERROR"


def init(nb_dir):
    os.chdir(nb_dir)
    sys.path[:0] = [nb_dir, ROOT]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--newtonbench_dir", default=os.environ.get("NEWTONBENCH_DIR"), required=not os.environ.get("NEWTONBENCH_DIR"))
    p.add_argument("--model", required=True, help="backbone name at EPIC_BASE_URL, e.g. gemini-3.1-flash-lite")
    p.add_argument("--method", default=os.environ.get("EPIC_METHOD", "epicompass"),
                   choices=["epicompass", "react", "hsearch", "popper"])
    p.add_argument("--judge", required=True, help="NewtonBench judge model for symbolic equivalence")
    p.add_argument("--modules", nargs="*", default=None, help="default: all m* modules")
    p.add_argument("--difficulties", nargs="*", default=["easy", "medium", "hard"])
    p.add_argument("--systems", nargs="*", default=SYSTEMS)
    p.add_argument("--versions", nargs="*", default=["v0", "v1", "v2"])
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--max_turns", type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=os.path.join(ROOT, "results", "newtonbench"))
    a = p.parse_args()
    a.newtonbench_dir = os.path.abspath(a.newtonbench_dir)
    a.out = os.path.abspath(a.out)
    os.environ["EPIC_METHOD"] = a.method
    modules = a.modules or sorted(m for m in os.listdir(os.path.join(a.newtonbench_dir, "modules"))
                                  if m.startswith("m") and os.path.isdir(os.path.join(a.newtonbench_dir, "modules", m)))
    jobs = [(a, m, d, s, v, t) for m in modules for d in a.difficulties for s in a.systems for v in a.versions
            for t in range(a.trials)]
    print(f"{len(jobs)} trials, method={a.method}, model={a.model}", flush=True)
    with Pool(a.workers, initializer=init, initargs=(a.newtonbench_dir,), maxtasksperchild=1) as pool:
        for path, msg in pool.imap_unordered(one, jobs):
            print(os.path.relpath(path, a.out), msg, flush=True)


if __name__ == "__main__":
    main()
