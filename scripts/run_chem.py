"""Run a method on ActiveSciBench-Chem (LLM-AutoSciLab) with the benchmark's own oracle and evaluation.

    python scripts/run_chem.py --autoscilab_dir /path/to/LLM-AutoSciLab --model gemini-3.1-flash-lite \
        --method epicompass --out results/chem

Tasks: the 27 mechanisms of the released manifest (configs/noise_studies/chembench_llm_autoscilab_noise27.json), law
version v0, at each difficulty; noise 0, grammar "universal". One JSON per task (existing tasks are skipped).
--hint adds a one-sentence clarification of the oracle's temperature / pH background effects to the mission (the same
text for every method).
"""
import argparse
import json
import os
import sys
import traceback
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def one(job):
    a, task = job
    tid = f"{task['domain']}__{task['difficulty']}__v0"
    out_dir = os.path.join(a.out, a.method, a.model.replace("/", "_"))
    path = os.path.join(out_dir, f"{tid}.json")
    if os.path.exists(path):
        return tid, "skip"
    os.makedirs(out_dir, exist_ok=True)
    try:
        from epicompass.adapters.chem import run_task
        r = run_task(task["domain"], task["difficulty"], "v0", a.model, method=a.method, hint=a.hint)
        with open(path + ".tmp", "w") as f:
            json.dump(r, f, default=str)
        os.replace(path + ".tmp", path)
        return tid, f"exact={r['exact']} rmsle={r['rmsle']}"
    except Exception:  # noqa: BLE001
        with open(os.path.join(out_dir, f"{tid}.err"), "w") as f:
            f.write(traceback.format_exc())
        return tid, "ERROR"


def init(as_dir):
    sys.path[:0] = [as_dir, ROOT]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--autoscilab_dir", default=os.environ.get("AUTOSCILAB_DIR"), required=not os.environ.get("AUTOSCILAB_DIR"))
    p.add_argument("--model", required=True)
    p.add_argument("--method", default=os.environ.get("EPIC_METHOD", "epicompass"),
                   choices=["epicompass", "react", "hsearch", "popper"])
    p.add_argument("--difficulties", nargs="*", default=["easy", "medium", "hard"])
    p.add_argument("--hint", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=os.path.join(ROOT, "results", "chem"))
    a = p.parse_args()
    a.autoscilab_dir, a.out = os.path.abspath(a.autoscilab_dir), os.path.abspath(a.out)
    os.environ["EPIC_METHOD"] = a.method
    manifest = json.load(open(os.path.join(a.autoscilab_dir, "configs", "noise_studies",
                                           "chembench_llm_autoscilab_noise27.json")))
    tasks = [{"domain": t["domain"], "difficulty": d} for d in a.difficulties for t in manifest]
    tasks = tasks[:a.limit] if a.limit else tasks
    print(f"{len(tasks)} tasks, method={a.method}, model={a.model}", flush=True)
    with Pool(a.workers, initializer=init, initargs=(a.autoscilab_dir,), maxtasksperchild=1) as pool:
        for tid, msg in pool.imap_unordered(one, [(a, t) for t in tasks]):
            print(tid, msg, flush=True)


if __name__ == "__main__":
    main()
