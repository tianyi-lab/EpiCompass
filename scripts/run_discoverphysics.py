"""Run a method on DiscoverPhysics: the benchmark's own ScienceAgent/run_discovery.py, unchanged, with its
DiscoveryAgent replaced by EpiDiscoveryAgent and the epistemic-state record added to the run JSON.

    python scripts/run_discoverphysics.py --discoverphysics_dir /path/to/DiscoverPhysics --method epicompass \
        -- <the usual run_discovery.py arguments, e.g. --world gravity --model gemini/gemini-3.1-flash-lite ...>

Run it with DiscoverPhysics' own environment. The component's LLM calls use --epic_model (default: the agent's model
without its provider prefix) at EPIC_BASE_URL; the agent's own calls go through DiscoverPhysics' llm_client.
"""
import argparse
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--discoverphysics_dir", default=os.environ.get("DISCOVERPHYSICS_DIR"),
                   required=not os.environ.get("DISCOVERPHYSICS_DIR"))
    p.add_argument("--method", default=os.environ.get("EPIC_METHOD", "epicompass"),
                   choices=["epicompass", "react", "hsearch", "popper"])
    p.add_argument("--epic_model", default=None)
    a, rest = p.parse_known_args()
    rest = [x for x in rest if x != "--"]
    sa = os.path.join(os.path.abspath(a.discoverphysics_dir), "ScienceAgent")
    sys.path[:0] = [sa, ROOT]
    os.environ["EPIC_METHOD"] = a.method

    from scienceagent import agent as agent_mod
    from epicompass.adapters.discoverphysics import EpiDiscoveryAgent

    model = next((rest[i + 1] for i, x in enumerate(rest[:-1]) if x == "--model"), "")
    epic_model = a.epic_model or (model.split("/", 1)[1] if "/" in model else model)

    class Agent(EpiDiscoveryAgent):
        def __init__(self, *args, **kw):
            super().__init__(*args, epic_model=epic_model, method=a.method, **kw)

    if a.method != "react":
        agent_mod.DiscoveryAgent = Agent                 # run_discovery imports it inside main()

    spec = importlib.util.spec_from_file_location("run_discovery", os.path.join(sa, "run_discovery.py"))
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)
    write = rd._write_run_json

    def write_with_record(path, world, model, timestamp, agent, *args, **kw):
        write(path, world, model, timestamp, agent, *args, **kw)
        if hasattr(agent, "epic_record"):
            import json
            with open(path) as f:
                data = json.load(f)
            data.update(agent.epic_record())
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
    rd._write_run_json = write_with_record

    sys.argv = [os.path.join(sa, "run_discovery.py")] + rest
    rd.main()


if __name__ == "__main__":
    main()
