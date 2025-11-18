import argparse
import sys
import runpy
import os


def main():
    parser = argparse.ArgumentParser(description="Run simulation modules")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--runner", action="store_true", help="Run simulation_runner")
    group.add_argument("--collect", action="store_true", help="Run data_collector")

    args, remaining = parser.parse_known_args()

    if args.runner:
        cmd = "simulation_runner"
    elif args.collect:
        cmd = "collector_runner"
    else:
        parser.error("Specify --runner / --collect")

    module_map = {
        "simulation_runner": "src.simulation_runner",
        "collector_runner": "src.collector_runner",
    }

    if cmd not in module_map:
        parser.error(f"Unknown command: {cmd}")

    module_name = module_map[cmd]

    repo_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    old_argv = sys.argv
    sys.argv = [module_name] + remaining
    try:
        runpy.run_module(module_name, run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()