from __future__ import annotations

import argparse

from src.utils.config import add_common_args, apply_overrides, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-out ability mask checks. Placeholder for the full MVP.")
    add_common_args(parser)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="YAML override key=value.")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    print("[zeroout_check] placeholder ready")
    print(f"[zeroout_check] results_dir={cfg['eval']['results_dir']}")


if __name__ == "__main__":
    main()
