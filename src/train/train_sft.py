from __future__ import annotations

import argparse

from src.utils.config import add_common_args, apply_overrides, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ordinary SFT baseline. Placeholder for the full MVP.")
    add_common_args(parser)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="YAML override key=value.")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    print("[train_sft] placeholder ready")
    print(f"[train_sft] output_dir={cfg['train']['output_dir']} logs_dir={cfg['train']['logs_dir']}")


if __name__ == "__main__":
    main()
