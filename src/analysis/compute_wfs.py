from __future__ import annotations

import argparse

from src.utils.config import add_common_args, apply_overrides, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ability-aware WFS scores. Placeholder for the full MVP.")
    add_common_args(parser)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="YAML override key=value.")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    print("[compute_wfs] placeholder ready")
    print(f"[compute_wfs] output_dir={cfg['ot']['output_dir']}")


if __name__ == "__main__":
    main()
