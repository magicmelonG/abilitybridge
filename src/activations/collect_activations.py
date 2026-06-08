from __future__ import annotations

import argparse

from src.utils.config import add_common_args, apply_overrides, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect selected-layer selected-token hidden states. Placeholder for the full MVP.")
    add_common_args(parser)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="YAML override key=value.")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    print("[collect_activations] placeholder ready")
    print(f"[collect_activations] output_dir={cfg['activations']['output_dir']}")
    print(f"[collect_activations] teacher_layers={cfg['activations']['teacher_layers']} student_layers={cfg['activations']['student_layers']}")


if __name__ == "__main__":
    main()
