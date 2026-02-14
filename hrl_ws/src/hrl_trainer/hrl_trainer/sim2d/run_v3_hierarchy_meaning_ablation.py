from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .train_rl_brainer_v3_online import run_v3_hierarchy_meaning_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V3 hierarchy-meaning tiered ablation")
    parser.add_argument(
        "--config",
        default="src/hrl_trainer/config/train_rl_brainer_v3_hierarchy_meaning_ablation.yaml",
        help="Path to hierarchy-meaning ablation config YAML",
    )
    parser.add_argument("--out", default="", help="Optional output JSON path")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out = run_v3_hierarchy_meaning_ablation(cfg)
    print(json.dumps(out, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
