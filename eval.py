# eval.py

import argparse
import os
import yaml
import mteb

from evaluation.model_wrapper import (
    SentenceTransformerWrapper,
    TransformerCLSWrapper,
)

WRAPPERS = {
    "SentenceTransformer": SentenceTransformerWrapper,
    "TransformerCLS":      TransformerCLSWrapper,
}

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_wrapper(model_cfg):
    cls = WRAPPERS[model_cfg["wrapper"]]
    return cls(model_cfg["path"], device=model_cfg.get("device"))

def evaluate(config_path: str, models: list[str] | None = None):
    cfg = load_config(config_path)
    datasets = cfg["evaluation"].get("datasets", [cfg["evaluation"]["dataset"]])
    tasks = mteb.get_tasks(datasets)

    # 선택된 모델만 평가하거나, 전체 모델 평가
    model_cfgs = (
        [m for m in cfg["models"] if m["name"] in models]
        if models
        else cfg["models"]
    )

    for model_cfg in model_cfgs:
        name    = model_cfg["name"]
        wrapper = build_wrapper(model_cfg)
        evaluator = mteb.MTEB(tasks=tasks)
        out_dir = os.path.join(cfg["evaluation"]["output_root"], name)

        print(f"\n>> Evaluating {name} on {datasets}")
        evaluator.run(wrapper, output_folder=out_dir,
                      batch_size=cfg["evaluation"]["batch_size"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MSMARCO evaluation via MTEB"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="evaluation/config.yaml",
        help="Path to evaluation config file",
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="*",
        help="Names of models to evaluate (default: all)",
    )
    args = parser.parse_args()
    evaluate(args.config, args.models)
