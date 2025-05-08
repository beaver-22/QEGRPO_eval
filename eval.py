# eval.py

import argparse
import os
import yaml
import mteb
from src._mtebwithqe import MTEBWithQE
from src._chatmodel import BaseChatModel, AdapterChatModel
from src.customtasks.msmarcowithqe import MSMARCOWithQE
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
    # ToDo: We need to get custom task (i.e. MSMARCOWithQE)
    datasets = cfg["evaluation"]["datasets"]
    tasks = [MSMARCOWithQE]

    retrieval_model_name = cfg["retrieval_model"]["name"]
    retrieval_model_wrapper = build_wrapper(cfg["retrieval_model"])

    # 선택된 모델만 평가하거나, 전체 모델 평가
    model_cfgs = (
        [m for m in cfg["expansion_models"] if m["name"] in models]
        if models
        else cfg["expansion_models"]
    )
    
    # Retrieval Model Along With Chat Model
    for model_cfg in model_cfgs:
        name    = model_cfg["name"]
        evaluator = MTEBWithQE(tasks=tasks)
        out_dir = os.path.join(cfg["evaluation"]["output_root"], name)
        print(f"\n>> Evaluating {name} on {datasets}")
        if model_cfg["adapted"]:
            expand_model = AdapterChatModel(adapter_path = model_cfg["path"], device=model_cfg.get("device"))
            evaluator.run_with_qe(retrieval_model = retrieval_model_wrapper, expansion_model=expand_model, output_folder=out_dir,
                      batch_size=cfg["evaluation"]["batch_size"])
        else:
            evaluator.run_with_qe(retrieval_model = retrieval_model_wrapper, expansion_model=BaseChatModel, output_folder=out_dir,
                      batch_size=cfg["evaluation"]["batch_size"])
        

        # 6) Load and print a comprehensive set of metrics for each dataset
        for ds in datasets:
            result_file = os.path.join(out_dir, ds, "retrieval.json")
            print(f"\nResults for dataset: {ds}")
            if os.path.exists(result_file):
                metrics = yaml.safe_load(open(result_file, "r"))

                # Default Metrics
                # 아래 모든 평가지표는 mteb의 MSMARCORetrieval.py 파일에 보면 MSMARCO class <-  AbsTaskRetrieval class에 정의되어 있음. 
                ndcg5 = metrics.get("ndcg@5")             
                ndcg10 = metrics.get("ndcg@10")
                recall1 = metrics.get("recall@1")
                recall5 = metrics.get("recall@5")
                recall10 = metrics.get("recall@10")
                recall20 = metrics.get("recall@20")       
                mrr10 = metrics.get("mrr@10")             
                precision10 = metrics.get("precision@10") 
                map10 = metrics.get("map@10")            

                # Print summary
                print(f"  nDCG@5     : {ndcg5:.4f}"      if ndcg5       is not None else "  nDCG@5     : N/A")
                print(f"  nDCG@10    : {ndcg10:.4f}"     if ndcg10      is not None else "  nDCG@10    : N/A")
                print(f"  Recall@1   : {recall1:.4f}"    if recall1     is not None else "  Recall@1   : N/A")
                print(f"  Recall@5   : {recall5:.4f}"    if recall5     is not None else "  Recall@5   : N/A")
                print(f"  Recall@10  : {recall10:.4f}"   if recall10    is not None else "  Recall@10  : N/A")
                print(f"  Recall@20  : {recall20:.4f}"   if recall20    is not None else "  Recall@20  : N/A")
                print(f"  MRR@10     : {mrr10:.4f}"      if mrr10       is not None else "  MRR@10     : N/A")
                print(f"  Precision@10: {precision10:.4f}" if precision10 is not None else "  Precision@10: N/A")
                print(f"  MAP@10     : {map10:.4f}"      if map10       is not None else "  MAP@10     : N/A")
            else:
                print(f"  ⚠️ Result file not found: {result_file}")
        
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
