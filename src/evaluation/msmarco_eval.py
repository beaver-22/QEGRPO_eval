import os
import yaml
import mteb

from .model_wrapper import (
    SentenceTransformerWrapper,
    TransformerCLSWrapper,
)

# config.yaml 의 wrapper 필드와 일치시켜야 함.
WRAPPERS = {
    "SentenceTransformer": SentenceTransformerWrapper,
    "TransformerCLS":      TransformerCLSWrapper,
}

def load_config(path=None):
    """
    Load evaluation configuration from YAML file.
    """
    path = path or os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_wrapper(model_cfg):
    """
    Instantiate the correct wrapper for a model based on config.
    """
    wrapper_name = model_cfg["wrapper"]
    if wrapper_name not in WRAPPERS:
        raise ValueError(f"Unknown wrapper: {wrapper_name}")
    cls = WRAPPERS[wrapper_name]
    return cls(model_cfg["path"], device=model_cfg.get("device", None))

def evaluate_all():
    """
    Run retrieval evaluation for all models listed in config.yaml.
    Logs a comprehensive set of metrics.
    """
    # 1) Load overall config
    cfg = load_config()

    # 2) Prepare the MTEB tasks
    datasets = cfg["evaluation"].get("datasets", [cfg["evaluation"]["dataset"]])
    tasks = mteb.get_tasks(datasets)

    # 3) Iterate over each model configuration
    for model_cfg in cfg["models"]:
        model_name = model_cfg["name"]
        print(f"\n>>> Evaluating model: {model_name} on {datasets}")

        # 4) Build wrapper and MTEB evaluator
        wrapper = build_wrapper(model_cfg)
        evaluator = mteb.MTEB(tasks=tasks)

        # 5) Run evaluation; results go to output_root/model_name/…
        output_dir = os.path.join(cfg["evaluation"]["output_root"], model_name)
        evaluator.run(
            wrapper,
            output_folder=output_dir,
            batch_size=cfg["evaluation"]["batch_size"],
        )

        # 6) Load and print a comprehensive set of metrics for each dataset
        for ds in datasets:
            result_file = os.path.join(output_dir, ds, "retrieval.json")
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
    evaluate_all()
