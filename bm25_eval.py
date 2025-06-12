from mteb import MTEB, get_model
from typing import List
import logging
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.AbsTaskRetrieval import HFDataLoader
from mteb.abstasks.TaskMetadata import TaskMetadata
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

class AdapterChatModel:
    def __init__(self, base_model_name="Qwen/Qwen3-1.7B", adapter_path="Chaew00n/QEGRPO-Qwen3-1.7B-BM25-temp1.2-lr1e-6", device="cuda:0"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        peft_model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model = peft_model.merge_and_unload()
        self.model = self.model.to(device)

    def generate_single_turn_response(self, user_input: List[str], batch_size: int = 128):
        all_responses = []
        for start_idx in tqdm(range(0, len(user_input), batch_size), desc="Generating responses"):
            texts = user_input[start_idx: start_idx + batch_size]
            prompt_inputs = self.tokenizer(
                    text=texts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
                ).to(self.device)
            prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
            prompt_completion_ids = self.model.generate(
                prompt_ids, attention_mask=prompt_mask, max_new_tokens=64
            )
            responses = self.tokenizer.batch_decode(prompt_completion_ids, skip_special_tokens=True)
            all_responses.extend(responses)
        return all_responses


class MSMARCO(AbsTaskRetrieval):
    ignore_identical_ids = True
    print("My Task Called!")

    metadata = TaskMetadata(
        name="MSMARCO",
        dataset={
            "path": "mteb/msmarco",
            "revision": "c5a29a104738b98a9e76336939199e264163d4a0",
        },
        description="MS MARCO is a collection of datasets focused on deep learning in search",
        reference="https://microsoft.github.io/msmarco/",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=[
            "Encyclopaedic",
            "Academic",
            "Blog",
            "News",
            "Medical",
            "Government",
            "Reviews",
            "Non-fiction",
            "Social",
            "Web",
        ],
        task_subtypes=["Question answering"],
        license="msr-la-nc",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{DBLP:journals/corr/NguyenRSGTMD16,
  archiveprefix = {arXiv},
  author = {Tri Nguyen and
Mir Rosenberg and
Xia Song and
Jianfeng Gao and
Saurabh Tiwary and
Rangan Majumder and
Li Deng},
  bibsource = {dblp computer science bibliography, https://dblp.org},
  biburl = {https://dblp.org/rec/journals/corr/NguyenRSGTMD16.bib},
  eprint = {1611.09268},
  journal = {CoRR},
  timestamp = {Mon, 13 Aug 2018 16:49:03 +0200},
  title = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
  url = {http://arxiv.org/abs/1611.09268},
  volume = {abs/1611.09268},
  year = {2016},
}
""",
        prompt={
            "query": "Given a web search query, retrieve relevant passages that answer the query"
        },
    )
    
    def load_data(self, **kwargs):
        print("My Task Getting Data!")
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        dataset_path = self.metadata_dict["dataset"]["path"]
        hf_repo_qrels = (
            dataset_path + "-qrels" if "clarin-knext" in dataset_path else None
        )
        for split in kwargs.get("eval_splits", self.metadata_dict["eval_splits"]):
            corpus, queries, qrels = HFDataLoader(
                hf_repo=dataset_path,
                hf_repo_qrels=hf_repo_qrels,
                streaming=False,
                keep_in_memory=False,
                trust_remote_code=self.metadata_dict["dataset"].get(
                    "trust_remote_code", False
                ),
            ).load(split=split)
            # Conversion from DataSet
            queries = {
                query["id"]: query["text"]
                for query in tqdm(queries, desc="Getting queries")
            }
            qid_list = list(queries.keys())
            text_list = list(queries.values())
            adapter_model = AdapterChatModel()
            expansions = adapter_model.generate_single_turn_response(
                [
                    """
        Write a factual and informative paragraph that provides relevant background information and detailed explanation to address the following question.
        Query: {text}
        Passage:
                    """.format(text=text) for text in text_list
                ]
            )
            queries_with_expansion = {
                qid: expansion
                for qid, expansion in zip(qid_list, expansions)
            }
            corpus = {
                doc["id"]: doc.get("title", "") + " " + doc["text"] for doc in corpus
            }
            self.corpus[split], self.queries[split], self.relevant_docs[split] = (
                corpus,
                queries_with_expansion,
                qrels,
            )

        self.data_loaded = True
# 2. 평가 실행
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("main")

    TASK_LIST_RETRIEVAL = [
        "MSMARCO"
    ]
    model = get_model("bm25s")  # 또는 SentenceTransformer, e5-base 등
    evaluation = MTEB(tasks=TASK_LIST_RETRIEVAL, task_langs=["en"])
    evaluation.run(model, output_folder="results")
