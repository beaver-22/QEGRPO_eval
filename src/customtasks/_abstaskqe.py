import logging
from typing import Any
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb import HFSubset, ScoresDict, RetrievalEvaluator
from mteb.abstasks import AbsTaskRetrieval
from tqdm import tqdm
from transformers import PreTrainedModel, GenerationMixin

from src._chatmodel import BaseChatModel

logger = logging.getLogger(__name__)


# adopted from "https://github.com/embeddings-benchmark/mteb/blob/31173dc0915baf29235ad79f00f643ecafd993df/mteb/abstasks/AbsTaskRetrieval.py#L252"
class AbsTaskRetrievalWithQE(AbsTaskRetrieval):

    metadata = TaskMetadata(
        name="MSMARCO-with-QE",
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

    def evaluate(
            self,
            retrieval_model,
            expansion_model: BaseChatModel,
            split: str = "test",
            subsets_to_run: list[HFSubset] | None = None,
            *,
            encode_kwargs: dict[str, Any] = {},
            **kwargs,
    ) -> dict[HFSubset, ScoresDict]:
        
        retriever = RetrievalEvaluator(
            retriever=retrieval_model,
            task_name=self.metadata.name,
            encode_kwargs=encode_kwargs,
            **kwargs,
        )

        scores = {}
        hf_subsets = list(self.hf_subsets) if self.is_multilingual else ["default"]
        if subsets_to_run is not None:
            hf_subsets = [s for s in hf_subsets if s in subsets_to_run]

        for hf_subset in hf_subsets:
            logger.info(f"Subset: {hf_subset}")

            if hf_subset == "default":
                corpus, queries, relevant_docs = (
                    self.corpus[split],
                    self.queries[split],
                    self.relevant_docs[split],
                )
            else:
                corpus, queries, relevant_docs = (
                    self.corpus[hf_subset][split],
                    self.queries[hf_subset][split],
                    self.relevant_docs[hf_subset][split],
                )
            
            # TODO: review the query expansion process in queries
            # START
        
            
            queries_with_expansion = {
                qid: text + " " + expansion_model.generate_single_turn_response(
                    user_input=f"""
                    Write a factual and informative paragraph that provides relevant background information and detailed explanation to address the following question.
                    Query: {text}
                    Passage:
                    """
                )
                for i, (qid, text) in enumerate(tqdm(queries.items(), desc="Expanding queries", leave=False))
                if i < 100  # i는 0부터 시작하므로 600개까지
            }

            scores[hf_subset] = self._evaluate_subset(
                retriever, corpus, queries_with_expansion, relevant_docs, hf_subset, **kwargs
            )
        return scores
