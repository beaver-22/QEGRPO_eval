import logging
from typing import Any

from mteb import HFSubset, ScoresDict, RetrievalEvaluator
from mteb.abstasks import AbsTaskRetrieval
from tqdm import tqdm
from transformers import PreTrainedModel, GenerationMixin

from src._chatmodel import BaseChatModel

logger = logging.getLogger(__name__)


# adopted from "https://github.com/embeddings-benchmark/mteb/blob/31173dc0915baf29235ad79f00f643ecafd993df/mteb/abstasks/AbsTaskRetrieval.py#L252"
class AbsTaskRetrievalWithQE(AbsTaskRetrieval):

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
            
            if not isinstance(expansion_model, BaseChatModel):
                raise TypeError("expansion_model must be `BaseChatModel`")

            queries_with_expansion = {
                qid: text + " " + expansion_model.generate_single_turn_response(text)
                for qid, text in tqdm(queries.items(), desc="Expanding queries", leave=False)
            }
            
            # END
            
            scores[hf_subset] = self._evaluate_subset(
                retriever, corpus, queries_with_expansion, relevant_docs, hf_subset, **kwargs
            )
        return scores
