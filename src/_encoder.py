import abc
from typing import List, Union, Dict, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from src.utils import pool


class Encoder(abc.ABC):
    """
    Abstract class for dense encoders, providing methods to encode texts, queries, and corpora into dense vectors.
    """

    @abc.abstractmethod
    def encode_queries(
            self, queries: List[str], **kwargs
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encodes a list of queries into dense vector representations.

        Args:
            queries (`List[str]`):
                A list of query strings to encode.
            **kwargs:
                Additional arguments passed to the encoder.

        Returns:
            `Union[torch.Tensor, np.ndarray]`:
                Encoded queries as a tensor or numpy array.
        """
        raise NotImplementedError

    def encode_corpus(
            self,
            corpus: Union[
                List[Dict[Literal["title", "text"], str]],
                Dict[Literal["title", "text"], List],
            ],
            **kwargs
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encodes a list of corpus documents into dense vector representations.

        Args:
            corpus (`Union[List[Dict[Literal["title", "text"], str]], Dict[Literal["title", "text"], List]]`):
                A list or dictionary of corpus documents to encode.
            **kwargs:
                Additional arguments passed to the encoder.

        Returns:
            `Union[torch.Tensor, np.ndarray]`:
                Encoded corpus documents as a tensor or numpy array.
        """
        raise NotImplementedError


class TransformersEncoder(Encoder):

    def __init__(
            self,
            model_name_or_path: str,
            pool_type: Literal["avg", "weighted_avg", "cls", "last"] = "avg",
            query_prompt: Optional[
                str] = None,
            doc_prompt: Optional[str] = None,
            **kwargs
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            **kwargs
        ).to(torch.device("cuda"))

        self.pool_type = pool_type
        self.query_prompt = query_prompt
        self.doc_prompt = doc_prompt

    def encode_queries(
            self, queries: List[str], batch_size: int = 16, **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        if self.query_prompt is not None:
            if self.query_prompt is not None:
                queries = [self.query_prompt + query for query in queries]

        return self.encode(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(
            self,
            corpus: Union[
                List[Dict[Literal["title", "text"], str]],
                Dict[Literal["title", "text"], List],
            ],
            batch_size: int = 8,
            **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:

        if isinstance(corpus, dict):
            sentences = [
                (
                    (corpus["title"][i] + " " + corpus["text"][i]).strip()
                    if "title" in corpus
                    else corpus["text"][i].strip()
                )
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                (
                    (doc["title"] + " " + doc["text"]).strip()
                    if "title" in doc
                    else doc["text"].strip()
                )
                for doc in corpus
            ]
        if self.doc_prompt is not None:
            sentences = [self.doc_prompt + s for s in sentences]
        return self.encode(sentences, batch_size=batch_size, **kwargs)

    def encode(
            self,
            texts: List[str],
            batch_size: int,
            max_length: int = 512,
            padding: bool = True,
            truncation: bool = True,
            return_tensors: str = "pt"
    ):
        embeddings = []
        with torch.no_grad():
            for start_idx in tqdm(range(0, len(texts), batch_size)):
                batch_dict = self.tokenizer(
                    texts[start_idx: start_idx + batch_size],
                    max_length=max_length,
                    padding=padding,
                    truncation=truncation,
                    return_tensors=return_tensors
                )
                batch_dict = batch_dict.to(self.model.device)
                ctx_output = self.model(**batch_dict)
                embedding = pool(
                    last_hidden_state=ctx_output.last_hidden_state,
                    attention_mask=batch_dict['attention_mask'],
                    pool_type=self.pool_type
                ).detach()
                embeddings.append(embedding)

        embeddings = torch.cat(embeddings, dim=0)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings
