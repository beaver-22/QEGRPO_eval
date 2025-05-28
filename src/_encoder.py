import abc
from typing import List, Literal, Optional

import torch
import torch.nn.functional as F
from mteb.encoder_interface import PromptType
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from src.utils import pool


class Encoder(abc.ABC):
    """
    Abstract class for dense encoders, providing methods to encode texts, queries, and corpora into dense vectors.
    """

    @abc.abstractmethod
    def encode(
            self,
            texts: List[str],
            batch_size: int,
            task_name: str,
            prompt_type: PromptType,
            max_length: int = 512,
            padding: bool = True,
            truncation: bool = True,
            return_tensors: str = "pt",
            **kwargs
    ):
        raise NotImplementedError


class TransformersEncoder(Encoder):

    def __init__(
            self,
            model_name_or_path: str,
            pool_type: Literal["avg", "weighted_avg", "cls", "last"] = "avg",
            query_prompt: Optional[
                str] = None,
            doc_prompt: Optional[str] = None,
            device: Optional[str] = "cuda",
            **kwargs
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            **kwargs
        ).to(torch.device(device))

        self.pool_type = pool_type
        self.query_prompt = query_prompt
        self.doc_prompt = doc_prompt

    def encode(
            self,
            texts: List[str],
            batch_size: int,
            task_name: str,
            prompt_type: PromptType,
            max_length: int = 512,
            padding: bool = True,
            truncation: bool = True,
            return_tensors: str = "pt",
            **kwargs
    ):
        task_name = task_name.lower()
        if prompt_type == PromptType.passage:
            texts = [self.doc_prompt + s for s in texts]
        elif prompt_type == PromptType.query:
            texts = [self.query_prompt + s for s in texts]

        embeddings = []
        with torch.no_grad():
            for start_idx in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
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
