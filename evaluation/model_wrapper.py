import numpy as np
import torch
from sentence_transformers import SentenceTransformer as _STWrapper
from transformers import AutoModel, AutoTokenizer


class BaseWrapper:
    """
    Abstract base class for any embedding-model wrapper.
    Must implement .encode(List[str]) → np.ndarray.
    """
    def encode(self, sentences, batch_size: int = 64, **kwargs) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerWrapper(BaseWrapper):
    """
    Wrapper for HuggingFace Sentence-Transformers models.
    """
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = _STWrapper(model_name).to(device)
        self.device = device

    def encode(self, sentences, batch_size: int = 64, **kwargs) -> np.ndarray:
        return self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            **kwargs
        )


class TransformerCLSWrapper(BaseWrapper):
    """
    Wrapper for generic Transformer-based models (e.g., BERT, fine-tuned variants).
    Uses [CLS] token embedding from last_hidden_state as sentence embedding.
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = AutoModel.from_pretrained(model_path).to(device)
        self.device    = device

    def encode(self, sentences, batch_size: int = 64, **kwargs) -> np.ndarray:
        embs = []
        max_length = kwargs.get("max_length", 256)

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            toks = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            with torch.no_grad():
                out = self.model(**toks)
                cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy()

            embs.append(cls_emb)

        return np.vstack(embs)
