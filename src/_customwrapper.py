from typing import Sequence, Any

import numpy as np
from mteb import SentenceTransformerWrapper
from mteb.encoder_interface import PromptType


class CustomSTWrapper(SentenceTransformerWrapper):
    
    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        
        new_kwargs = {
            **kwargs,
            "convert_to_numpy": False,
            "convert_to_tensor": True,
            "batch_size": 256,
        }
        
        return super().encode(sentences, task_name=task_name, prompt_type=prompt_type, **new_kwargs)