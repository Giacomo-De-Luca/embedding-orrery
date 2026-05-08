import importlib

import numpy as np
import numpy.typing as npt
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

## Unused, - an example template found in the ChromaDB documentation


class TransformerEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(
        self,
        model_name: str = "dbmdz/bert-base-turkish-cased",
        cache_dir: str | None = None,
    ):
        try:
            from transformers import AutoModel, AutoTokenizer

            self._torch = importlib.import_module("torch")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        except ImportError:
            raise ValueError(
                "The transformers and/or pytorch python package is not installed. Please install it with "
                "`pip install transformers` or `pip install torch`"
            )

    @staticmethod
    def _normalize(vector: npt.NDArray) -> npt.NDArray:
        """Normalizes a vector to unit length using L2 norm."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def __call__(self, input: Documents) -> Embeddings:
        inputs = self._tokenizer(input, padding=True, truncation=True, return_tensors="pt")
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pooling
        return [e.tolist() for e in self._normalize(embeddings)]
