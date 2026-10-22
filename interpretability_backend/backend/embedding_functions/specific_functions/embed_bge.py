import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings
from FlagEmbedding import BGEM3FlagModel

from ...utils.truncation import TruncationTracker


class EmbedTextBGE(EmbeddingFunction[Documents]):
    """BGE-M3 dense vectors on ``device``, ``batch_size`` texts per forward pass.

    Texts longer than ``max_length`` tokens are truncated and counted in
    ``truncation`` (the job reports them as a warning).
    """

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        device: str = "mps",
        use_fp16: bool = True,
        max_length: int = 8192,
        batch_size: int = 12,
        pooling_method: str = "cls",
    ) -> None:

        # FlagEmbedding takes `devices` (a `device` kwarg is silently ignored and
        # the library auto-picks every CUDA GPU, else MPS, else CPU).
        self.model = BGEM3FlagModel(
            model, devices=device, pooling_method=pooling_method, use_fp16=use_fp16
        )
        self.max_length = max_length
        self.batch_size = batch_size
        self.truncation = TruncationTracker(max_length)

    def __call__(self, input: Documents) -> Embeddings:
        texts = list(input)
        self.truncation.count(getattr(self.model, "tokenizer", None), texts)

        embeddings = np.array(
            self.model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
            )["dense_vecs"]
        )

        # embed the documents somehow
        ### Embedidng type wants a list of np.ndarray
        ### it returns a np array containing that contains other arrays
        ### we explictely convert to np.array first for the type checker
        return list(embeddings)
