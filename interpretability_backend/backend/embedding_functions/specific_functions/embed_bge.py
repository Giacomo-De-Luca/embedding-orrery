import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings
from FlagEmbedding import BGEM3FlagModel


class EmbedTextBGE(EmbeddingFunction[Documents]):
    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        device: str = "mps",
        use_fp16: bool = True,
        max_length: int = 8192,
        batch_size: int = 12,
        pooling_method: str = "cls",
    ) -> None:

        self.model = BGEM3FlagModel(
            model, device=device, pooling_method=pooling_method, use_fp16=use_fp16
        )
        self.max_length = max_length
        self.batch_size = batch_size

    def __call__(self, input: Documents) -> Embeddings:

        embeddings = np.array(
            self.model.encode(
                input,
                batch_size=self.batch_size,
                max_length=self.max_length,
            )["dense_vecs"]
        )

        # embed the documents somehow
        ### Embedidng type wants a list of np.ndarray
        ### it returns a np array containing that contains other arrays
        ### we explictely convert to np.array first for the type checker
        return list(embeddings)
