import os

import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()


class EmbedTextGemini(EmbeddingFunction[Documents]):
    def __init__(
        self, model: str = "gemini-embedding-001", task_type: str = "SEMANTIC_SIMILARITY"
    ) -> None:
        self.model = model
        self.task_type = task_type
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.GEMINI_API_KEY)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    def __call__(self, input: Documents) -> Embeddings:

        response = self.client.models.embed_content(
            model=self.model,
            contents=list(input),
            config=types.EmbedContentConfig(task_type=self.task_type),
        )
        # embed the documents somehow
        ### Embedidng type wants a list of np.ndarray
        ### embedding.embeddings is a list of Embedding objects
        ### made using a pydantic field
        return (
            ([np.array(e.values) for e in response.embeddings])
            if response.embeddings is not None
            else []
        )
