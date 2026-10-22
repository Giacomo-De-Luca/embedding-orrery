import os

import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()


class EmbedTextGemini(EmbeddingFunction[Documents]):
    """Gemini embeddings, one vector per input text.

    Each text is sent as its own ``Content``: google-genai merges a plain list
    of strings into one Content for gemini-embedding-2, which the API embeds as
    a single aggregated vector. ``task_type`` is the gemini-embedding-001 task
    field (gemini-embedding-2 has none and reads its task from a text prefix);
    ``text_prefix`` is prepended to every text (query prefixes).
    """

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        task_type: str | None = None,
        text_prefix: str | None = None,
        client=None,
    ) -> None:
        self.model = model
        self.task_type = task_type
        self.text_prefix = text_prefix or ""
        self.client = client or genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def __call__(self, input: Documents) -> Embeddings:
        texts = [self.text_prefix + text for text in input]
        if not texts:
            return []
        contents = [types.Content(parts=[types.Part.from_text(text=text)]) for text in texts]
        embeddings = self._embed(contents).embeddings or []
        # Outside the retry: a short response is not transient, and vectors
        # must never be assigned to the wrong documents.
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Gemini returned {len(embeddings)} embeddings for {len(texts)} inputs"
            )
        return [np.array(e.values, dtype=np.float32) for e in embeddings]

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _embed(self, contents: list[types.Content]):
        config = types.EmbedContentConfig(task_type=self.task_type) if self.task_type else None
        return self.client.models.embed_content(model=self.model, contents=contents, config=config)
