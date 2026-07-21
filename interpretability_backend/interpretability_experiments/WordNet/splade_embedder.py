"""
SPLADE Sparse Embedder with Word-Level Extraction

This module provides SPLADE (Sparse Lexical and Expansion) embeddings with support for:
1. Sentence-level embeddings (standard SPLADE)
2. Word-level embeddings (extract sparse vector at target word's token position)

SPLADE produces 30,522-dimensional sparse vectors where each dimension corresponds
to a BERT vocabulary token, making dimensions directly interpretable.
"""

from dataclasses import dataclass

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


@dataclass
class SparseEmbedding:
    """Sparse embedding representation with indices and values."""

    indices: list[int]
    values: list[float]

    def to_dict(self) -> dict[str, list]:
        return {"indices": self.indices, "values": self.values}

    @classmethod
    def from_dict(cls, d: dict) -> "SparseEmbedding":
        return cls(indices=d["indices"], values=d["values"])

    def top_k(self, k: int = 50) -> list[tuple[int, float]]:
        """Return top k (index, value) pairs sorted by value."""
        pairs = list(zip(self.indices, self.values, strict=True))
        return sorted(pairs, key=lambda x: x[1], reverse=True)[:k]


class SPLADEEmbedder:
    """
    SPLADE embedder supporting both sentence-level and word-level extraction.

    SPLADE uses BERT's MLM head to produce sparse vectors where each dimension
    corresponds to a vocabulary token. This enables interpretable embeddings
    where we know exactly what each dimension represents.
    """

    def __init__(
        self, model_name: str = "naver/splade-cocondenser-ensembledistil", device: str | None = None
    ):
        """
        Initialize the SPLADE embedder.

        Args:
            model_name: HuggingFace model ID for SPLADE
            device: Device to use (auto-detected if None)
        """
        self.model_name = model_name

        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Loading SPLADE model: {model_name}")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.vocab_size = self.tokenizer.vocab_size
        print(f"Vocabulary size: {self.vocab_size}")

    def _compute_sparse_vector(
        self, logits: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply SPLADE transformation: log saturation + ReLU + max pooling.

        Args:
            logits: Raw MLM logits [batch, seq_len, vocab_size]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Sparse vector [batch, vocab_size]
        """
        # Log saturation with ReLU: log(1 + ReLU(x))
        sparse = torch.log(1 + torch.relu(logits))

        # Mask out padding tokens
        sparse = sparse * attention_mask.unsqueeze(-1)

        # Max pooling across sequence
        sparse, _ = torch.max(sparse, dim=1)

        return sparse

    def _to_sparse_embedding(self, vector: torch.Tensor, threshold: float = 0.0) -> SparseEmbedding:
        """
        Convert dense vector to sparse representation.

        Args:
            vector: Dense vector [vocab_size]
            threshold: Minimum value to include in sparse representation

        Returns:
            SparseEmbedding with non-zero indices and values
        """
        vector = vector.cpu()

        # Get non-zero indices
        mask = vector > threshold
        indices = torch.nonzero(mask, as_tuple=True)[0].tolist()
        values = vector[mask].tolist()

        return SparseEmbedding(indices=indices, values=values)

    def embed_sentence(self, text: str, threshold: float = 0.0) -> SparseEmbedding:
        """
        Standard SPLADE embedding: max-pool across all token positions.

        Args:
            text: Input text to embed
            threshold: Minimum value for sparse representation

        Returns:
            SparseEmbedding for the entire sentence
        """
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(
            self.device
        )

        with torch.no_grad():
            output = self.model(**tokens)
            logits = output.logits  # [1, seq_len, vocab_size]

            sparse = self._compute_sparse_vector(logits, tokens.attention_mask)

        return self._to_sparse_embedding(sparse.squeeze(0), threshold)

    def embed_sentences_batch(
        self, texts: list[str], batch_size: int = 32, threshold: float = 0.0
    ) -> list[SparseEmbedding]:
        """
        Batch embed multiple sentences.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch
            threshold: Minimum value for sparse representation

        Returns:
            List of SparseEmbeddings
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            tokens = self.tokenizer(
                batch_texts, return_tensors="pt", truncation=True, max_length=512, padding=True
            ).to(self.device)

            with torch.no_grad():
                output = self.model(**tokens)
                logits = output.logits

                sparse = self._compute_sparse_vector(logits, tokens.attention_mask)

            for j in range(sparse.size(0)):
                embeddings.append(self._to_sparse_embedding(sparse[j], threshold))

        return embeddings

    def _find_word_positions(self, input_ids: torch.Tensor, word_token_ids: list[int]) -> list[int]:
        """
        Find the positions of a word's tokens in the input sequence.

        Args:
            input_ids: Tokenized input [seq_len]
            word_token_ids: Token IDs for the target word

        Returns:
            List of positions where the word tokens appear
        """
        input_ids = input_ids.tolist()
        word_len = len(word_token_ids)

        positions = []
        for i in range(len(input_ids) - word_len + 1):
            if input_ids[i : i + word_len] == word_token_ids:
                positions.extend(range(i, i + word_len))
                break  # Use first occurrence

        return positions

    def embed_word_in_context(
        self, sentence: str, target_word: str, threshold: float = 0.0
    ) -> SparseEmbedding | None:
        """
        Extract sparse vector at target word's token position(s).

        This gives a word-level embedding that captures the word's meaning
        in the context of the sentence, without pooling over other tokens.

        Args:
            sentence: Full sentence containing the target word
            target_word: The word to extract embedding for
            threshold: Minimum value for sparse representation

        Returns:
            SparseEmbedding for the target word, or None if word not found
        """
        # Tokenize sentence
        tokens = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(
            self.device
        )

        # Get token IDs for the target word (without special tokens)
        word_tokens = self.tokenizer(target_word, add_special_tokens=False)["input_ids"]

        # Find word positions in sentence
        positions = self._find_word_positions(tokens["input_ids"][0], word_tokens)

        if not positions:
            # Word not found as exact token match, try lowercase
            word_tokens_lower = self.tokenizer(target_word.lower(), add_special_tokens=False)[
                "input_ids"
            ]
            positions = self._find_word_positions(tokens["input_ids"][0], word_tokens_lower)

        if not positions:
            return None

        with torch.no_grad():
            output = self.model(**tokens)
            logits = output.logits  # [1, seq_len, vocab_size]

            # Extract logits at target word positions
            word_logits = logits[0, positions, :]  # [num_positions, vocab_size]

            # Apply SPLADE transformation
            sparse = torch.log(1 + torch.relu(word_logits))

            # Pool if multi-token word (max pooling across token positions)
            if len(positions) > 1:
                sparse, _ = torch.max(sparse, dim=0)
            else:
                sparse = sparse.squeeze(0)

        return self._to_sparse_embedding(sparse, threshold)

    def decode_tokens(
        self, sparse_emb: SparseEmbedding, top_k: int = 50
    ) -> list[tuple[str, float]]:
        """
        Convert sparse indices to human-readable tokens.

        Args:
            sparse_emb: Sparse embedding to decode
            top_k: Number of top tokens to return

        Returns:
            List of (token_string, weight) tuples
        """
        top_items = sparse_emb.top_k(top_k)
        return [(self.tokenizer.decode([idx]).strip(), weight) for idx, weight in top_items]

    def token_id_to_string(self, token_id: int) -> str:
        """Convert a token ID to its string representation."""
        return self.tokenizer.decode([token_id]).strip()

    def string_to_token_id(self, token_str: str) -> int | None:
        """Convert a token string to its ID (if exact match exists)."""
        tokens = self.tokenizer.encode(token_str, add_special_tokens=False)
        if len(tokens) == 1:
            return tokens[0]
        return None


def test_splade_embedder():
    """Test the SPLADE embedder with example usage."""
    print("=" * 60)
    print("Testing SPLADE Embedder")
    print("=" * 60)

    embedder = SPLADEEmbedder()

    # Test 1: Sentence-level embedding
    print("\n1. Sentence-level embedding:")
    text = "cat: a small feline mammal often kept as a pet"
    emb = embedder.embed_sentence(text)
    print(f"   Input: {text}")
    print(f"   Non-zero dimensions: {len(emb.indices)}")
    print(f"   Top tokens: {embedder.decode_tokens(emb, top_k=10)}")

    # Test 2: Word-level embedding from context
    print("\n2. Word-level embedding (word in context):")
    sentence = "The cat sat on the mat."
    word = "cat"
    emb = embedder.embed_word_in_context(sentence, word)
    if emb:
        print(f"   Sentence: {sentence}")
        print(f"   Target word: {word}")
        print(f"   Non-zero dimensions: {len(emb.indices)}")
        print(f"   Top tokens: {embedder.decode_tokens(emb, top_k=10)}")
    else:
        print(f"   Word '{word}' not found in sentence")

    # Test 3: Multi-token word
    print("\n3. Multi-token word embedding:")
    sentence = "I saw a bobcat in the forest."
    word = "bobcat"
    emb = embedder.embed_word_in_context(sentence, word)
    if emb:
        print(f"   Sentence: {sentence}")
        print(f"   Target word: {word}")
        print(f"   Non-zero dimensions: {len(emb.indices)}")
        print(f"   Top tokens: {embedder.decode_tokens(emb, top_k=10)}")
    else:
        print(f"   Word '{word}' not found in sentence")

    # Test 4: Compare sentence vs word-level
    print("\n4. Comparison: Sentence vs Word-level:")
    sentence = "The bank is located by the river."
    word = "bank"

    sentence_emb = embedder.embed_sentence(sentence)
    word_emb = embedder.embed_word_in_context(sentence, word)

    print(f"   Sentence: {sentence}")
    print(f"   Sentence-level top tokens: {embedder.decode_tokens(sentence_emb, top_k=5)}")
    if word_emb:
        print(f"   Word-level ('{word}') top tokens: {embedder.decode_tokens(word_emb, top_k=5)}")

    print("\n" + "=" * 60)
    print("Tests complete!")


if __name__ == "__main__":
    test_splade_embedder()
