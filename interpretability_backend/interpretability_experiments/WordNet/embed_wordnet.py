"""
Embed WordNet definitions using SPLADE sparse embeddings.

This script supports two embedding strategies:
1. Word-level: Extract embeddings from example sentences at target word position
2. Definition-level: Embed "{word}: {definition}" as a sentence

Both strategies are designed to find monosemantic interpretable dimensions.
"""

import os
from collections.abc import Generator
from dataclasses import dataclass

from tqdm import tqdm

try:
    from .splade_embedder import SparseEmbedding, SPLADEEmbedder
    from .storage import QdrantStorage, WordEmbeddingRecord
    from .wordnet_parser import WordNetParser
except ImportError:
    from interpretability.interpretability_experiments.WordNet.wordnet_parser import WordNetParser
    from splade_embedder import SparseEmbedding, SPLADEEmbedder
    from storage import QdrantStorage, WordEmbeddingRecord


# ========== Configuration ==========

WORDNET_XML_PATH = "interpretability/resources/english-wordnet-2024.xml"
QDRANT_DB_PATH = "interpretability/resources/vector_db_qdrant"

# Collection names for different strategies
COLLECTION_WORD_LEVEL = "wordnet_word_level"
COLLECTION_DEFINITION = "wordnet_definition"

BATCH_SIZE = 100


@dataclass
class EmbeddingResult:
    """Result from embedding a word."""

    word: str
    strategy: str
    context: str
    synset_id: str | None
    pos: str | None
    sparse_embedding: SparseEmbedding
    top_tokens: list[str]


def embed_wordnet_word_level(
    wn: WordNetParser, splade: SPLADEEmbedder, limit: int | None = None
) -> Generator[EmbeddingResult, None, None]:
    """
    Strategy A: Word-level extraction from example sentences.

    For each word with examples, embed the word in the context of its
    example sentence, extracting only the sparse vector at the word's
    token position(s).

    Args:
        wn: WordNet parser
        splade: SPLADE embedder
        limit: Optional limit on number of words to process

    Yields:
        EmbeddingResult for each word-example pair
    """
    all_words = wn.get_all_words()
    if limit:
        all_words = all_words[:limit]

    for word in tqdm(all_words, desc="Word-level embedding", unit="word"):
        definitions = wn.get_definitions(word)

        for defn in definitions:
            examples = defn.get("examples", [])
            if not examples:
                continue

            for i, example in enumerate(examples[:2]):  # Max 2 examples per sense
                # Check if word appears in example
                if word.lower() not in example.lower():
                    continue

                sparse_emb = splade.embed_word_in_context(example, word)

                if sparse_emb is None or len(sparse_emb.indices) == 0:
                    continue

                top_tokens = [t for t, _ in splade.decode_tokens(sparse_emb, top_k=20)]

                yield EmbeddingResult(
                    word=word,
                    strategy="word_level",
                    context=example,
                    synset_id=defn.get("synset_id"),
                    pos=defn.get("part_of_speech"),
                    sparse_embedding=sparse_emb,
                    top_tokens=top_tokens,
                )


def embed_wordnet_definition(
    wn: WordNetParser, splade: SPLADEEmbedder, limit: int | None = None
) -> Generator[EmbeddingResult, None, None]:
    """
    Strategy B: Sentence-level embedding from definitions.

    Embed "{word}: {definition}" as a complete sentence using
    standard SPLADE (max-pool across all positions).

    Args:
        wn: WordNet parser
        splade: SPLADE embedder
        limit: Optional limit on number of words to process

    Yields:
        EmbeddingResult for each word
    """
    all_words = wn.get_all_words()
    if limit:
        all_words = all_words[:limit]

    for word in tqdm(all_words, desc="Definition embedding", unit="word"):
        definitions = wn.get_definitions(word)

        if not definitions:
            continue

        # Use first definition
        defn = definitions[0]
        text = f"{word}: {defn['definition']}"

        sparse_emb = splade.embed_sentence(text)

        if len(sparse_emb.indices) == 0:
            continue

        top_tokens = [t for t, _ in splade.decode_tokens(sparse_emb, top_k=20)]

        yield EmbeddingResult(
            word=word,
            strategy="definition",
            context=text,
            synset_id=defn.get("synset_id"),
            pos=defn.get("part_of_speech"),
            sparse_embedding=sparse_emb,
            top_tokens=top_tokens,
        )


def results_to_records(results: list[EmbeddingResult]) -> list[WordEmbeddingRecord]:
    """Convert embedding results to storage records."""
    records = []
    for i, result in enumerate(results):
        record = WordEmbeddingRecord(
            id=f"{result.word}_{result.strategy}_{i}",
            word=result.word,
            strategy=result.strategy,
            context=result.context,
            synset_id=result.synset_id,
            pos=result.pos,
            sparse_embedding=result.sparse_embedding,
            top_tokens=result.top_tokens,
        )
        records.append(record)
    return records


def run_embedding_pipeline(
    strategy: str = "both", limit: int | None = None, recreate: bool = False
):
    """
    Run the embedding pipeline for the specified strategy.

    Args:
        strategy: "word_level", "definition", or "both"
        limit: Optional limit on number of words
        recreate: If True, recreate collections
    """
    print("=" * 70)
    print("WordNet SPLADE Embedding Pipeline")
    print("=" * 70)

    # Check WordNet file
    if not os.path.exists(WORDNET_XML_PATH):
        print(f"Error: WordNet XML file not found at: {WORDNET_XML_PATH}")
        return

    # Initialize WordNet parser
    print("\nStep 1: Parsing WordNet...")
    wn = WordNetParser(WORDNET_XML_PATH)
    wn.parse()
    stats = wn.get_stats()
    print(f"  Loaded {stats['total_words']:,} words")

    # Initialize SPLADE embedder
    print("\nStep 2: Loading SPLADE model...")
    splade = SPLADEEmbedder()

    # Initialize storage
    print("\nStep 3: Initializing Qdrant storage...")
    storage_word_level = None
    storage_definition = None

    if strategy in ["word_level", "both"]:
        storage_word_level = QdrantStorage(
            db_path=QDRANT_DB_PATH, collection_name=COLLECTION_WORD_LEVEL
        )
        storage_word_level.create_collection(recreate=recreate, dense_dim=384)

    if strategy in ["definition", "both"]:
        storage_definition = QdrantStorage(
            db_path=QDRANT_DB_PATH, collection_name=COLLECTION_DEFINITION
        )
        storage_definition.create_collection(recreate=recreate, dense_dim=384)

    # Run word-level embedding
    if strategy in ["word_level", "both"]:
        print("\n" + "=" * 70)
        print("Strategy A: Word-Level Embedding from Examples")
        print("=" * 70)

        results = []
        batch = []

        for result in embed_wordnet_word_level(wn, splade, limit):
            batch.append(result)

            if len(batch) >= BATCH_SIZE:
                records = results_to_records(batch)
                storage_word_level.add_records(records)
                results.extend(batch)
                batch = []

        # Add remaining
        if batch:
            records = results_to_records(batch)
            storage_word_level.add_records(records)
            results.extend(batch)

        print(f"\n  Total word-level embeddings: {len(results):,}")
        print(f"  Stored in collection: {COLLECTION_WORD_LEVEL}")

    # Run definition embedding
    if strategy in ["definition", "both"]:
        print("\n" + "=" * 70)
        print("Strategy B: Definition-Level Embedding")
        print("=" * 70)

        results = []
        batch = []

        for result in embed_wordnet_definition(wn, splade, limit):
            batch.append(result)

            if len(batch) >= BATCH_SIZE:
                records = results_to_records(batch)
                storage_definition.add_records(records)
                results.extend(batch)
                batch = []

        # Add remaining
        if batch:
            records = results_to_records(batch)
            storage_definition.add_records(records)
            results.extend(batch)

        print(f"\n  Total definition embeddings: {len(results):,}")
        print(f"  Stored in collection: {COLLECTION_DEFINITION}")

    print("\n" + "=" * 70)
    print("Embedding pipeline complete!")
    print("=" * 70)


def main():
    """Main entry point with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(description="Embed WordNet using SPLADE sparse embeddings")
    parser.add_argument(
        "--strategy",
        choices=["word_level", "definition", "both"],
        default="both",
        help="Embedding strategy to use",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of words to process (for testing)"
    )
    parser.add_argument(
        "--recreate", action="store_true", help="Recreate collections (delete existing data)"
    )

    args = parser.parse_args()

    run_embedding_pipeline(strategy=args.strategy, limit=args.limit, recreate=args.recreate)


if __name__ == "__main__":
    main()
