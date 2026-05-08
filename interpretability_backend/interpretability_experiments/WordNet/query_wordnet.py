"""
Query the embedded WordNet using semantic search.

This script provides an interface to search the ChromaDB-embedded WordNet
definitions using natural language queries.

It reads from the precomputed embeddings created by embed_wordnet.py.
"""

import sys

from interpretability.utils.utils import setup_collection


def query_wordnet(collection, query_text, n_results=10, show_examples=False):
    """
    Query the WordNet collection.

    Args:
        collection: ChromaDB collection
        query_text: Natural language query
        n_results: Number of results to return
        show_examples: Whether to show usage examples
    """
    results = collection.query(query_texts=[query_text], n_results=n_results)

    print(f"\nQuery: '{query_text}'")
    print(f"Top {n_results} results:")
    print("=" * 70)

    for i, (doc_id, doc, distance) in enumerate(
        zip(results["ids"][0], results["documents"][0], results["distances"][0]), 1
    ):
        meta = results["metadatas"][0][i - 1]

        print(f"\n{i}. {meta['word']} ({meta['pos']}) - similarity: {1 - distance:.4f}")
        print(f"   Definition: {meta['definition']}")

        if show_examples and meta.get("examples"):
            examples = meta["examples"].split(" | ")
            print("   Examples:")
            for ex in examples:
                print(f"     • {ex}")

    print("\n" + "=" * 70)


def interactive_mode(collection):
    """Run in interactive query mode."""
    print("\n" + "=" * 70)
    print("Interactive WordNet Semantic Search")
    print("=" * 70)
    print("Enter your queries to find semantically similar words.")
    print("Commands:")
    print("  - Type a query to search")
    print("  - Type 'examples' to toggle showing examples")
    print("  - Type 'quit' or 'exit' to quit")
    print("=" * 70)

    show_examples = False

    while True:
        try:
            query = input("\nQuery: ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if query.lower() == "examples":
                show_examples = not show_examples
                print(f"Examples display: {'ON' if show_examples else 'OFF'}")
                continue

            # Allow user to specify number of results
            if query.startswith("top "):
                parts = query.split(" ", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    n_results = int(parts[1])
                    print(f"Set number of results to: {n_results}")
                    continue

            query_wordnet(collection, query, n_results=10, show_examples=show_examples)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    # Check if vector database exists
    try:
        collection, device = setup_collection()
        count = collection.count()
        print("✓ Connected to WordNet vector database")
        print(f"✓ Collection contains {count:,} word definitions")
        print(f"✓ Using device: {device}")
    except Exception as e:
        print("Error: Could not connect to vector database.")
        print("Make sure you have run 'embed_wordnet.py' first to create the embeddings.")
        print(f"Details: {e}")
        sys.exit(1)

    # Check if query provided as command line argument
    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
        query_wordnet(collection, query_text, n_results=10)
    else:
        # Interactive mode
        interactive_mode(collection)


if __name__ == "__main__":
    main()
