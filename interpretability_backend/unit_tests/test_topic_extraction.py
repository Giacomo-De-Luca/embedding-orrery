#!/usr/bin/env python3
"""
Test script for topic extraction on a specific collection.

Usage:
    python test_topic_extraction.py [collection_name]

Example:
    python test_topic_extraction.py imdb
"""

import asyncio
import logging
import sys

from interpretability_backend.backend.clients.chromadb_client import ChromaDBClient
from interpretability_backend.backend.services.topic_extraction_service import (
    TopicExtractionConfig,
    extract_topics,
)

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")


def print_topics_result(result):
    """Print topic extraction results in a readable format."""
    print("\n" + "=" * 80)
    print(f"TOPIC EXTRACTION RESULTS FOR: {result.collection_name}")
    print("=" * 80)

    if result.error:
        print(f"\n❌ ERROR: {result.error}")
        return

    print(f"\n✅ Extraction completed in {result.duration_seconds:.2f} seconds")
    print("\n📊 SUMMARY:")
    print(f"   • Topics Found: {result.num_topics}")
    print(f"   • Noise Points: {result.num_noise_points}")
    print(f"   • Total Topics (including noise): {result.num_topics + 1}")

    if result.topics:
        print("\n📋 TOPIC DETAILS:\n")

        for topic_info in result.topics:
            print(f"   Topic {topic_info.topic_id}: {topic_info.label or 'Unlabeled'}")
            print(f"   Count: {topic_info.count} items")

            # Print top keywords
            if topic_info.keywords:
                keywords_str = ", ".join([
                    f"{word} ({score:.3f})"
                    for word, score in topic_info.keywords[:5]
                ])
                print(f"   Keywords: {keywords_str}")

            print()

    print("=" * 80)


async def main():
    """Main test function."""
    # Get collection name from command line or use default
    collection_name = sys.argv[1] if len(sys.argv) > 1 else "imdb"

    print(f"\n🔬 Testing Topic Extraction on collection: '{collection_name}'")
    print("=" * 80)

    # Check if collection exists
    print("\n1️⃣ Checking if collection exists...")
    client = ChromaDBClient()

    try:
        collection_info = client.get_collection_info(collection_name)
        print(f"   ✅ Collection found: {collection_info['count']} items")

        metadata = collection_info['metadata']
        has_projections = metadata.get('has_projections', False)

        if not has_projections:
            print(f"\n❌ ERROR: Collection '{collection_name}' does not have projections!")
            print("   Run projections first before extracting topics.")
            return

        print("   ✅ Collection has projections")

        # Check available projection types
        print("\n   Available projection info:")
        print(f"      • Embedding dim: {metadata.get('embedding_dim', 'unknown')}")
        print(f"      • Embedding model: {metadata.get('embedding_model', 'unknown')}")

    except Exception as e:
        print(f"\n❌ ERROR: Collection '{collection_name}' not found: {e}")
        print("\n   Available collections:")
        collections = client.list_collections()
        for col in collections:
            print(f"      • {col['name']} ({col['count']} items)")
        return

    # Configure topic extraction
    print("\n2️⃣ Configuring topic extraction...")
    config = TopicExtractionConfig(
        collection_name=collection_name,
        min_topic_size=10,
        n_keywords=10,
        use_llm_labels=True,
        llm_provider="gemini",
        llm_model="gemini-3-flash-preview",
        projection_type="umap_2d"
    )

    print("   Configuration:")
    print(f"      • Min topic size: {config.min_topic_size}")
    print(f"      • Keywords per topic: {config.n_keywords}")
    print(f"      • Use LLM labels: {config.use_llm_labels}")
    print(f"      • LLM provider: {config.llm_provider}")
    print(f"      • LLM model: {config.llm_model}")
    print(f"      • Projection type: {config.projection_type}")

    # Extract topics
    print("\n3️⃣ Extracting topics...")
    print("   (This may take 10-60 seconds depending on collection size)")

    try:
        result = await asyncio.to_thread(extract_topics, config)

        # Print results
        print_topics_result(result)

        # Verify metadata was updated
        print("\n4️⃣ Verifying metadata update...")
        collection_info = client.get_collection_info(collection_name)
        metadata = collection_info['metadata']

        if metadata.get('has_topics'):
            print("   ✅ Collection metadata updated successfully")
            print(f"      • Topic count: {metadata.get('topic_count')}")
            print(f"      • Extracted at: {metadata.get('topics_extracted_at')}")
        else:
            print("   ⚠️  Metadata update may have failed")

    except Exception as e:
        print(f"\n❌ ERROR during extraction: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n✅ Test completed successfully!")
    print("\nNext steps:")
    print("   1. Open frontend: http://localhost:3000")
    print(f"   2. Select collection: {collection_name}")
    print("   3. Color by: topic_label or topic_id")
    print("   4. View topics in the legend")


if __name__ == "__main__":
    asyncio.run(main())
