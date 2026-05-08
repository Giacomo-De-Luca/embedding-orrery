from pathlib import Path

import chromadb
from chromadb.config import Settings

# Path to vector db
db_path = Path("interpretability/resources/vector_db").resolve()
print(f"Checking database at: {db_path}")

try:
    client = chromadb.PersistentClient(
        path=str(db_path), settings=Settings(anonymized_telemetry=False)
    )

    collections = client.list_collections()
    print(f"Found {len(collections)} collections")

    for col in collections:
        print(f"\nCollection: {col.name}")
        print(f"Count: {col.count()}")

        # Get one item
        results = col.get(limit=1, include=["metadatas"])
        if results["metadatas"]:
            meta = results["metadatas"][0]
            print("Sample metadata keys:", meta.keys())
            if "pca_2d" in meta:
                print(f"pca_2d value: {meta['pca_2d']}")
                print(f"pca_2d type: {type(meta['pca_2d'])}")
            else:
                print("pca_2d NOT found in metadata")
        else:
            print("No items in collection")

except Exception as e:
    print(f"Error: {e}")
