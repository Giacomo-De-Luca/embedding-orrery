# run_embedding_bge.py
import pickle
from time import time

import numpy as np
from FlagEmbedding import BGEM3FlagModel

MAX_LENGTH = 1024

print("Loading data...")
with open("example_sentence.pickle", "rb") as handle:
    sentences = pickle.load(handle)
sorted_sentences = sorted(sentences, key=len)
print(f"Loaded {len(sorted_sentences)} sentences.")

print("Loading model...")
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="mps")

print("Encoding...")
time_start = time()
output = model.encode(
    sorted_sentences,
    batch_size=16,
    max_length=MAX_LENGTH,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
print(f"Encoding took {time() - time_start:.2f} seconds")

# Save dense embeddings as numpy
np.save("embeddings_bge_dense.npy", output["dense_vecs"])
print(f"Saved dense embeddings: {output['dense_vecs'].shape}")

# Save sparse (lexical weights) as pickle - list of dicts
with open("embeddings_bge_sparse.pickle", "wb") as f:
    pickle.dump(output["lexical_weights"], f)
print(f"Saved sparse embeddings: {len(output['lexical_weights'])} entries")

# Save colbert vecs as pickle - list of variable-length arrays
with open("embeddings_bge_colbert.pickle", "wb") as f:
    pickle.dump(output["colbert_vecs"], f)
print(f"Saved colbert embeddings: {len(output['colbert_vecs'])} entries")
