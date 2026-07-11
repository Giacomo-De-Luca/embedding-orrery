---
title: Orrery — Embedding Observatory
emoji: 🪐
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Explore embedding spaces — EMNLP abstracts, emotions, colors
---

# Orrery — Embedding Observatory (demo)

Interactive exploration of embedding spaces: 2D/3D scatter plots with topic
clustering, semantic search, text filtering, and temporal analytics.

## Collections in this demo

| Collection | Items | Embeddings | What to try |
|---|---|---|---|
| **EMNLP abstracts** | 13,980 | gemini-embedding-2 (3072-d) | 60 LLM-labeled research topics; search "hallucination in summarization"; filter by year |
| **emotion** | 1,000 | all-MiniLM-L6-v2 (384-d) | Emotion-labeled tweets; semantic search runs fully inside the Space |
| **xkcd colors** | 954 | gemini-embedding-2 (3072-d) | Color names embedded as text — color by `mapped_colour` and see the rainbow |

Tips: pick a collection top-left, color by `topic_label`, enable the nebula
overlay (3D), right-drag to rotate, and use the search panel for semantic or
substring search.

## This Space is read-only

The full platform embeds datasets from HuggingFace or local files, extracts
and labels topics, trains probes on embedding spaces, and runs live SAE
(sparse autoencoder) inference with steering — those write paths are disabled
here (`ORRERY_READ_ONLY=1` blocks all GraphQL mutations server-side).

**Want to embed your own data?** Duplicate this Space, then:

1. Set a Space **Variable** `ORRERY_READ_ONLY=0` (re-enables mutations and
   keeps your data in the Space's storage; add persistent storage if you want
   it to survive restarts).
2. Add your embedding-provider keys as Space **Secrets** as needed:
   `GEMINI_API_KEY`, `CHROMA_OPENAI_API_KEY`, `CHROMA_COHERE_API_KEY`, …
   (SentenceTransformers models run locally with no key).
3. To also unhide the Collections/SAE pages, change the
   `NEXT_PUBLIC_DEMO_MODE` build ARG default to `0` in the `Dockerfile`
   (it is baked into the frontend bundle at build time).

Semantic search on the two Gemini-embedded collections uses this Space's
`GEMINI_API_KEY` secret (one embedding call per query). If the secret is
absent, browsing, topics, and filtering still work — only text-query semantic
search on those two collections is unavailable.

## Source

Built from the Orrery repository — backend: FastAPI + Strawberry GraphQL with
a DuckDB/ChromaDB dual-store; frontend: Next.js + a forked Plotly.js WebGL
renderer.
