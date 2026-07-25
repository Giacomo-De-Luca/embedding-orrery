---
title: Orrery — Embedding Observatory
emoji: 🪐
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
header: mini
fullWidth: true
short_description: Explore embedding spaces and SAE features — EMNLP, WordNet, colors
---

# Orrery — Embedding Observatory (demo)

Interactive exploration of embedding spaces: 2D/3D scatter plots with topic
clustering, semantic search, text filtering, and temporal analytics.

## Start here

- **[Take the 90-second tour]({{SPACE_URL}}?tour=1)** — a guided walk through
  the map, topics, semantic search, and analytics.
- **[Explore research topics]({{SPACE_URL}}?preset=emnlp-topics)** — 13,980
  EMNLP abstracts clustered into 60 LLM-labeled research topics.
- **[Explore the color manifold]({{SPACE_URL}}?preset=xkcd-manifold)** — 954
  color names embedded as text; the rainbow re-emerges from language alone.
- **[Inspect SAE features]({{SPACE_URL}}?tour=sae)** — start on a map of
  16,384 sparse-autoencoder features of gemma-3-4b-it, open one, see what
  makes it fire, and replay a chat steered with it.
- **[Open fullscreen]({{SPACE_DIRECT_URL}})** — the app on its own page,
  without the HF chrome.

## Collections in this demo

| Collection | Items | Embeddings | What to try |
|---|---|---|---|
| **EMNLP abstracts** | 13,980 | gemini-embedding-2 (3072-d) | 60 LLM-labeled research topics; search "hallucination in summarization"; filter by year; rank abstracts by SAE feature in the Search panel |
| **emotion** | 1,000 | all-MiniLM-L6-v2 (384-d) | Emotion-labeled tweets; semantic search runs fully inside the Space |
| **xkcd colors** | 954 | gemini-embedding-2 (3072-d) | Color names embedded as text — color by `mapped_colour` and see the rainbow |
| **WordNet senses** | 212,478 | — | An atlas of English word senses — the platform at scale (first load takes a moment) |
| **SAE label map** | 15,854 | all-MiniLM-L6-v2 (384-d) | Every gemma-3-4b-it SAE feature label as a point — right-click one to open it in the explorer |

Tips: pick a collection top-right, color by `topic_label`, enable the nebula
overlay (3D), right-drag to rotate, and use the search panel for semantic or
substring search. The view state round-trips through the URL, so any view you
reach is shareable by copying the address.

## This Space is read-only

SAE feature *browsing* is live: the explorer serves labels, logits, activation
examples, and feature→document search from pre-computed tables. The steered
chat opens too, replaying conversations recorded against the real engine.
What stays disabled is everything that needs a running language model — prompt
activations, steering, and live generation — plus the write paths: embedding
new datasets, topic extraction, probe training, and label editing
(`ORRERY_READ_ONLY=1` blocks all GraphQL mutations server-side).

**Want to embed your own data?** Run the full application from the
[Orrery GitHub repository](https://github.com/Giacomo-De-Luca/orrery) with its
published backend/frontend images or local Docker build. Duplicated Spaces do
not inherit this Space's secrets, including the read token for its private demo
seed, and are therefore not the supported installation path.

Semantic search on the two Gemini-embedded collections uses this Space's
`GEMINI_API_KEY` secret (one embedding call per query). If the secret is
absent, browsing, topics, and filtering still work — only text-query semantic
search on those two collections is unavailable.

## Source

Built from the Orrery repository — backend: FastAPI + Strawberry GraphQL with
a DuckDB/ChromaDB dual-store; frontend: Next.js + a forked Plotly.js WebGL
renderer.
