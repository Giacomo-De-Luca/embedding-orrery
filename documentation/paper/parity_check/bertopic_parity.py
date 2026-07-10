"""Parity check: Orrery topic pipeline vs BERTopic on identical inputs.

For each dataset: embed with all-MiniLM-L6-v2, then run
  (a) Orrery as shipped: _reduce_for_clustering (UMAP 5-D, min_dist=0, cosine,
      random_state=7) -> GenerateTopics (HDBSCAN min_cluster_size=10 + c-TF-IDF,
      english stop words)
  (b) BERTopic with the identical UMAP/HDBSCAN/vectorizer configuration.
Score both with the same evaluator (in-space silhouette, DBCV, diversity,
C_v / U_Mass via gensim) + ARI between the two labelings.

Writes JSON + markdown + LaTeX (booktabs) to the scratchpad.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/Users/jack/EmbeddingVisualisation")
sys.path.insert(0, "/Users/jack/EmbeddingVisualisation/interpretability_backend")

import numpy as np
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score

from interpretability_backend.backend.services.topic_extraction_service import (
    _reduce_for_clustering,
)
from interpretability_backend.backend.topic_extraction.cluster_and_label import GenerateTopics
from interpretability_backend.evaluation.quality_metrics import TopicQualityEvaluator

SCRATCH = Path("/private/tmp/claude-501/-Users-jack-EmbeddingVisualisation/0c146576-f833-4cb2-915c-4e4242141c75/scratchpad")
SEED_DB = "/Users/jack/EmbeddingVisualisation/interpretability_backend/resources/seed/main.duckdb"
MIN_TOPIC_SIZE = 10
UMAP_KW = dict(n_components=5, min_dist=0.0, n_neighbors=15)  # shipped defaults


# ---------------------------------------------------------------- datasets
def load_emotion():
    import duckdb

    con = duckdb.connect(SEED_DB, read_only=True)
    rows = con.execute(
        "select document from items_emotion order by row_index"
    ).fetchall()
    con.close()
    return [r[0] or "" for r in rows]


def load_20ng():
    from sklearn.datasets import fetch_20newsgroups

    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    return [d.strip() for d in data.data]


def load_ag_news():
    from datasets import load_dataset

    ds = load_dataset("ag_news", split="test")  # 7,600 docs
    return [r["text"] for r in ds]


DATASETS = [
    ("emotion (1k, in-house)", load_emotion),
    ("AG News test (7.6k)", load_ag_news),
    ("20 Newsgroups (18.8k)", load_20ng),
]


# ---------------------------------------------------------------- pipelines
def run_orrery(docs, reduced):
    gen = GenerateTopics(documents=docs, min_topic_size=MIN_TOPIC_SIZE, language="english")
    df = gen.generate_clusters(reduced)
    topics_data = gen.extract_topics(df, n_words=10)
    return df["Topic"].to_numpy(), topics_data, gen.hdbscan_model


def run_bertopic(docs, emb):
    from umap import UMAP
    from bertopic import BERTopic

    tm = BERTopic(
        umap_model=UMAP(metric="cosine", random_state=7, verbose=False, **UMAP_KW),
        hdbscan_model=HDBSCAN(
            min_cluster_size=MIN_TOPIC_SIZE,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
            gen_min_span_tree=True,
        ),
        vectorizer_model=CountVectorizer(stop_words="english"),
        verbose=False,
    )
    labels, _ = tm.fit_transform(docs, emb)
    labels = np.asarray(labels)
    topics_data = {
        int(t): [(w, float(s)) for w, s in tm.get_topic(t)]
        for t in set(labels.tolist())
    }
    return labels, topics_data, tm.hdbscan_model, tm.umap_model.embedding_


def score(labels, space, topics_data, docs, model):
    ev = TopicQualityEvaluator()
    res = ev.evaluate(
        labels=labels,
        projection_coords=space,
        topics_data=topics_data,
        documents=docs,
        language="english",
        hdbscan_model=model,
        sample_size=10_000_000,  # exact silhouette for parity
    )
    mask = labels != -1
    return {
        "topics": int(len(set(labels[mask].tolist()))),
        "noise_pct": float((~mask).mean() * 100),
        "dbcv": res["dbcv"],
        "silhouette": res["silhouette_cluster_space"],
        "diversity": res["topic_diversity"],
        "c_v": res["coherence_cv"],
        "u_mass": res["coherence_umass"],
    }


# ---------------------------------------------------------------- main
def main():
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer("all-MiniLM-L6-v2")
    results = []

    for name, loader in DATASETS:
        t0 = time.time()
        try:
            docs = loader()
        except Exception as e:
            print(f"[skip] {name}: dataset load failed: {e}", flush=True)
            continue
        # Empty docs break c-TF-IDF joins; keep alignment by replacing.
        docs = [d if isinstance(d, str) and d.strip() else "(empty)" for d in docs]
        print(f"[{name}] {len(docs)} docs; embedding...", flush=True)
        emb = st.encode(docs, batch_size=256, show_progress_bar=False)
        emb = np.asarray(emb)

        print(f"[{name}] Orrery pipeline...", flush=True)
        reduced = _reduce_for_clustering(emb, **UMAP_KW)
        o_labels, o_topics, o_model = run_orrery(docs, reduced)
        ours = score(o_labels, reduced, o_topics, docs, o_model)

        print(f"[{name}] BERTopic...", flush=True)
        b_labels, b_topics, b_model, b_space = run_bertopic(docs, emb)
        bert = score(b_labels, b_space, b_topics, docs, b_model)

        ari = float(adjusted_rand_score(o_labels, b_labels))
        sil_diff = (
            abs(ours["silhouette"] - bert["silhouette"])
            if ours["silhouette"] is not None and bert["silhouette"] is not None
            else None
        )
        cv_diff = (
            abs(ours["c_v"] - bert["c_v"])
            if ours["c_v"] is not None and bert["c_v"] is not None
            else None
        )
        results.append(
            {
                "dataset": name,
                "n_docs": len(docs),
                "ours": ours,
                "bertopic": bert,
                "ari": ari,
                "abs_diff_silhouette": sil_diff,
                "abs_diff_c_v": cv_diff,
                "seconds": round(time.time() - t0, 1),
            }
        )
        print(f"[{name}] done in {time.time()-t0:.0f}s  ARI={ari:.4f}", flush=True)

    (SCRATCH / "bertopic_parity.json").write_text(json.dumps(results, indent=2))

    # ---- markdown ----
    def f4(x):
        return "—" if x is None else f"{x:.4f}"

    md = ["| Dataset | N | Pipeline | Topics | Noise % | Silhouette | C_v | Diversity | U_Mass | DBCV | ARI |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        for key, label in (("ours", "Orrery"), ("bertopic", "BERTopic")):
            m = r[key]
            md.append(
                f"| {r['dataset'] if key=='ours' else ''} | {r['n_docs'] if key=='ours' else ''} | {label} "
                f"| {m['topics']} | {m['noise_pct']:.1f} | {f4(m['silhouette'])} | {f4(m['c_v'])} "
                f"| {f4(m['diversity'])} | {f4(m['u_mass'])} | {f4(m['dbcv'])} "
                f"| {f4(r['ari']) if key=='ours' else ''} |"
            )
    (SCRATCH / "bertopic_parity.md").write_text("\n".join(md))
    print("\n".join(md))

    # ---- LaTeX (booktabs) ----
    tex = [
        "% Parity check: Orrery topic pipeline vs BERTopic (identical UMAP seed/params,",
        "% HDBSCAN min_cluster_size=10, english stop words; all-MiniLM-L6-v2 embeddings).",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Dataset & Pipeline & Topics & Noise\\,\\% & Silhouette & $C_v$ & Diversity & ARI \\\\",
        "\\midrule",
    ]
    for r in results:
        o, b = r["ours"], r["bertopic"]
        short = r["dataset"].split(" (")[0]
        tex.append(
            f"\\multirow{{2}}{{*}}{{{short} ({r['n_docs']:,})}} & Orrery & {o['topics']} & {o['noise_pct']:.1f} "
            f"& {f4(o['silhouette'])} & {f4(o['c_v'])} & {f4(o['diversity'])} & \\multirow{{2}}{{*}}{{{f4(r['ari'])}}} \\\\"
        )
        tex.append(
            f" & BERTopic & {b['topics']} & {b['noise_pct']:.1f} "
            f"& {f4(b['silhouette'])} & {f4(b['c_v'])} & {f4(b['diversity'])} & \\\\"
        )
        tex.append("\\midrule")
    tex[-1] = "\\bottomrule"
    tex += [
        "\\end{tabular}",
        "\\caption{Parity between Orrery's topic pipeline and BERTopic on identical inputs"
        " (same MiniLM embeddings, UMAP 5-D $\\min\\_dist{=}0$ seed 7, HDBSCAN"
        " $\\text{min\\_cluster\\_size}{=}10$, English stop words). ARI compares the two"
        " document--topic assignments.}",
        "\\label{tab:bertopic-parity}",
        "\\end{table}",
    ]
    (SCRATCH / "bertopic_parity.tex").write_text("\n".join(tex))
    print(f"\nWrote {SCRATCH}/bertopic_parity.{{json,md,tex}}")


if __name__ == "__main__":
    main()
