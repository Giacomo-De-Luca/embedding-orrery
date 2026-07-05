### Sections


**Abstract**

is an open-source platform for interactive visualisation of embedding spaces with native support for Sparse Autoencoder (SAE) interpretability. Given any local or HuggingFace dataset, it produces an explorable 3D constellation through a single pipeline of embedding, projection, clustering, automatic topic labelling, and, optionally, collection of per-document SAE feature activations. The resulting map can be searched visually, semantically, lexically (BM25), or by SAE feature names. SAE decoder vectors are visualised as a navigable constellation of their own, where inputting a document highlights the features it activates. Individual features can be inspected in a Neuronpedia-style dashboard and applied directly to steer the model. The interface sustains 60 fps on 250k points, including animations and nebula shaders, on 8 GB of RAM. Orrery is released under the Apache 2.0 license at [URL].


Introduction

related works: 

1. embedding projector
2. embedding atlas
3. nomic atlas
4. **Latent Scope**

Artistic works : 

**embedding galaxy (Google)**

nebula

https://artsexperiments.withgoogle.com/tsnemap/

Not really working: 

**DataMapPlot**

### Embedding Visualiser:

#### Techniques:

1. TSNE
2. UMAP
3. Self Organising maps
4. PCA

!Screenshot 2026-06-23 at 13.59.33.png

Paper

gallery: 

Concreteness

Colours

!Screenshot 2026-06-23 at 14.01.26.png

Visualisation of the English words in the brysbaert dataset.  

!Gemini_XKCD_PCA.png

Visualisation of the embeddings of the English colour words from the XKCD colour survey using Gemini-embeddings-2. Each point is coloured with original colour associated with the colour word. The embedding space self organises around the colour space. 

[]()

!Screenshot 2026-06-06 at 12.36.26.png

Sample from the Emotions Dataset

#### Competing Systems

Tensorflow embedding projector

Embedding Atlas

Nomic Embedding

**Latent Scope**

**DataMapPlot**

**Artistic projects:** 

**embedding galaxy (Google)**

https://artsexperiments.withgoogle.com/tsnemap/

Nebula

Contribution

A collection manager to embed, project and store vectors collections inside chroma. 

An optimised 3d scatterplot library for visualising textual collections inspired by star cartography with extensive filtering and colour support

### System Design

The system employs two different concepts: datasets and collections. A dataset is comprised of documents (image, text, audio etc.) which are stored and indexed inside DuckDB. Each dataset can have 1 or more associated collections. A collection is a a list of vectors with are stored inside ChromaDB, projections for the collection which are stored inside DuckDB and optionally a list of topics which are also stored in duckDB. A collection can also have SAE activations for each document, which are also stored inside DuckDB. 

By decoupling datasets and collections, it’s possible to embed multiple time the same dataset with different prompts, column combinations, models, extracted topics etc. without having to store multiple time a dataset, which was the main limitation of chroma. 

Starting from a dataset, which can be loaded from either HuggingFace or local (JSON, parquet, csv)

The dataset is loaded as a preview. 

Embedding the dataset can be skipped for datasets that already have a vector column. The Topic modeling and SAE activations are entirely optional. The only required step is computing and storing projections for the dataset. 

1. **Embedding the dataset*:** 

From the dataset, a column combination can be chosen to be embedded together with a prompt. For making template, one can use the column names between graph brackets, keeping the prompt outside the brackets. 

{word}: {definition}

The dataset is then embedded using one of the providers: SentenceTransformers, Ollama, BGE with Flag embeddings, Qwen, or using API: OpenAI, Cohere, HuggingFace, Gemini. For Gemma and Gemini, task types are supported. 

The resulting embedding and embedding function are stored inside chroma db. The columns which are selected as metadata are stored in DuckDB. 

1. **Computing and storing projections.** 

Once the embeddings are computed, the projections for the embeddings are computed and stored inside DuckDB. We automatically compute 2d and 3d projections for both PCA and UMAP. 

1. **Topic modeling*.** 

If topic extraction is selected, a BERTopic-like pipeline is triggered. Since BERTopic is designed to be modular, we allow multiple clustering algorithms: HDBSCAN, K-means, GMM and Spectral clustering (good for small dataset, but using o(n^2) memory).

Labels for the topic are generated either automatically using cTF-IDF, or using LLM labels. We reuse directly the same prompt as BERTopic for the topic generation. There is optionally a second topic reduction step, which allows to reduce the number of topics to a set amount. 

1. **Sparse Autoencoders activations*.** 

once the collection is stored in the database, it is possible to collect sparse autoencoder activations for the dataset. This feature is supported only with the Gemma-scope sparse autoencoders (any size, any layers). By selecting this feature, each document in the dataset goes in a forward pass through gemma and SAE activations are collected at the token level. For each feature in the SAE we collect the max activation across token and we store the activations in a sparse matrix in DuckDB. 

### Searching the collection

Once the collection is stored in the database, it can be visualised and searched. The collection can be searched using semantic search, text search and sparse autoencoder search. Search results are highlighed in the scatterplot as a costellation. The colour of the stars get warmer the higher is the similarity to the input. For text search it is possible to select the column(s) in which to search, to select between exact and partial matches and if to use case sensitivity. 

It is also possible to semantic search by clicking on any point on the scatterplot. By clicking on a point it automatically zooms in on the point, call the semantic search and retrieves the top k most similar documents in the original space. 

Sparse autoencoder search allows to search by selecting the names of the features. By selecting the ‘football’ feature, it retrieves the top k documents that activates the named feature(s) the most, in order. 

SAE collections also allow to write a prompt and see highlighted in the scatterplot all the points which correspond to activated features. SAE collections also allow to right click on features to directly inspect the feature in a dedicated page. Those features can be directly injected or ablated in the model to test the steering. 

#### filtering

There are multiple options to filter the visualised collection. The most imediate one is using the legend panel. One of the columns in the dataset can be selected to colour the dataset and it appears in the legend panel. If the column contains categorical data, it is displayed as a scrollable and searchable list of points with legend and count. Categories or topics can be selected or unselected there and they are automatically filtered out. 

If the column used is numerical, it is instead displayed as a coloured histogram with handles, which is used to set the colour scale. In the analytics panel is instead possible to select the column in the count histogram, and move the right and left handles to filter graphically the dataset. 

For more precise filtering, there is a filtering option that can add multiple filters, using equality, different, is in, is not in, and ranges. 

### Colour Support

The application has advanced colour scale support to allow both interpretability research and display of historical corpora. 

The application supports categorical, diverging, sequential, and monochrome colour scales. Other than d3 default colour scales, it supports as well Crameri’s scientific colour maps for perceptually uniform and accessible visualisation of scientific datasets. 

## SAE Support

The visualiser provides expanded support for sparse autoencoders, which supports two different workflow. First, the one regarding SAE collections, that can be directly ingested and downloaded with a single click 

This support is thanks to a special kind of collection, called SAE_collection in the database. 

### 1. Visualising SAE collections

The space of the features in a SAE can be visualised in two ways. First, taking for each of the SAE features the corresponding row in the decoder matrix, which has dimension R = n dim of the residual stream of the model. Second, by labelling each feature using an Autointerpreter, and embedding with a traditional encoder. Both possibilities produce a {feature_index, label, vector, top_logit, documents} dataset, which can be projected in 3d and visualised. We offer the possibility of directly embedding the SAE features for any of the Gemma-scope-2 sae, and placing them directly into DuckDB/chroma, downloading them from the Neuronpedia s3 bucket. 

Once a SAE collection is projected into the embedding space, it has two different features compared to a normal collection. First, it can be directly searched by writing a prompt. Since each point in the scatterplot correspond to a feature, the features that activate the most in the prompt via max pooling are highlighted and sorted by activations. 

Second, since the collection can be searched both through vector similarity or text search on the labels, once an interesting feature is found, using right click is possible to directly expand the view of the feature in the *features* page and direct steer the model. 

### 2 Feature explorer.

To make the features more searchable, we introduced a page inspired by Neuronpedia / Anthropic and GPT papers and SAE vis. In this page, features can be searched in three different ways. Text search on the labels, semantic search on the labels, and prompt searching. Prompt searching allows to write a prompt and examine for each token which feature gets activated, or having the max / mean activation list over the prompt. 

### Direct steering.

Differently from Neuronpedia, we keep the SAE feature explorer and the direct steering in the same page. Once you find an interesting feature, you can directly inject it in the model and steer the output

## Examples cases:

### Concreteness and psycholinguistic scales

In Geometry of Truth is shown that the true and false statements are linearly separable in the embedding space. This separation is so clear that even a mass mean probe is able to predict the binary classes. A mass mean probe is a probe in which you get the mean vector in class a, mean vector in class b, and the direction is the difference between a and b. 

One of the main use case of an embedding visualiser for interpretability is finding visually for linear directions in the embedding space. We show that this experiment can be directly replicated on multiple new directions through the visualiser. First we embed the entire dataset of the Concreteness ratings for English words, and visualise it. Each point correspond to an English words, colours to the concreteness - abstract rating given by English speaker. While it’s clear visually that the direction is linearly separable, we also train linear and nonlinear probes. The direction can be reconstructed at 93 % from the embedding vectors. We also replicate this on the Glasgow norm dataset, which is comprised by 7 different psycholinguistic directions (dominance, arousal, concreteness, familiarity, gender, imageability, valence). We also display them directly and train probes on them. We test those directions both on MiniLM and Gemma-embedding-300m.Results are in appendix. Of those, concreteness, imageability, valence are linearly decodable.  It seems that the embedding space self-organises around linear directions. The amount of linear directions in the embedding space that is found is still unknown. We hope the ease of use of the visualiser is going to be helpful in the future in speeding up research.  

### XKCD colour survey

Nonlinear directions in the embedding space can still be visualised. We visualise the dataset from the XKCD colour survey. This dataset is generated from the responses of 200k people asked to label colour patches, and is comprised by about 1k colour terms in English, associated with a hex colour code. We visualise the dataset in 3d in two different ways. First, associating each point to its exact colour. We also generate a linear scale using Hilbert strips to reduce three dimensional colour spaces to a single direction, which is detailed in the appendix. Each point is then associated with its colour in the original space. Projecting the points in the space shows intuitively that the distribution of words in the embedding space resembles the colour space. To quantify this effect, we run a Mantel test with the correlation between each point in the embedding space and each point in the original space, resulting in 0.4 correlation in the Umap space and about 0 in the original space. We also train probes on the colour directions, which are all almost perfectly decodable. This interesting result shows that UMAP actually makes visible relationship that are hidden in the embedding space. 

https://aclanthology.org/2021.conll-1.9.pdf

**Digital humanities dataset: the Lacan and Sanskrit corpus.** 

To show how the platform can be used for the digital humanities, we embed two different corpus: first the entire Lacan corpus at the sentence level, then a subsection of the Sanskrit Travelogue corpus (a corpus of all the currently digitalised Sanskrit literature). We show that the embedding space clearly separates the texts by their time period. For the Lacan’s corpus we show that even with 250k documents and nebula mode activated, the scatterplot is still perfectly working, allowing humanities scholars to search and access the dataset visually. 

### Refusal Direction

To test the correctness of the steering we replicate and adapt the code from Arditi’s refusal is mediated by a single direction in the embedding space to work with Gemma-3-4b-it. We don’t direct port Arditi’s code, but we adapt the methodology to work with our fork of Gemma-Pytorch. Also, unlike Arditi’s paper, which does geometrical steering, ablating the refusal direction at every layer, we use instead ActAdd, similarly to how we steer for SAE. We try ablating the refusal vector at every layer to find which layer change more prompts of HarmBench from safe to unsafe. we also grid search which is the best activation range which steers the model but doesn’t destroy the coherence of the prompts. We find that steering at layer 11 allows us to go from n/128 prompts accepted by the model to the entirety of refusal bench accepted, in a result which is surprisingly superior to the initial paper. We release the full code for the experiment, plus the refusal vector directly in the repo to contribute to research on LLM safety. 

Find a semantic pattern → identify candidate features or directions → inspect evidence → intervene in the model → observe the behavioral effect.

Fixes: 

check if docker is working

clustering cannot cluster on original space

check if ‘all’ works for huggingFace

check if prompt is repeated in configuration

full ingestion of sae collections

demo with SAE

remake bar chart, make bar chart filter properly

control click for bar chart and legend to multi select

**nebula making white spots with clusters mixed**

light mode support: highlight points, card on the left, icons

**lock dark mode at the start**

change the position of the show label button

not loading the whole document corpus

allow saving changed colours for categories as custom palettes

add filtering for categories in advanced

add ranges for filtering in advanced

update next

frontend unit test

bar chart should hide unclustered

add baseline to the chat interface

TESTING:

docker for testers

questionnaire

coherence scores for topic modeling

probes for directions

Orrery unifies corpus-scale semantic exploration with internal feature interpretation and causal model intervention.