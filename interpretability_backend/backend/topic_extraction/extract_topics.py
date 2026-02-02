
def _extract_words_per_topic(
        self,
        words: List[str],
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix = None,
        fine_tune_representation: bool = True,
        calculate_aspects: bool = True,
        embeddings: np.ndarray = None,
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Based on tf_idf scores per topic, extract the top n words per topic.

        If the top words per topic need to be extracted, then only the `words` parameter
        needs to be passed. If the top words per topic in a specific timestamp, then it
        is important to pass the timestamp-based c-TF-IDF matrix and its corresponding
        labels.

        Arguments:
            words: List of all words (sorted according to tf_idf matrix position)
            documents: DataFrame with documents and their topic IDs
            c_tf_idf: A c-TF-IDF matrix from which to calculate the top words
            fine_tune_representation: If True, the topic representation will be fine-tuned using representation models.
                                      If False, the topic representation will remain as the base c-TF-IDF representation.
            calculate_aspects: Whether to calculate additional topic aspects
            embeddings: Pre-trained document embeddings. These can be used
                        instead of an embedding model

        Returns:
            topics: The top words per topic
        """
        if c_tf_idf is None:
            c_tf_idf = self.c_tf_idf_

        labels = sorted(list(documents.Topic.unique()))
        labels = [int(label) for label in labels]

        # Get at least the top 30 indices and values per row in a sparse c-TF-IDF matrix
        top_n_words = max(self.top_n_words, 30)
        indices = self._top_n_idx_sparse(c_tf_idf, top_n_words)
        scores = self._top_n_values_sparse(c_tf_idf, indices)
        sorted_indices = np.argsort(scores, 1)
        indices = np.take_along_axis(indices, sorted_indices, axis=1)
        scores = np.take_along_axis(scores, sorted_indices, axis=1)

        # Get top 30 words per topic based on c-TF-IDF score
        base_topics = {
            label: [
                (words[word_index], score) if word_index is not None and score > 0 else ("", 0.00001)
                for word_index, score in zip(indices[index][::-1], scores[index][::-1])
            ]
            for index, label in enumerate(labels)
        }

        # Fine-tune the topic representations
        topics = base_topics.copy()
        if not self.representation_model or not fine_tune_representation:
            # Default representation: c_tf_idf + top_n_words
            topics = {label: values[: self.top_n_words] for label, values in topics.items()}
        elif fine_tune_representation and isinstance(self.representation_model, list):
            for tuner in self.representation_model:
                topics = tuner.extract_topics(self, documents, c_tf_idf, topics)
        elif fine_tune_representation and isinstance(self.representation_model, KeyBERTInspired):
            topics = self.representation_model.extract_topics(self, documents, c_tf_idf, topics, embeddings)
        elif fine_tune_representation and isinstance(self.representation_model, BaseRepresentation):
            topics = self.representation_model.extract_topics(self, documents, c_tf_idf, topics)
        elif fine_tune_representation and isinstance(self.representation_model, dict):
            if self.representation_model.get("Main"):
                main_model = self.representation_model["Main"]
                if isinstance(main_model, BaseRepresentation):
                    topics = main_model.extract_topics(self, documents, c_tf_idf, topics)
                elif isinstance(main_model, list):
                    for tuner in main_model:
                        topics = tuner.extract_topics(self, documents, c_tf_idf, topics)
                else:
                    raise TypeError(f"unsupported type {type(main_model).__name__} for representation_model['Main']")
            else:
                # Default representation: c_tf_idf + top_n_words
                topics = {label: values[: self.top_n_words] for label, values in topics.items()}
        else:
            raise TypeError(f"unsupported type {type(self.representation_model).__name__} for representation_model")

        # Extract additional topic aspects
        if calculate_aspects and isinstance(self.representation_model, dict):
            for aspect, aspect_model in self.representation_model.items():
                if aspect != "Main":
                    aspects = base_topics.copy()
                    if not aspect_model:
                        # Default representation: c_tf_idf + top_n_words
                        aspects = {label: values[: self.top_n_words] for label, values in aspects.items()}
                    if isinstance(aspect_model, list):
                        for tuner in aspect_model:
                            aspects = tuner.extract_topics(self, documents, c_tf_idf, aspects)
                    elif isinstance(aspect_model, BaseRepresentation):
                        aspects = aspect_model.extract_topics(self, documents, c_tf_idf, aspects)
                    else:
                        raise TypeError(
                            f"unsupported type {type(aspect_model).__name__} for representation_model[{aspect!r}]"
                        )
                    self.topic_aspects_[aspect] = aspects

        return topics

    def _reduce_topics(self, documents: pd.DataFrame, use_ctfidf: bool = False) -> pd.DataFrame:
        """Reduce topics to self.nr_topics.

        Arguments:
            documents: Dataframe with documents and their corresponding IDs and Topics
            use_ctfidf: Whether to calculate distances between topics based on c-TF-IDF embeddings. If False, semantic
                        embeddings are used.

        Returns:
            documents: Updated dataframe with documents and the reduced number of Topics
        """
        logger.info("Topic reduction - Reducing number of topics")
        initial_nr_topics = len(self.get_topics())

        if isinstance(self.nr_topics, int):
            if self.nr_topics < initial_nr_topics:
                documents = self._reduce_to_n_topics(documents, use_ctfidf)
            else:
                logger.info(
                    f"Topic reduction - Number of topics ({self.nr_topics}) is equal or higher than the clustered topics({len(self.get_topics())})."
                )
                self._extract_topics(documents, verbose=self.verbose)
                return documents
        elif isinstance(self.nr_topics, str):
            documents = self._auto_reduce_topics(documents, use_ctfidf)
        else:
            raise ValueError("nr_topics needs to be an int or 'auto'! ")

        logger.info(
            f"Topic reduction - Reduced number of topics from {initial_nr_topics} to {len(self.get_topic_freq())}"
        )
        return documents

    def _reduce_to_n_topics(self, documents: pd.DataFrame, use_ctfidf: bool = False) -> pd.DataFrame:
        """Reduce topics to self.nr_topics.

        Arguments:
            documents: Dataframe with documents and their corresponding IDs and Topics
            use_ctfidf: Whether to calculate distances between topics based on c-TF-IDF embeddings. If False, semantic
                        embedding are used.

        Returns:
            documents: Updated dataframe with documents and the reduced number of Topics
        """
        topics = documents.Topic.tolist().copy()

        # Create topic distance matrix
        topic_embeddings = select_topic_representation(
            self.c_tf_idf_, self.topic_embeddings_, use_ctfidf, output_ndarray=True
        )[0][self._outliers :]
        distance_matrix = 1 - cosine_similarity(topic_embeddings)
        np.fill_diagonal(distance_matrix, 0)

        # Cluster the topic embeddings using AgglomerativeClustering
        if version.parse(sklearn_version) >= version.parse("1.4.0"):
            cluster = AgglomerativeClustering(self.nr_topics - self._outliers, metric="precomputed", linkage="average")
        else:
            cluster = AgglomerativeClustering(
                self.nr_topics - self._outliers,
                affinity="precomputed",
                linkage="average",
            )
        cluster.fit(distance_matrix)
        new_topics = [cluster.labels_[topic] if topic != -1 else -1 for topic in topics]

        # Track mappings and sizes of topics for merging topic embeddings
        mapped_topics = {from_topic: to_topic for from_topic, to_topic in zip(topics, new_topics)}
        basic_mappings = defaultdict(list)
        for key, val in sorted(mapped_topics.items()):
            basic_mappings[val].append(key)
        mappings = {
            topic_to: {
                "topics_from": topics_from,
                "topic_sizes": [self.topic_sizes_[topic] for topic in topics_from],
            }
            for topic_to, topics_from in basic_mappings.items()
        }

        # Map topics
        documents.Topic = new_topics
        self._update_topic_size(documents)
        self.topic_mapper_.add_mappings(mapped_topics, topic_model=self)

        # Update representations
        documents = self._sort_mappings_by_frequency(documents)
        self._extract_topics(documents, mappings=mappings, verbose=self.verbose)

        self._update_topic_size(documents)
        return documents

    def _auto_reduce_topics(self, documents: pd.DataFrame, use_ctfidf: bool = False) -> pd.DataFrame:
        """Reduce the number of topics automatically using HDBSCAN.

        Arguments:
            documents: Dataframe with documents and their corresponding IDs and Topics
            use_ctfidf: Whether to calculate distances between topics based on c-TF-IDF embeddings. If False, the
                        embeddings from the embedding model are used.

        Returns:
            documents: Updated dataframe with documents and the reduced number of Topics
        """
        topics = documents.Topic.tolist().copy()
        unique_topics = sorted(list(documents.Topic.unique()))[self._outliers :]

        # Find similar topics
        embeddings = select_topic_representation(
            self.c_tf_idf_, self.topic_embeddings_, use_ctfidf, output_ndarray=True
        )[0]
        norm_data = normalize(embeddings, norm="l2")

        if HAS_HDBSCAN:
            predictions = HDBSCAN(
                min_cluster_size=2,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            ).fit_predict(norm_data[self._outliers :])
        else:
            predictions = SK_HDBSCAN(
                min_cluster_size=2, metric="euclidean", cluster_selection_method="eom", n_jobs=-1
            ).fit_predict(norm_data[self._outliers :])

        # Map clusters to their lowest topic_id
        cluster_to_lowest = {}
        for cluster, topic_id in zip(predictions, unique_topics):
            if cluster != -1:  # Ignore unclustered items
                if cluster not in cluster_to_lowest:
                    cluster_to_lowest[cluster] = topic_id
                else:
                    cluster_to_lowest[cluster] = min(cluster_to_lowest[cluster], topic_id)

        # Map each topic_id to the lowest topic_id in its cluster
        mapped_topics = {}
        for cluster, topic_id in zip(predictions, unique_topics):
            if cluster == -1:
                mapped_topics[topic_id] = topic_id  # No clustering, stays the same
            else:
                mapped_topics[topic_id] = cluster_to_lowest[cluster]

        documents.Topic = documents.Topic.map(mapped_topics).fillna(documents.Topic).astype(int)
        mapped_topics = {from_topic: to_topic for from_topic, to_topic in zip(topics, documents.Topic.tolist())}

        # Track mappings and sizes of topics for merging topic embeddings
        mappings = defaultdict(list)
        for key, val in sorted(mapped_topics.items()):
            mappings[val].append(key)
        mappings = {
            topic_to: {
                "topics_from": topics_from,
                "topic_sizes": [self.topic_sizes_[topic] for topic in topics_from],
            }
            for topic_to, topics_from in mappings.items()
        }

        # Update documents and topics
        self.topic_mapper_.add_mappings(mapped_topics, topic_model=self)
        documents = self._sort_mappings_by_frequency(documents)
        self._extract_topics(documents, mappings=mappings, verbose=self.verbose)
        self._update_topic_size(documents)
        return documents
