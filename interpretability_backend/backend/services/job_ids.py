"""Job ids shared by progress events, job state and cancellation.

Every long-running job reports progress, records job state and registers its
cancel event under one id. They are built here only, so a modal subscribing to
a job's progress, the Jobs list and the ``cancelJob`` mutation always agree.
Mirrored by ``embedding_visualization/lib/utils/jobIds.ts`` (both sides are
tested against the same literal examples).
"""


class JobIds:
    """Id builders for every long-running job."""

    @staticmethod
    def embedding(collection_name: str) -> str:
        return collection_name

    @staticmethod
    def topic_extraction(collection_name: str) -> str:
        # Same id as the embedding: the embed pipeline's topic step reports on it.
        return collection_name

    @staticmethod
    def topic_reduction(collection_name: str) -> str:
        return f"{collection_name}_reduce"

    @staticmethod
    def topic_evaluation(collection_name: str) -> str:
        return f"{collection_name}_evaluate"

    @staticmethod
    def llm_labeling(collection_name: str) -> str:
        return f"{collection_name}_llm_labeling"

    @staticmethod
    def sae_activations(collection_name: str) -> str:
        return f"{collection_name}_sae_activations"

    @staticmethod
    def probe(collection_name: str) -> str:
        return f"{collection_name}_probe"

    @staticmethod
    def sae_prepare(model_size: str, variant: str, layer: int, hook_type: str, width: str) -> str:
        return f"sae_prepare_{model_size}_{variant}_{layer}_{hook_type}_{width}"
