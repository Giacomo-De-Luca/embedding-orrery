"""
Utility for handling duplicate IDs during embedding.
"""


class IDDeduplicator:
    """
    Stateful helper to handle duplicate IDs by appending sequential numbers.
    Example: ["cat", "cat", "cat"] -> ["cat", "cat_1", "cat_2"] or ["cat_1", "cat_2", "cat_3"]

    Based on the user request ["cat", "cat", "cat"] ---> ["cat_1", "cat_2", "cat_3"],
    we will use 1-based indexing for ALL occurrences.
    """

    def __init__(self):
        self.id_counts: dict[str, int] = {}

    def get_unique_id(self, base_id: str) -> str:
        """
        Get a unique ID by appending a sequential number if duplicates are found.

        Args:
            base_id: The original ID.

        Returns:
            A unique ID (e.g., "cat_1", "cat_2").
        """
        # We always append _N even for the first occurrence to match user request:
        # ["cat", "cat", "cat"] ---> ["cat_1", "cat_2", "cat_3"]
        self.id_counts[base_id] = self.id_counts.get(base_id, 0) + 1
        return f"{base_id}_{self.id_counts[base_id]}"
