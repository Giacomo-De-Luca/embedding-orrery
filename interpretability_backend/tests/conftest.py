"""
Pytest configuration and fixtures for backend tests.
"""

import pytest
import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))


@pytest.fixture
def sample_metadata():
    """Sample item metadata for testing."""
    return [
        {"word": "cat", "pos": "n", "definition": "a small feline"},
        {"word": "dog", "pos": "n", "definition": "a loyal canine"},
        {"word": "run", "pos": "v", "definition": "to move quickly"},
        {"word": "fast", "pos": "a", "definition": "moving with speed"},
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embedding vectors for testing."""
    import numpy as np
    return np.random.rand(4, 384).tolist()


@pytest.fixture
def temp_csv_file(tmp_path):
    """Create a temporary CSV file for testing."""
    csv_content = """id,name,value
1,Alice,100
2,Bob,200
3,Charlie,300
"""
    file_path = tmp_path / "test_data.csv"
    file_path.write_text(csv_content)
    return str(file_path)


@pytest.fixture
def temp_json_file(tmp_path):
    """Create a temporary JSON file for testing."""
    import json
    data = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]
    file_path = tmp_path / "test_data.json"
    file_path.write_text(json.dumps(data))
    return str(file_path)
