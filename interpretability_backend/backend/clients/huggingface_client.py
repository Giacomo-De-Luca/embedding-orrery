"""HuggingFace dataset client for fetching dataset information and embedding datasets."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class SplitInfo:
    """Information about a dataset split."""

    name: str
    num_rows: int | None = None
    num_bytes: int | None = None


@dataclass
class FeatureInfo:
    """Information about a dataset feature/column."""

    name: str
    dtype: str
    description: str | None = None


@dataclass
class ConfigInfo:
    """Information about a dataset configuration."""

    name: str
    splits: list[SplitInfo] = field(default_factory=list)
    features: list[FeatureInfo] = field(default_factory=list)


@dataclass
class DatasetInfo:
    """Complete information about a HuggingFace dataset."""

    dataset_id: str
    description: str | None = None
    license: str | None = None
    configs: list[ConfigInfo] = field(default_factory=list)
    default_config: str | None = None
    error: str | None = None


@dataclass
class DatasetPreview:
    """Preview rows from a dataset."""

    dataset_id: str
    config: str | None
    split: str
    columns: list[str]
    rows: list[dict[str, Any]]
    total_rows: int | None = None
    error: str | None = None


class PortionStrategy(Enum):
    """Strategy for selecting which rows to embed."""

    FIRST_N = "first_n"
    RANDOM_SAMPLE = "random_sample"
    ROW_RANGE = "row_range"
    ALL = "all"


@dataclass
class PortionConfig:
    """Configuration for selecting dataset portion."""

    strategy: PortionStrategy
    n: int | None = None  # For FIRST_N and RANDOM_SAMPLE
    start: int | None = None  # For ROW_RANGE
    end: int | None = None  # For ROW_RANGE
    seed: int = 42  # For RANDOM_SAMPLE


def _dtype_to_string(dtype: Any) -> str:
    """Convert HuggingFace dtype to string representation."""
    dtype_str = str(dtype)

    if "Value" in dtype_str:
        if hasattr(dtype, "dtype"):
            return str(dtype.dtype)
        return dtype_str
    elif "ClassLabel" in dtype_str:
        return "class_label"
    elif "Sequence" in dtype_str:
        return "sequence"
    elif "Image" in dtype_str:
        return "image"
    elif "Audio" in dtype_str:
        return "audio"
    else:
        return dtype_str


def get_dataset_info(dataset_id: str) -> DatasetInfo:
    """
    Fetch comprehensive information about a HuggingFace dataset.

    Uses streaming mode to avoid downloading full datasets.

    Args:
        dataset_id: HuggingFace dataset ID (e.g., "squad", "glue")

    Returns:
        DatasetInfo with configs, splits, features, and metadata
    """
    try:
        from datasets import get_dataset_config_names, load_dataset_builder
        from huggingface_hub import dataset_info as hf_dataset_info
    except ImportError:
        return DatasetInfo(
            dataset_id=dataset_id, error="datasets and huggingface_hub packages are required"
        )

    try:
        # Get dataset card info from Hub
        description = None
        license_info = None

        try:
            hub_info = hf_dataset_info(dataset_id)
            if hub_info.card_data:
                license_info = getattr(hub_info.card_data, "license", None)
                # Handle case where license is a list
                if isinstance(license_info, list):
                    license_info = ", ".join(str(l) for l in license_info) if license_info else None
            if hub_info.description:
                description = hub_info.description[:500]
        except Exception:
            pass

        # Get available configurations
        try:
            config_names = get_dataset_config_names(dataset_id)
        except Exception:
            config_names = [None]

        configs = []
        default_config = None

        for config_name in config_names:
            try:
                builder = load_dataset_builder(dataset_id, name=config_name)

                splits = []
                if builder.info.splits:
                    for split_name, split_info in builder.info.splits.items():
                        raw_rows = getattr(split_info, "num_examples", None)
                        raw_bytes = getattr(split_info, "num_bytes", None)
                        splits.append(
                            SplitInfo(
                                name=split_name,
                                num_rows=int(raw_rows) if raw_rows is not None else None,
                                num_bytes=int(raw_bytes) if raw_bytes is not None else None,
                            )
                        )

                features = []
                if builder.info.features:
                    for feat_name, feat_type in builder.info.features.items():
                        features.append(
                            FeatureInfo(name=feat_name, dtype=_dtype_to_string(feat_type))
                        )

                if not description and builder.info.description:
                    description = builder.info.description[:500]

                if not license_info and builder.info.license:
                    license_info = builder.info.license

                configs.append(
                    ConfigInfo(name=config_name or "default", splits=splits, features=features)
                )

                if default_config is None:
                    default_config = config_name or "default"

            except Exception:
                configs.append(ConfigInfo(name=config_name or "default", splits=[], features=[]))

        return DatasetInfo(
            dataset_id=dataset_id,
            description=description,
            license=license_info,
            configs=configs,
            default_config=default_config,
        )

    except Exception as e:
        return DatasetInfo(dataset_id=dataset_id, error=str(e))


def get_dataset_preview(
    dataset_id: str, config: str | None = None, split: str = "train", n_rows: int = 5
) -> DatasetPreview:
    """
    Fetch preview rows from a HuggingFace dataset using streaming.

    Args:
        dataset_id: HuggingFace dataset ID
        config: Configuration name (None for default)
        split: Split name (default: "train")
        n_rows: Number of rows to preview (default: 5)

    Returns:
        DatasetPreview with column names and sample rows
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return DatasetPreview(
            dataset_id=dataset_id,
            config=config,
            split=split,
            columns=[],
            rows=[],
            error="datasets package is required",
        )

    try:
        config_name = config if config and config != "default" else None
        dataset = load_dataset(dataset_id, name=config_name, split=split, streaming=True)

        columns = list(dataset.features.keys()) if hasattr(dataset, "features") else []

        rows = []
        for i, row in enumerate(dataset):
            if i >= n_rows:
                break
            row_dict = {}
            for key, value in row.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    row_dict[key] = value
                elif isinstance(value, list):
                    row_dict[key] = value[:10] if len(value) > 10 else value
                else:
                    row_dict[key] = str(value)[:200]
            rows.append(row_dict)

        # Try to get total row count
        total_rows = None
        try:
            info = get_dataset_info(dataset_id)
            for cfg in info.configs:
                if cfg.name == (config or "default"):
                    for s in cfg.splits:
                        if s.name == split:
                            total_rows = s.num_rows
                            break
        except Exception:
            pass

        return DatasetPreview(
            dataset_id=dataset_id,
            config=config,
            split=split,
            columns=columns,
            rows=rows,
            total_rows=total_rows,
        )

    except Exception as e:
        return DatasetPreview(
            dataset_id=dataset_id, config=config, split=split, columns=[], rows=[], error=str(e)
        )


def load_dataset_portion(
    dataset_id: str,
    config: str | None = None,
    split: str = "train",
    portion: PortionConfig | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """
    Load a portion of a HuggingFace dataset.

    Args:
        dataset_id: HuggingFace dataset ID
        config: Configuration name (None for default)
        split: Split name
        portion: Portion configuration (default: all rows)

    Returns:
        Tuple of (list of row dicts, total rows in split)
    """
    import random

    from datasets import load_dataset

    if portion is None:
        portion = PortionConfig(strategy=PortionStrategy.ALL)

    config_name = config if config and config != "default" else None

    if portion.strategy == PortionStrategy.FIRST_N:
        dataset = load_dataset(dataset_id, name=config_name, split=split, streaming=True)

        rows = []
        for i, row in enumerate(dataset):
            if i >= portion.n:
                break
            rows.append(dict(row))

        total = None
        try:
            info = get_dataset_info(dataset_id)
            for cfg in info.configs:
                if cfg.name == (config or "default"):
                    for s in cfg.splits:
                        if s.name == split:
                            total = s.num_rows
        except Exception:
            total = len(rows)

        return rows, total or len(rows)

    elif portion.strategy == PortionStrategy.RANDOM_SAMPLE:
        dataset = load_dataset(dataset_id, name=config_name, split=split)

        total = len(dataset)
        n = min(portion.n, total)

        random.seed(portion.seed)
        indices = random.sample(range(total), n)

        rows = [dict(dataset[i]) for i in indices]
        return rows, total

    elif portion.strategy == PortionStrategy.ROW_RANGE:
        dataset = load_dataset(dataset_id, name=config_name, split=split)

        total = len(dataset)
        start = max(0, portion.start or 0)
        end = min(total, portion.end or total)

        rows = [dict(dataset[i]) for i in range(start, end)]
        return rows, total

    else:  # ALL
        dataset = load_dataset(dataset_id, name=config_name, split=split)

        total = len(dataset)
        rows = [dict(row) for row in dataset]
        return rows, total
