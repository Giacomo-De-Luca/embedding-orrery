"""Warm Docker volumes for the optional SAE demo profile.

The runner is cache-only: it downloads model/SAE assets and prepares the SAE
metadata in shared volumes, but it never loads Gemma into memory. The UI still
controls model loading via the existing "Load Model" action.
"""

import os
import time
from dataclasses import dataclass
from typing import Any

import requests
from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class SaeWarmupConfig:
    backend_health_url: str
    graphql_url: str
    checkpoint: str
    layer: int
    width: str
    hook_type: str
    model_size: str
    variant: str
    wait_timeout_seconds: int
    poll_interval_seconds: float

    @classmethod
    def from_env(cls) -> "SaeWarmupConfig":
        return cls(
            backend_health_url=os.getenv("ORRERY_BACKEND_HEALTH_URL", "http://backend:8000/health"),
            graphql_url=os.getenv("ORRERY_GRAPHQL_URL", "http://backend:8000/graphql"),
            checkpoint=os.getenv("ORRERY_MODEL_CHECKPOINT", "google/gemma-3-4b-it"),
            layer=int(os.getenv("ORRERY_SAE_LAYER", "9")),
            width=os.getenv("ORRERY_SAE_WIDTH", "16k"),
            hook_type=os.getenv("ORRERY_SAE_HOOK_TYPE", "resid_post"),
            model_size=os.getenv("ORRERY_SAE_MODEL_SIZE", "4b"),
            variant=os.getenv("ORRERY_SAE_VARIANT", "it"),
            wait_timeout_seconds=int(os.getenv("ORRERY_BACKEND_WAIT_TIMEOUT", "600")),
            poll_interval_seconds=float(os.getenv("ORRERY_BACKEND_POLL_INTERVAL", "2")),
        )


class SaeWarmupRunner:
    def __init__(self, config: SaeWarmupConfig):
        self.config = config

    def run(self) -> None:
        self.wait_for_backend()
        self.prefetch_model()
        self.prepare_sae_data()
        print("SAE warmup complete. Cached assets are ready for UI-triggered model loading.")

    def wait_for_backend(self) -> None:
        deadline = time.monotonic() + self.config.wait_timeout_seconds
        print(f"Waiting for backend health at {self.config.backend_health_url}")

        last_error = "backend did not respond"
        while time.monotonic() < deadline:
            try:
                response = requests.get(self.config.backend_health_url, timeout=5)
                if response.ok:
                    print("Backend is healthy.")
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except requests.RequestException as exc:
                last_error = str(exc)

            time.sleep(self.config.poll_interval_seconds)

        raise RuntimeError(f"Timed out waiting for backend health: {last_error}")

    def prefetch_model(self) -> None:
        token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
            or os.getenv("HUGGINGFACE_API_KEY")
        )

        print(f"Prefetching HuggingFace model snapshot: {self.config.checkpoint}")
        snapshot_download(repo_id=self.config.checkpoint, token=token)
        print("Model snapshot is present in the HuggingFace cache.")

    def prepare_sae_data(self) -> None:
        payload = {
            "query": """
                mutation PrepareSaeData($input: PrepareSaeInput!) {
                  prepareSaeData(input: $input) {
                    status
                    error
                    modelId
                    saeId
                    featuresInserted
                    durationSeconds
                    featuresParquet
                  }
                }
            """,
            "variables": {
                "input": {
                    "layer": self.config.layer,
                    "width": self.config.width,
                    "hookType": self.config.hook_type,
                    "modelSize": self.config.model_size,
                    "variant": self.config.variant,
                    "skipDownload": False,
                    "includeActivations": False,
                    "createCollection": False,
                }
            },
        }

        print(
            "Preparing SAE data "
            f"layer={self.config.layer} width={self.config.width} "
            f"hook={self.config.hook_type} model={self.config.model_size}-{self.config.variant}"
        )
        response = requests.post(self.config.graphql_url, json=payload, timeout=None)
        response.raise_for_status()
        data: dict[str, Any] = response.json()

        if data.get("errors"):
            raise RuntimeError(f"GraphQL prepareSaeData errors: {data['errors']}")

        result = data.get("data", {}).get("prepareSaeData")
        if not result:
            raise RuntimeError(f"GraphQL prepareSaeData returned no result: {data}")
        if result.get("status") != "completed" or result.get("error"):
            raise RuntimeError(f"SAE preparation failed: {result}")

        print(
            "SAE data prepared: "
            f"{result['modelId']} / {result['saeId']} "
            f"features={result['featuresInserted']} "
            f"parquet={result.get('featuresParquet')}"
        )


def main() -> None:
    SaeWarmupRunner(SaeWarmupConfig.from_env()).run()


if __name__ == "__main__":
    main()
