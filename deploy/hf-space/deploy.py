"""Deploy the read-only Orrery demo to a HuggingFace Docker Space.

Uploads the working tree (filtered) to the Space repo with huggingface_hub,
which handles LFS automatically for the large demo-seed files — no git-history
surgery needed, and the ~313 MB seed never touches the GitHub repo.

Three commits per deploy:
  1. the filtered repo tree (Dockerfile, backend, frontend, demo seed, deploy/)
  2. deploy/hf-space/README_SPACE.md → README.md  (Space card + frontmatter)
  3. Dockerfile.dockerignore → .dockerignore  (the seed-including variant, so
     the HF builder gets the seed even if it ignores per-Dockerfile files)

Prerequisites:
  - the demo seed exists (see documentation/HF_SPACE_DEMO.md):
      uv run python -m interpretability_backend.scripts.build_seed_snapshot \
          --collections emotion xkcd_hilbert_gemini acl_abstracts_emnlp_findings \
          --datasets emotion xkcd_hilbert acl_abstracts_emnlp_findings \
          --output interpretability_backend/resources/seed_demo
  - a HuggingFace token with write access (env HF_TOKEN or `hf auth login`)

Usage:
    uv run python deploy/hf-space/deploy.py --repo-id <user>/orrery-demo [--create]

After the first deploy, set the Space secret GEMINI_API_KEY (Settings →
Variables and secrets) to enable semantic search on the Gemini collections.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = REPO_ROOT / "deploy" / "hf-space"
DEMO_SEED = REPO_ROOT / "interpretability_backend" / "resources" / "seed_demo"

# Everything the Dockerfile does not need stays out of the Space repo. The
# live data stores are the critical ones (main.duckdb is ~23 GB on disk).
# NOTE: fnmatch patterns match the FULL relative path — "benchmarks/*" does
# not cover "interpretability_backend/benchmarks/*", and ".env" does not
# cover nested ".env" files; both need explicit entries.
IGNORE_PATTERNS = [
    ".git/*",
    ".git*",
    ".venv/*",
    ".claude/*",
    ".codex/*",
    "**/__pycache__/*",
    "**/*.pyc",
    ".pytest_cache/*",
    ".ruff_cache/*",
    ".DS_Store",
    "**/.DS_Store",
    # secrets — root AND nested (e.g. embedding_visualization/.env.local)
    ".env",
    ".env.*",
    "**/.env",
    "**/.env.*",
    # not needed to build the image
    "docs/*",
    "documentation/*",
    "gallery/*",
    "benchmarks/*",
    "references/*",
    "neuronpedia/*",
    "node_modules/*",
    "Papers/*",
    # experiments + local data (includes copyrighted corpora, e.g. Lacan TSVs
    # under interpretability_experiments/ — must never reach the public repo)
    "interpretability_backend/interpretability_experiments/*",
    "interpretability_backend/test/*",
    "interpretability_backend/benchmarks/*",
    # frontend build artifacts / deps (rebuilt in the image)
    "embedding_visualization/node_modules/*",
    "embedding_visualization/.next/*",
    "embedding_visualization/out/*",
    "embedding_visualization/coverage/*",
    "embedding_visualization/Users/*",
    "**/tsconfig.tsbuildinfo",
    "**/target/*",
    # backend live stores + local-only data (the demo seed ships instead;
    # the committed seed is unused by the Space image, which sets
    # ORRERY_SEED_DIR to seed_demo)
    "interpretability_backend/resources/main.duckdb",
    "interpretability_backend/resources/main.duckdb.*",
    "interpretability_backend/resources/vector_db/*",
    "interpretability_backend/resources/seed/*",
    "interpretability_backend/resources/uploads/*",
    "interpretability_backend/resources/job_state.json",
    "interpretability_backend/resources/sae_labels/*",
    "interpretability_backend/resources/sae_vectors/*",
    "interpretability_backend/resources/refusal_direction/*",
    "interpretability_backend/resources/experiments/*",
    "interpretability_backend/resources/extracted_activations/*",
    "interpretability_backend/resources/probing_results/*",
    "interpretability_backend/resources/psycolinguistics/*",
    "interpretability_backend/resources/acl_anthology/*",
    "interpretability_backend/resources/datasets/*",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repo-id", required=True, help="Space repo, e.g. <user>/orrery-demo")
    parser.add_argument("--create", action="store_true", help="create the Space if it is missing")
    parser.add_argument("--private", action="store_true", help="create the Space as private")
    args = parser.parse_args()

    if not (DEMO_SEED / "main.duckdb").exists():
        print(
            f"ERROR: demo seed not found at {DEMO_SEED}.\n"
            "Build it first (backend stopped) — see the module docstring.",
            file=sys.stderr,
        )
        return 1

    api = HfApi()

    if args.create:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="space",
            space_sdk="docker",
            private=args.private,
            exist_ok=True,
        )
        print(f"[deploy] space ensured: {args.repo_id}")

    print(f"[deploy] uploading filtered tree from {REPO_ROOT} (large files go via LFS)…")
    api.upload_folder(
        folder_path=str(REPO_ROOT),
        repo_id=args.repo_id,
        repo_type="space",
        ignore_patterns=IGNORE_PATTERNS,
        commit_message="Deploy Orrery demo",
    )

    print("[deploy] uploading Space README (frontmatter) and .dockerignore…")
    api.upload_file(
        path_or_fileobj=str(DEPLOY_DIR / "README_SPACE.md"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="space",
        commit_message="Space README",
    )
    api.upload_file(
        path_or_fileobj=str(REPO_ROOT / "Dockerfile.dockerignore"),
        path_in_repo=".dockerignore",
        repo_id=args.repo_id,
        repo_type="space",
        commit_message="Space .dockerignore (includes demo seed)",
    )

    space_url = f"https://huggingface.co/spaces/{args.repo_id}"
    print(
        f"\nDeployed. Watch the build at {space_url}\n"
        "Reminder: set the Space secret GEMINI_API_KEY to enable semantic "
        "search on the EMNLP + xkcd collections."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
