# src/literature_buddy/models/model_loader.py
"""Ensure Transformers model weights are present before backend construction.

- Only active when provider == "transformers".
- Uses HF snapshot_download / from_pretrained with local_dir to ensure
  weights are downloaded once into the configured directory.
- Logs progress and avoids re-downloading if the directory already contains
  a valid snapshot.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_transformers_model(
    model_id_or_path: str,
    cache_dir: str | Path | None = None,
) -> str:
    """Ensure a Transformers model is available locally.

    Returns the local path to the model directory.

    - If `model_id_or_path` looks like a local path (exists on disk), it is
      returned as-is.
    - Otherwise, treats it as a Hugging Face model ID and downloads (or
      verifies) the snapshot into `cache_dir` (or HF default cache).
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError

    local_path = Path(model_id_or_path).expanduser()
    if local_path.exists() and (local_path / "config.json").exists():
        logger.info(f"Model already present at: {local_path}")
        return str(local_path)

    # Treat as HF model ID
    model_id = model_id_or_path
    logger.info(f"Model not found locally; downloading {model_id} ...")

    try:
        downloaded = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            ignore_patterns=["*.msgpack", "*.h5", "*.mlmodel"],
        )
        logger.info(f"Model downloaded to: {downloaded}")
        return downloaded
    except RepositoryNotFoundError as e:
        raise RuntimeError(
            f"Model '{model_id}' not found on Hugging Face Hub. "
            "Check the model ID or use a local path."
        ) from e