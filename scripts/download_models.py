#!/usr/bin/env python3
"""Download the model weights a profile needs into ./models (Transformers profiles),
or print the `ollama pull` commands (Ollama profiles).

    python scripts/download_models.py --profile 16gb
    python scripts/download_models.py --profile 24gb --dry-run
"""

from __future__ import annotations

import argparse
import sys

from literature_buddy.config import available_profiles, load_config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="16gb", choices=available_profiles())
    ap.add_argument("--models-dir", default=None, help="target directory (default: ./models)")
    ap.add_argument("--dry-run", action="store_true", help="only print what would be downloaded")
    args = ap.parse_args()

    cfg = load_config(profile=args.profile, environ={})
    target = (cfg.models_path if not args.models_dir else __import__("pathlib").Path(args.models_dir).resolve())

    hf: list[str] = []
    ollama: list[str] = []
    for mc in (cfg.llm, cfg.vlm if not cfg.vlm.same_as_llm and cfg.vlm.enabled else None):
        if mc is None or not mc.model:
            continue
        (hf if mc.provider == "transformers" else ollama if mc.provider == "ollama" else []).append(mc.model)
    if cfg.embedding.provider == "sentence_transformers":
        hf.append(cfg.embedding.model)
    elif cfg.embedding.provider == "ollama":
        ollama.append(cfg.embedding.model)
    if cfg.retrieval.rerank:
        hf.append(cfg.retrieval.reranker_model)

    for name in dict.fromkeys(ollama):
        print(f"ollama pull {name}")
    if not hf:
        return 0
    print(f"\nHugging Face repos -> {target}")
    for repo in dict.fromkeys(hf):
        print(f"  {repo}")
    if args.dry_run:
        return 0
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub missing: pip install -r requirements-transformers.txt", file=sys.stderr)
        return 1
    target.mkdir(parents=True, exist_ok=True)
    for repo in dict.fromkeys(hf):
        dest = target / repo.replace("/", "__")
        print(f"\nDownloading {repo} ...")
        snapshot_download(repo_id=repo, local_dir=str(dest))
        print(f"  -> {dest}")
    print("\nDone. Set LB_OFFLINE=true to guarantee nothing is fetched from the Hub afterwards.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
