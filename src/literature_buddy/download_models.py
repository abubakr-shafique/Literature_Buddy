#!/usr/bin/env python3
"""Download all required models to local directory."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from literature_buddy.config.settings import load_settings
from literature_buddy.models.model_loader import ModelLoader


def main():
    """Download all models."""
    print("=" * 60)
    print("Literature Buddy - Model Downloader")
    print("=" * 60)
    
    # Load settings
    config_path = Path(__file__).parent / "config" / "settings.yaml"
    settings = load_settings(config_path)
    
    # Create model loader
    model_loader = ModelLoader(settings)
    
    print(f"\nModel directory: {model_loader.model_path}")
    print("\nThis will download:")
    print("1. Embedding model (all-MiniLM-L6-v2) ~90MB")
    print("2. LLM (TinyLlama-1.1B-Chat) ~2.4GB")
    print("3. Reranker (ms-marco-MiniLM-L-6-v2) ~90MB")
    print("\nTotal: ~2.6GB")
    
    response = input("\nContinue? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    print("\n" + "=" * 60)
    
    # Download models
    try:
        print("\n[1/3] Downloading embedding model...")
        model_loader.load_embedding_model(force_download=True)
        print("✓ Embedding model downloaded")
        
        print("\n[2/3] Downloading LLM (this may take a while)...")
        model_loader.load_llm(force_download=True)
        print("✓ LLM downloaded")
        
        print("\n[3/3] Downloading reranker...")
        model_loader.load_reranker(force_download=True)
        print("✓ Reranker downloaded")
        
        print("\n" + "=" * 60)
        print("All models downloaded successfully!")
        print(f"Models saved to: {model_loader.model_path.absolute()}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Internet connection")
        print("2. Sufficient disk space (~3GB)")
        print("3. Required packages: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()