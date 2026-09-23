import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import src.config as config


def test_rag_package_import():
    import src.rag as rag
    assert rag is not None

def test_rag_modules_import():
    import src.rag.embeddings as embeddings
    import src.rag.vector_store as vector_store
    import src.rag.retriever as retriever
    import src.rag.prompts as prompts
    import src.rag.llm_client as llm_client
    import src.rag.pipeline as pipeline

    assert embeddings is not None
    assert vector_store is not None
    assert retriever is not None
    assert prompts is not None
    assert llm_client is not None
    assert pipeline is not None

def test_vector_store_config():
    assert hasattr(config, "RAG_EMBEDDING_MODEL")
    assert config.RAG_EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    
    assert hasattr(config, "VECTOR_STORE_DIR")
    assert isinstance(config.VECTOR_STORE_DIR, Path)
    assert config.VECTOR_STORE_DIR.exists()
    
    assert hasattr(config, "VECTOR_INDEX_PATH")
    assert hasattr(config, "VECTOR_METADATA_PATH")
    assert hasattr(config, "DEFAULT_TOP_K")
    assert config.DEFAULT_TOP_K == 5

def test_vector_store_gitkeep_exists():
    gitkeep = config.VECTOR_STORE_DIR / ".gitkeep"
    assert gitkeep.exists(), "artifacts/vector_store/.gitkeep must exist"

def test_fastapi_app_import():
    from src.api.main import app
    assert app is not None
    assert app.title == "Safety Analytics Unified API"
