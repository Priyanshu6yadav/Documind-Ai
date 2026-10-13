from langchain_ollama import OllamaEmbeddings
from app.core.config import settings

def get_embeddings() -> OllamaEmbeddings:
    """
    Create a local embedding client using ollama
    """
    embeddings = OllamaEmbeddings(
        model =settings.embedding_model,
        base_url = settings.ollama_base_url
    )
    return embeddings