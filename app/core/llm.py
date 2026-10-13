from langchain_ollama import ChatOllama
from app.core.config import settings


def get_llm() -> ChatOllama:
    """
    Initialize the local qwen3 llm using ollama
    """
    llm = ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        num_ctx=settings.ollama_num_ctx,
        num_predict=512,
        reasoning=False,
    )

    return llm