from app.core.llm import get_llm
from app.embeddings.embedding_model import get_embeddings


def test_llm():
    print("\nTesting local LLM...")

    llm = get_llm()

    response = llm.invoke(
        "Explain machine learning in two sentences."
    )

    print("LLM Response:", response.content)

    assert response.content


def test_embeddings():
    print("\nTesting local embeddings...")

    embeddings = get_embeddings()

    vector = embeddings.embed_query(
        "What is machine learning?"
    )

    print("Embedding dimensions:", len(vector))
    print("First 5 values:", vector[:5])

    assert len(vector) > 0