from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_reranker_model() -> CrossEncoder:
    """
    Load and cache the local cross-encoder model.
    """

    model = CrossEncoder(
        MODEL_NAME,
        device="cpu",
        max_length=512,
    )

    return model


def rerank_documents(
    question: str,
    documents: list[Document],
    top_k: int = 4,
) -> list[Document]:
    """
    Rerank retrieved documents by query relevance.
    """

    if not documents:
        return []

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    # Step 1: Load the cross-encoder
    model = get_reranker_model()

    # Step 2: Create query-document pairs
    pairs = [
        (question, document.page_content)
        for document in documents
    ]

    # Step 3: Predict relevance scores
    scores = model.predict(
        pairs,
        batch_size=8,
        show_progress_bar=False,
    )

    # Step 4: Combine documents with scores
    scored_documents = list(zip(documents, scores))

    # Step 5: Sort by descending relevance
    scored_documents.sort(
        key=lambda item: float(item[1]),
        reverse=True,
    )

    # Step 6: Return the top documents
    return [
        document
        for document, score in scored_documents[:top_k]
    ]


if __name__ == "__main__":

    from app.retrieval.hybrid_retriever import get_hybrid_results

    question = input("Enter your question: ").strip()

    if not question:
        raise ValueError("Please enter a question.")

    # Retrieve more candidates than we need
    documents = get_hybrid_results(
        question=question,
        file_path="data/raw/sample.pdf",
        dense_k=8,
        sparse_k=8,
        final_k=12,
    )

    # Rerank and keep the top 4
    reranked = rerank_documents(
        question=question,
        documents=documents,
        top_k=4,
    )

    print(f"\nReranked results: {len(reranked)}")

    for index, document in enumerate(reranked, start=1):

        print(f"\n--- Rank {index} ---")
        print(document.page_content[:400])

        print("Source:", document.metadata.get("source"))

        page = document.metadata.get("page")

        print(
            "Page:",
            page + 1 if isinstance(page, int) else "Unknown"
        )