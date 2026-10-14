from hashlib import sha256

from langchain_core.documents import Document

from app.retrieval.dense_retriever import get_dense_retriever
from app.retrieval.sparse_retriever import get_sparse_retriever


def get_document_id(document: Document) -> str:
    """
    Generate a stable identifier for a document chunk.
    """

    source = str(document.metadata.get("source", ""))
    page = str(document.metadata.get("page", ""))
    start_index = str(document.metadata.get("start_index", ""))

    raw_id = f"{source}|{page}|{start_index}|{document.page_content}"

    return sha256(raw_id.encode("utf-8")).hexdigest()


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """
    Combine ranked document lists using RRF.
    """

    scores = {}
    documents = {}

    for results in result_lists:

        seen_in_list = set()

        for rank, document in enumerate(results, start=1):

            document_id = get_document_id(document)

            if document_id in seen_in_list:
                continue

            seen_in_list.add(document_id)

            documents[document_id] = document

            if document_id not in scores:
                scores[document_id] = 0.0

            scores[document_id] += 1 / (k + rank)

    ranked_ids = sorted(
        scores,
        key=lambda document_id: scores[document_id],
        reverse=True,
    )

    return [documents[document_id] for document_id in ranked_ids]


def get_hybrid_results(
    question: str,
    file_path: str,
    dense_k: int = 8,
    sparse_k: int = 8,
    final_k: int = 4,
) -> list[Document]:
    """
    Retrieve documents using dense and sparse search,
    then combine their rankings.
    """

    dense_retriever = get_dense_retriever(k=dense_k)

    sparse_retriever = get_sparse_retriever(
        file_path=file_path,
        k=sparse_k,
    )

    dense_results = dense_retriever.invoke(question)
    sparse_results = sparse_retriever.invoke(question)

    fused_results = reciprocal_rank_fusion(
        [dense_results, sparse_results]
    )

    return fused_results[:final_k]


if __name__ == "__main__":

    question = input("Enter your question: ").strip()

    if not question:
        raise ValueError("Please enter a question.")

    results = get_hybrid_results(
        question=question,
        file_path="data/raw/sample.pdf",
    )

    print(f"\nHybrid retrieval returned {len(results)} chunks.")

    for index, document in enumerate(results, start=1):

        print(f"\n--- Result {index} ---")
        print(document.page_content[:400])

        print("Source:", document.metadata.get("source"))

        page = document.metadata.get("page")

        print(
            "Page:",
            page + 1 if isinstance(page, int) else "Unknown"
        )