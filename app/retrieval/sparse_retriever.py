import re

from langchain_community.retrievers import BM25Retriever

from app.ingestion.document_loader import load_pdf
from app.ingestion.text_splitter import split_documents


def tokenize(text: str) -> list[str]:
    """
    Convert text into lowercase search tokens.
    """
    return re.findall(r"\b\w+(?:[.-]\w+)*\b", text.lower())


def get_sparse_retriever(
    file_path: str,
    k: int = 3,
):
    """
    Build a BM25 retriever from PDF chunks.
    """

    # Step 1: Load the PDF
    documents = load_pdf(file_path)

    # Step 2: Split PDF pages into chunks
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks found in the PDF.")

    # Step 3: Build the BM25 index
    retriever = BM25Retriever.from_documents(
        documents=chunks,
        preprocess_func=tokenize,
        k=k,
    )

    return retriever


if __name__ == "__main__":

    pdf_path = "data/raw/sample.pdf"

    retriever = get_sparse_retriever(
        file_path=pdf_path,
        k=4,
    )

    question = input("Enter your search question: ").strip()

    if not question:
        raise ValueError("Please enter a question.")

    results = retriever.invoke(question)

    print(f"\nRetrieved {len(results)} chunks")

    for index, document in enumerate(results, start=1):

        print(f"\n--- Result {index} ---")
        print(document.page_content[:400])

        print("Source:", document.metadata.get("source"))

        page = document.metadata.get("page")

        print(
            "Page:",
            page + 1 if isinstance(page, int) else "Unknown"
        )