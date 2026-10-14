from pathlib import Path

from langchain_chroma import Chroma

from app.embeddings.embedding_model import get_embeddings
from app.ingestion.document_loader import load_pdf
from app.ingestion.text_splitter import split_documents


DB_PATH = "data/chroma"
COLLECTION_NAME = "documind_documents"


def get_vector_store():
    """
    Create or open the local ChromaDB vector store.
    """

    Path(DB_PATH).mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=DB_PATH,
    )

    return vector_store


def index_pdf(file_path: str):
    """
    Load, split, embed, and store a PDF.
    """

    documents = load_pdf(file_path)

    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks were extracted from the PDF.")

    vector_store = get_vector_store()

    ids = vector_store.add_documents(chunks)

    return len(ids)


if __name__ == "__main__":

    pdf_path = "data/raw/sample.pdf"

    if not Path(pdf_path).is_file():
        raise FileNotFoundError(
            f"Please add a PDF at {pdf_path}"
        )

    # Index the PDF
    count = index_pdf(pdf_path)

    print(f"Stored {count} chunks in ChromaDB.")

    # Search the stored documents
    vector_store = get_vector_store()

    query = "What is the main topic of this document?"

    results = vector_store.similarity_search(
        query=query,
        k=3,
    )

    print(f"\nRetrieved {len(results)} chunks.")

    for index, document in enumerate(results, start=1):

        print(f"\n--- Result {index} ---")

        print("Text:", document.page_content[:300])

        print("Source:", document.metadata.get("source"))

        page = document.metadata.get("page")

        print(
            "Page:",
            page + 1 if isinstance(page, int) else "Unknown"
        )