from app.retrieval.vector_store import get_vector_store

def get_dense_retriever(k: int = 3):
    """
    Create a dense retriever using ChromaDB
    """

    # Step 1: Open the existing vector database
    vector_store = get_vector_store()

    # Step 2: Convert the vector store into a retriever
    retriever = vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs={'k':k},
    )

    # step 3: return the retriever
    return retriever

if __name__ == "__main__":

    retriever = get_dense_retriever(k=3)

    question = "What is the main topic of this document?"

    documents = retriever.invoke(question)

    print("Question:", question)
    print("Retrieved documents:", len(documents))

    for index, document in enumerate(documents, start=1):

        print(f"\n--- Retrieved Document {index} ---")

        print("Content:")
        print(document.page_content[:400])

        print("Source:", document.metadata.get("source"))

        page = document.metadata.get("page")

        print(
            "Page:",
            page + 1 if isinstance(page, int) else "Unknown"
        )