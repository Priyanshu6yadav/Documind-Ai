from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.ingestion.document_loader import load_pdf

def split_documents(documents):
    """
    Split Langchian Documents object into smaller chunks
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)

    return chunks

if __name__ == "__main__":

    # Step 1: Load the PDF
    documents = load_pdf("data/raw/sample.pdf")

    # Step 2: Split into chunks
    chunks = split_documents(documents)

    # Step 3: Display results
    print("Total PDF pages:", len(documents))
    print("Total chunks:", len(chunks))

    if chunks:
        print("\nFirst chunk:")
        print(chunks[0].page_content[:500])

        print("\nFirst chunk metadata:")
        print(chunks[0].metadata)