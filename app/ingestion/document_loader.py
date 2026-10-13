from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str):
    """
    Load a PDF and return LangChain Document objects.
    """

    # Step 1: Convert string path into a Path object
    path = Path(file_path)

    # Step 2: Check whether the PDF exists
    if not path.is_file():
        raise FileNotFoundError(
            f"PDF file not found: {path}"
        )

    # Step 3: Initialize the PDF loader
    loader = PyPDFLoader(str(path), mode="page")

    # Step 4: Extract pages as Document objects
    documents = loader.load()

    # Step 5: Return the extracted documents
    return documents


if __name__ == "__main__":

    pdf_path = "data/raw/sample.pdf"

    documents = load_pdf(pdf_path)

    print("Total Documents:", len(documents))

    if documents:
        print("\nFirst Page Content:")
        print(documents[0].page_content[:500])

        print("\nFirst Page Metadata:")
        print(documents[0].metadata)