import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

from config import DOCUMENT_PATH, DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

def main():
    documents = []

    for file in os.listdir(DOCUMENT_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DOCUMENT_PATH, file))
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    texts = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=DB_PATH
    )

    try:
        db.persist()
    except Exception:
        pass

    print("Documents ingested successfully!")

if __name__ == "__main__":
    main()