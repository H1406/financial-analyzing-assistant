import os

from ingestion.pdf_extractor import extract_documents


def load_documents(folder):
    documents = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if not os.path.isfile(path):
            continue

        with open(path, "rb") as handle:
            file_bytes = handle.read()

        documents.extend(extract_documents(file_bytes, file))

    return documents
