def chunk_text(text, chunk_size=400, overlap=50):

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def chunk_documents(documents, chunk_size, overlap):

    chunked_docs = []

    for doc in documents:

        chunks = chunk_text(doc["text"], chunk_size, overlap)

        for i, chunk in enumerate(chunks):

            chunked_docs.append({
                "text": chunk,
                "source": doc["source"],
                "page": doc["page"],
                "chunk_id": i
            })

    return chunked_docs