def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += max(chunk_size - overlap, 1)

    return chunks


def _section_prefix(doc):
    title = (doc.get("section_title") or "").strip()
    section_type = (doc.get("section_type") or "").strip()

    if not title:
        return ""

    if section_type and section_type != "body":
        return f"{title}\n[{section_type}]\n"

    return f"{title}\n"


def chunk_documents(documents, chunk_size, overlap):
    chunked_docs = []

    for doc_index, doc in enumerate(documents):
        text = doc["text"].strip()

        if not text:
            continue

        prefix = _section_prefix(doc)
        full_text = f"{prefix}{text}".strip()
        raw_chunks = [full_text] if len(full_text) <= chunk_size else chunk_text(full_text, chunk_size, overlap)

        for chunk_index, chunk in enumerate(raw_chunks):
            chunked_docs.append({
                "text": chunk,
                "source": doc["source"],
                "page": doc["page"],
                "chunk_id": f"{doc_index}-{chunk_index}",
                "section_title": doc.get("section_title"),
                "section_type": doc.get("section_type", "body"),
            })

    return chunked_docs
