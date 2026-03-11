import os
import fitz


def load_documents(folder):
    documents = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if file.endswith(".pdf"):
            doc = fitz.open(path)

            for page_num, page in enumerate(doc):
                text = page.get_text()

                documents.append({
                    "text": text,
                    "source": file,
                    "page": page_num
                })

    return documents