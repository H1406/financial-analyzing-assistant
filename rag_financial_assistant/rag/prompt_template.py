def build_prompt(query, contexts):

    context_block = "\n\n".join(
        [f"Context {i+1}: {c['text']}" for i, c in enumerate(contexts)]
    )

    prompt = f"""
You are a financial analysis assistant.

Use the provided context to answer the question.

Context:
{context_block}

Question:
{query}

Instructions:
- Answer based only on the provided context


Answer:
"""

    return prompt

# - If the answer cannot be found, say "The information is not available in the documents."