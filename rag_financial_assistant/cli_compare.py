import argparse

from rag.langchain_pipeline import LangChainRAGPipeline
from rag.rag_pipeline import RAGPipeline


def print_result(title, answer, contexts):
    print(f"\n{title}")
    print("-" * len(title))
    print("\nAnswer:\n")
    print(answer)
    print("\nSources:\n")
    for context in contexts:
        print(f"{context['source']} (page {context['page']})")


def main():
    parser = argparse.ArgumentParser(description="Compare custom and LangChain RAG pipelines.")
    parser.add_argument(
        "--pipeline",
        choices=["custom", "langchain", "both"],
        default="both",
        help="Pipeline version to run.",
    )
    args = parser.parse_args()

    custom_rag = None
    langchain_rag = None

    if args.pipeline in ("custom", "both"):
        custom_rag = RAGPipeline()

    if args.pipeline == "langchain":
        langchain_rag = LangChainRAGPipeline()
    elif args.pipeline == "both":
        langchain_rag = LangChainRAGPipeline(generator=custom_rag.generator)

    print("Financial RAG Assistant Pipeline Compare")
    print("----------------------------------------")

    while True:
        query = input("\nAsk a question (or type exit): ")

        if query == "exit":
            break

        if custom_rag is not None:
            answer, contexts = custom_rag.run(query)
            print_result("Custom Pipeline", answer, contexts)

        if langchain_rag is not None:
            answer, contexts = langchain_rag.run(query)
            print_result("LangChain Pipeline", answer, contexts)


if __name__ == "__main__":
    main()
