import yaml
from retrieval.retriever import Retriever


def main():

    config = yaml.safe_load(open("config.yaml"))

    retriever = Retriever(config)

    while True:

        query = input("\nEnter query: ")

        results = retriever.retrieve(query)

        print("\nTop Results:\n")

        for i, r in enumerate(results):

            print(f"Result {i+1}")
            print(f"Source: {r['source']} (page {r['page']})")
            print(f"Score: {r['score']}")
            print(r["text"][:300])
            print("-"*50)


if __name__ == "__main__":
    main()