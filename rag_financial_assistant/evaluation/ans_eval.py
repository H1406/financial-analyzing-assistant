# evaluation/evaluate.py

from rag.rag_pipeline import RAGPipeline
import re


def extract_number(text):
    text = text.replace(",", "").replace("$", "")
    return float(re.findall(r"-?\d+\.?\d*", text)[0])


def numeric_accuracy(pred, gt, tolerance = 1e-1):
    try:
        p = extract_number(pred)
        g = extract_number(gt)
        if p is None or g is None:
            return 0
        return 1 if abs(p - g) <= tolerance else 0
    except:
        return 0


def exact_match(pred, gt):
    return int(pred.strip().lower() == gt.strip().lower())


def evaluate(dataset):

    rag = RAGPipeline()

    results = []

    for sample in dataset:

        query = sample["question"]
        gt = sample["answer"]

        pred, contexts = rag.run(query)

        em = exact_match(pred, gt)
        num_acc = numeric_accuracy(pred, gt)

        results.append({
            "query": query,
            "pred": pred,
            "gt": gt,
            "exact_match": em,
            "numeric_accuracy": num_acc,
            "retrieved_sources": contexts
        })

    # aggregate
    avg_em = sum(r["exact_match"] for r in results) / len(results)
    avg_num = sum(r["numeric_accuracy"] for r in results) / len(results)

    print("\n=== Evaluation Results ===")
    print(f"Exact Match: {avg_em:.3f}")
    print(f"Numeric Accuracy: {avg_num:.3f}")

    return results
