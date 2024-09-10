import os
import random
from typing import Dict, List
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import time

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

print("\n1. Downloading and loading dataset...")
dataset = "scifact"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(os.getcwd(), "datasets")
data_path = util.download_and_unzip(url, out_dir)
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test") # train also available
print(f"Full dataset loaded. Corpus size: {len(corpus)}, Queries: {len(queries)}, Relevance judgments: {len(qrels)}")

print("\n2. Preparing full corpus for BM25...")
corpus_ids = list(corpus.keys())
tokenized_corpus = [corpus[doc_id]['text'].split() for doc_id in corpus_ids]
print(f"Full corpus prepared. Total documents: {len(tokenized_corpus)}")

print("\n4. Initializing BM25 model...")
start_time = time.time()
bm25 = BM25Okapi(tokenized_corpus)
end_time = time.time()
print(f"BM25 model initialized. Time taken: {end_time - start_time:.2f} seconds")

# Function to retrieve top-k documents for a query
def search(query: str, top_k: int = 100) -> Dict[str, float]:
    tokenized_query = query.split()
    doc_scores = bm25.get_scores(tokenized_query)
    top_k_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
    return {corpus_ids[i]: float(doc_scores[i]) for i in top_k_indices}

print("\n5. Retrieving results for all queries...")
results = {}
start_time = time.time()
for query_id, query in tqdm(queries.items(), total=len(queries), desc="Retrieving"):
    results[query_id] = search(query)
end_time = time.time()
print(f"Retrieval completed. Time taken: {end_time - start_time:.2f} seconds")

print("\n6. Evaluating the model...")
evaluator = EvaluateRetrieval()
metrics = evaluator.evaluate(qrels, results, [1, 3, 5, 10, 100])

# Calculate MRR
mrr = evaluator.evaluate_custom(qrels, results, [1, 10, 100], metric="mrr")

print("\n6. Evaluation results:")
metric_names = ["NDCG", "MAP", "Recall", "Precision"]
for metric_name, scores in zip(metric_names, metrics):
    print(f"{metric_name}:")
    for k, score in scores.items():
        print(f"  {k}: {score:.5f}")
print("\nMRR:")
for k, score in mrr.items():
    print(f"  MRR@{k}: {score:.5f}")
