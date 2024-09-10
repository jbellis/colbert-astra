import os
from typing import Dict, List
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import time

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

# Download and load a BEIR dataset
print("\n1. Downloading and loading dataset...")
dataset = "msmarco"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(os.getcwd(), "datasets")
data_path = util.download_and_unzip(url, out_dir)
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
print(f"Dataset loaded. Corpus size: {len(corpus)}, Queries: {len(queries)}, Relevance judgments: {len(qrels)}")

# Prepare corpus for BM25
print("\n2. Preparing corpus for BM25...")
corpus_ids = list(corpus.keys())
tokenized_corpus = [corpus[doc_id]['text'].split() for doc_id in corpus_ids]
print(f"Corpus prepared. Total documents: {len(tokenized_corpus)}")

# Initialize BM25 model
print("\n3. Initializing BM25 model...")
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

# Retrieve results
print("\n4. Retrieving results for queries...")
results = {}
start_time = time.time()
for query_id, query in tqdm(queries.items(), total=len(queries), desc="Retrieving"):
    results[query_id] = search(query)
end_time = time.time()
print(f"Retrieval completed. Time taken: {end_time - start_time:.2f} seconds")

# Evaluate the model
print("\n5. Evaluating the model...")
evaluator = EvaluateRetrieval()
metrics = evaluator.evaluate(qrels, results, [1, 3, 5, 10, 100])

# Print metrics
print("\n6. Evaluation results:")
for metric, scores in metrics.items():
    print(f"{metric}:")
    for k, score in scores.items():
        print(f"  @{k}: {score:.4f}")

print("\nBM25 benchmarking completed.")