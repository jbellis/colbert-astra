import os
import random
from typing import Dict, List
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import time

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

# Download and load the MSMARCO dataset
print("\n1. Downloading and loading dataset...")
dataset = "msmarco"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(os.getcwd(), "datasets")
data_path = util.download_and_unzip(url, out_dir)
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
print(f"Full dataset loaded. Corpus size: {len(corpus)}, Queries: {len(queries)}, Relevance judgments: {len(qrels)}")

# Create a smaller subset of the corpus while keeping all queries
print("\n2. Creating a corpus subset while keeping all queries...")
subset_size = 100000  # Target number of documents in the subset

# Collect all relevant document IDs from qrels
relevant_doc_ids = set()
for query_relevance in qrels.values():
    relevant_doc_ids.update(query_relevance.keys())

print(f"Number of relevant documents: {len(relevant_doc_ids)}")

# Ensure we include all relevant documents in our corpus subset
subset_corpus_ids = list(relevant_doc_ids)

# Add random documents until we reach the desired subset size
remaining_docs = subset_size - len(subset_corpus_ids)
if remaining_docs > 0:
    additional_docs = random.sample(list(set(corpus.keys()) - set(subset_corpus_ids)), remaining_docs)
    subset_corpus_ids.extend(additional_docs)
elif remaining_docs < 0:
    print(f"Warning: Number of relevant documents ({len(relevant_doc_ids)}) exceeds target subset size ({subset_size}).")
    print("Proceeding with all relevant documents.")

# Create the final corpus subset
subset_corpus = {doc_id: corpus[doc_id] for doc_id in subset_corpus_ids}

print(f"Subset created. Corpus size: {len(subset_corpus)}, Queries: {len(queries)}, Relevance judgments: {len(qrels)}")

# Prepare corpus for BM25
print("\n3. Preparing corpus subset for BM25...")
corpus_ids = list(subset_corpus.keys())
tokenized_corpus = [subset_corpus[doc_id]['text'].split() for doc_id in corpus_ids]
print(f"Corpus subset prepared. Total documents: {len(tokenized_corpus)}")

# Initialize BM25 model
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

# Retrieve results
print("\n5. Retrieving results for all queries...")
results = {}
start_time = time.time()
for query_id, query in tqdm(queries.items(), total=len(queries), desc="Retrieving"):
    results[query_id] = search(query)
end_time = time.time()
print(f"Retrieval completed. Time taken: {end_time - start_time:.2f} seconds")

# Evaluate the model
print("\n6. Evaluating the model...")
evaluator = EvaluateRetrieval()
metrics = evaluator.evaluate(qrels, results, [1, 3, 5, 10, 100])

# Calculate MRR
mrr = evaluator.evaluate_custom(qrels, results, [1, 10, 100], metric="mrr")

# Print metrics
print("\n7. Evaluation results:")
metric_names = ["NDCG", "MAP", "Recall", "Precision"]
for metric_name, scores in zip(metric_names, metrics):
    print(f"{metric_name}:")
    for k, score in scores.items():
        print(f"  {k}: {score:.5f}")

# Print MRR
print("\nMRR:")
for k, score in mrr.items():
    print(f"  MRR@{k}: {score:.5f}")

print("\nBM25 benchmarking on MSMARCO subset (all queries) completed.")
