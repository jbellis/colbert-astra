import os
import time
from typing import Dict, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from openai import OpenAI
from cassandra.concurrent import execute_concurrent_with_args

from db import DB, db
from serve import get_top_chunk_ids


def download_and_load_dataset(dataset: str = "scifact") -> Tuple[dict, dict, dict]:
    print("Downloading and loading dataset...")
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="train")

    print(f"Dataset loaded. Corpus size: {len(corpus)}, Queries: {len(queries)}, Relevance judgments: {len(qrels)}")
    return corpus, queries, qrels


def compute_and_store_embeddings(corpus: dict, db: DB):
    client = OpenAI(api_key=open('secrets/openai_key', 'r').read().splitlines()[0])

    print("Computing embeddings and updating documents...")
    start_time = time.time()

    batch_size = 32
    all_doc_items = list(corpus.items())
    
    for i in tqdm(range(0, len(all_doc_items), batch_size), desc="Processing batches"):
        batch = all_doc_items[i:i+batch_size]
        
        # Compute embeddings for the batch
        texts = [doc['text'] for _, doc in batch]
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        embeddings = [item.embedding for item in response.data]

        # Execute updates concurrently
        ids = [doc_id for doc_id, _ in batch]
        execute_concurrent_with_args(db.session, db.update_dense_embedding_stmt,
                                     zip(embeddings, ids))

    end_time = time.time()
    print(f"Embedding computation and update completed. Time taken: {end_time - start_time:.2f} seconds")


def search_and_benchmark(queries: dict) -> Dict[str, Dict[str, float]]:
    def search(query_item: Tuple[str, str]) -> Tuple[str, Dict[str, float]]:
        query_id, query = query_item
        return (query_id, get_top_chunk_ids(query, 100))

    print("Retrieving results for all queries...")
    start_time = time.time()
    
    num_threads = 8
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = dict(tqdm(executor.map(search, queries.items()), total=len(queries), desc="Retrieving"))
    
    end_time = time.time()
    print(f"Retrieval completed. Time taken: {end_time - start_time:.2f} seconds")
    return results


def evaluate_model(qrels: dict, results: dict):
    print("Evaluating the model...")
    evaluator = EvaluateRetrieval()
    metrics = evaluator.evaluate(qrels, results, [10, 100])
    metric_names = ["NDCG"]
    for metric_name, scores in zip(metric_names, metrics):
        print(f"{metric_name}:")
        for k, score in scores.items():
            print(f"  {k}: {score:.5f}")


def main():
    corpus, queries, qrels = download_and_load_dataset()
    compute_and_store_embeddings(corpus, db)
    # results = search_and_benchmark(queries)
    # evaluate_model(qrels, results)


if __name__ == "__main__":
    main()
