import itertools
import os
import time
from typing import Dict, Tuple
from tqdm import tqdm

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from db import DB, db
from serve import retrieve_colbert
from colbert.infra.config import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.indexing.collection_encoder import CollectionEncoder
from cassandra.concurrent import execute_concurrent_with_args


def download_and_load_dataset(dataset: str = "scifact") -> Tuple[dict, dict, dict]:
    print("Downloading and loading dataset...")
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    print(f"Full dataset loaded. Corpus size: {len(corpus)}, Queries: {len(queries)}, Relevance judgments: {len(qrels)}")
    return corpus, queries, qrels


def compute_and_store_embeddings(corpus: dict, db: DB):
    print("Preparing corpus for ColBERT...")
    cf = ColBERTConfig(checkpoint='checkpoints/colbertv2.0')
    cp = Checkpoint(cf.checkpoint, colbert_config=cf)
    encoder = CollectionEncoder(cf, cp)

    print("Encoding and inserting documents...")
    start_time = time.time()
    for doc_id, doc in tqdm(corpus.items(), desc="Encoding and inserting"):
        title = doc['title']
        content = doc['text']

        embeddings_flat, counts = encoder.encode_passages([content])

        # split up embeddings_flat by counts, a list of the number of tokens in each passage
        start_indices = [0] + list(itertools.accumulate(counts[:-1]))
        embeddings_by_part = [embeddings_flat[start:start + count] for start, count in zip(start_indices, counts)]
        assert len(embeddings_by_part) == 1 # only one part
        embeddings = embeddings_by_part[0]

        future = db.session.execute_async(db.insert_chunk_stmt, (title, 0, content, None))
        execute_concurrent_with_args(db.session, db.insert_colbert_stmt, [(title, 0, i, e) for i, e in enumerate(embeddings)])
        future.result()

    end_time = time.time()
    print(f"Encoding and insertion completed. Time taken: {end_time - start_time:.2f} seconds")


def search_and_benchmark(queries: dict) -> Dict[str, Dict[str, float]]:
    def search(query: str, top_k: int = 100) -> Dict[str, float]:
        results = retrieve_colbert(query)
        return {result['title']: 1.0 / (i + 1) for i, result in enumerate(results[:top_k])}

    print("Retrieving results for all queries...")
    results = {}
    start_time = time.time()
    for query_id, query in tqdm(queries.items(), total=len(queries), desc="Retrieving"):
        results[query_id] = search(query)
    end_time = time.time()
    print(f"Retrieval completed. Time taken: {end_time - start_time:.2f} seconds")
    return results


def evaluate_model(qrels: dict, results: dict):
    print("Evaluating the model...")
    evaluator = EvaluateRetrieval()
    metrics = evaluator.evaluate(qrels, results, [10])
    metric_names = ["NDCG"]
    for metric_name, scores in zip(metric_names, metrics):
        print(f"{metric_name}:")
        for k, score in scores.items():
            print(f"  {k}: {score:.5f}")


def main():
    corpus, queries, qrels = download_and_load_dataset()
    compute_and_store_embeddings(corpus, db)
    results = search_and_benchmark(queries)
    evaluate_model(qrels, results)


if __name__ == "__main__":
    main()
