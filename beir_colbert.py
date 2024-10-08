import itertools
import os
import time
import threading
from typing import Dict, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from db import DB, db
from serve import get_top_ids_colbert
from colbert.infra.config import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.indexing.collection_encoder import CollectionEncoder
from cassandra.concurrent import execute_concurrent_with_args


def download_and_load_dataset(dataset: str = "scifact") -> Tuple[dict, dict, dict]:
    print("Downloading and loading dataset...")
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    corpus, all_queries, all_qrels = GenericDataLoader(data_folder=data_path).load(split="train")
    # Limit queries to the first 100 items
    queries = dict(list(all_queries.items())[:100])
    qrels = {qid: all_qrels[qid] for qid in queries.keys() if qid in all_qrels}

    print(f"Dataset loaded. Corpus size: {len(corpus)}, Queries: {len(queries)}, Relevance judgments: {len(qrels)}")
    return corpus, queries, qrels


thread_local_storage = threading.local()

def get_threadlocals():
    if not hasattr(thread_local_storage, 'encoder'):
        cf = ColBERTConfig(checkpoint='checkpoints/colbertv2.0')
        cp = Checkpoint(cf.checkpoint, colbert_config=cf)
        thread_local_storage.encoder = CollectionEncoder(cf, cp)
    return thread_local_storage.encoder


def process_document(doc_item):
    encoder = get_threadlocals()
    doc_id, doc = doc_item
    title = doc['title']
    content = doc['text']

    embeddings_flat, counts = encoder.encode_passages([content])

    # split up embeddings_flat by counts, a list of the number of tokens in each passage
    start_indices = [0] + list(itertools.accumulate(counts[:-1]))
    embeddings_by_part = [embeddings_flat[start:start + count] for start, count in zip(start_indices, counts)]
    assert len(embeddings_by_part) == 1  # only one part (chunk)
    embeddings = embeddings_by_part[0]

    # Use the _id from the BEIR corpus as the chunk_id
    future = db.session.execute_async(db.insert_chunk_stmt, ((doc_id), title, content, None))
    execute_concurrent_with_args(db.session, db.insert_colbert_stmt,
                                 [((doc_id), i, e) for i, e in enumerate(embeddings)])
    future.result()


def compute_and_store_embeddings(corpus: dict, db: DB):
    print("Encoding and inserting documents...")
    start_time = time.time()
    
    num_threads = 3  # vram-limited :(
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(process_document, corpus.items()), total=len(corpus), desc="Encoding and inserting"))

    end_time = time.time()
    print(f"Encoding and insertion completed. Time taken: {end_time - start_time:.2f} seconds")


def search_and_benchmark(queries: dict, n_ann_docs: int, n_colbert_candidates: int) -> Dict[str, Dict[str, float]]:
    def search(query_item: Tuple[str, str]) -> Tuple[str, Dict[str, float]]:
        query_id, query = query_item
        return (query_id, get_top_ids_colbert(query, n_ann_docs, n_colbert_candidates))

    start_time = time.time()
    num_threads = 8
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = dict(tqdm(executor.map(search, queries.items()), total=len(queries), desc="Retrieving"))
    end_time = time.time()

    print(f"  Time: {end_time - start_time:.2f} seconds")
    return results


def evaluate_model(qrels: dict, results: dict):
    evaluator = EvaluateRetrieval()
    metrics = evaluator.evaluate(qrels, results, [1, 5, 10, 100])
    metric_names = ["NDCG"]
    for metric_name, scores in zip(metric_names, metrics):
        for k, score in scores.items():
            print(f"  {k}: {score:.5f}")


def main():
    corpus, queries, qrels = download_and_load_dataset()
    # compute_and_store_embeddings(corpus, db)
    for n_ann_docs in [20, 40, 80, 160, 240, 320]:
        for n_colbert_candidates in [20, 40, 80, 160, 240, 320]:
            print(f"ANN,COLBERT: {n_ann_docs}, {n_colbert_candidates}")
            results = search_and_benchmark(queries, n_ann_docs, n_colbert_candidates)
            evaluate_model(qrels, results)


if __name__ == "__main__":
    main()
