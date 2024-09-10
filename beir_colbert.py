import os
import time
from typing import Dict
from tqdm import tqdm

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from db import DB
from serve import retrieve_colbert
from colbert.infra.config import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.indexing.collection_encoder import CollectionEncoder

print("\n1. Downloading and loading dataset...")
dataset = "scifact"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(os.getcwd(), "datasets")
data_path = util.download_and_unzip(url, out_dir)
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
print(f"Full dataset loaded. Corpus size: {len(corpus)}, Queries: {len(queries)}, Relevance judgments: {len(qrels)}")

print("\n2. Preparing corpus for ColBERT...")
db = DB()

# Initialize ColBERT encoder
cf = ColBERTConfig(checkpoint='checkpoints/colbertv2.0')
cp = Checkpoint(cf.checkpoint, colbert_config=cf)
encoder = CollectionEncoder(cf, cp)

print("\n3. Encoding and inserting documents...")
start_time = time.time()
for doc_id, doc in tqdm(corpus.items(), desc="Encoding and inserting"):
    title = doc['title']
    content = doc['text']
    
    # Encode the document
    embeddings, _ = encoder.encode_passages([content])
    
    # Insert into database
    db.session.execute(db.insert_chunk_stmt, (title, 0, content, None))  # Using part 0 for single-part documents
    db.session.execute_async(db.insert_colbert_stmt, [(title, 0, i, e) for i, e in enumerate(embeddings[0])])

end_time = time.time()
print(f"Encoding and insertion completed. Time taken: {end_time - start_time:.2f} seconds")

# Function to retrieve top-k documents for a query
def search(query: str, top_k: int = 100) -> Dict[str, float]:
    results = retrieve_colbert(query)
    return {result['title']: 1.0 / (i + 1) for i, result in enumerate(results[:top_k])}

print("\n4. Retrieving results for all queries...")
results = {}
start_time = time.time()
for query_id, query in tqdm(queries.items(), total=len(queries), desc="Retrieving"):
    results[query_id] = search(query)
end_time = time.time()
print(f"Retrieval completed. Time taken: {end_time - start_time:.2f} seconds")

print("\n5. Evaluating the model...")
evaluator = EvaluateRetrieval()
metrics = evaluator.evaluate(qrels, results, [10])
metric_names = ["NDCG"]
for metric_name, scores in zip(metric_names, metrics):
    print(f"{metric_name}:")
    for k, score in scores.items():
        print(f"  {k}: {score:.5f}")
