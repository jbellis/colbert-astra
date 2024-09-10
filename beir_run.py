import os
from typing import Dict, List

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25

# Download and load a BEIR dataset
dataset = "msmarco"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(os.getcwd(), "datasets")
data_path = util.download_and_unzip(url, out_dir)
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# Initialize BM25 model
model = BM25(index_name=dataset)

# Index the corpus
model.index(corpus)

# Retrieve results (format of results is identical to qrels)
results = model.search(queries, k=100)

# Evaluate your model with NDCG@k, MAP@K ...
k_values = [1,3,5,10,100]
ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, k_values)

# Print top-k metrics
print("NDCG@k:", ndcg)
print("MAP@k:", _map)
print("Recall@k:", recall)
print("Precision@k:", precision)
