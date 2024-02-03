import gzip
import itertools
import json
import random
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from tqdm.auto import tqdm

from colbert.infra.config import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.indexing.collection_encoder import CollectionEncoder

#
# Compute embeddings for articles in .json files, and save as .json.gz
#

thread_local_storage = threading.local()
def get_threadlocals():
    if not hasattr(thread_local_storage, 'db_handle'):
        cf = ColBERTConfig(checkpoint='checkpoints/colbertv2.0')
        cp = Checkpoint(cf.checkpoint, colbert_config=cf)
        thread_local_storage.encoder = CollectionEncoder(cf, cp)
    return thread_local_storage.encoder


def process_file(full_path: Path):
    encoder = get_threadlocals()

    compressed_path = full_path.with_suffix('.json.gz')
    if compressed_path.exists():
        print(f'{compressed_path} already exists')
        return

    # load json contents
    with open(full_path, 'r') as f:
        doc = json.load(f)
    title = doc['title']

    passages = []
    part = 0
    while str(part) in doc:
        passages.append(doc[str(part)]['content'])
        part += 1

    # encode the passages!
    embeddings_flat, counts = encoder.encode_passages(passages)

    # split up embeddings_flat by counts, a list of the number of tokens in each passage
    start_indices = [0] + list(itertools.accumulate(counts[:-1]))
    embeddings_by_part = [embeddings_flat[start:start+count] for start, count in zip(start_indices, counts)]
    for part, embeddings in enumerate(embeddings_by_part):
        doc[str(part)]['colbert_embedding'] = embeddings.tolist()

    # save the doc with compression
    with gzip.open(compressed_path, 'wt', encoding='UTF-8') as f:
        json.dump(doc, f)

    print(f'Created {len(embeddings_flat)} embeddings for {title} in {part} parts')


def main():
    print("Inserting data")
    num_threads = 8
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        chunks_path = Path('../wiki50k/chunks')
        executor.map(process_file, chunks_path.iterdir())

if __name__ == '__main__':
    main()
