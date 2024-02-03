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

from cassandra.concurrent import execute_concurrent_with_args
from db import DB

#
# Compute embeddings for articles in .json files, and load directly into Cassandra
#

thread_local_storage = threading.local()
def get_threadlocals():
    if not hasattr(thread_local_storage, 'db_handle'):
        thread_local_storage.db_handle = DB()

        cf = ColBERTConfig(checkpoint='checkpoints/colbertv2.0')
        cp = Checkpoint(cf.checkpoint, colbert_config=cf)
        thread_local_storage.encoder = CollectionEncoder(cf, cp)
    return thread_local_storage.db_handle, thread_local_storage.encoder


def load_file(full_path):
    db, encoder = get_threadlocals()

    # load json contents
    with open(full_path, 'r') as f:
        doc = json.load(f)
    title = doc['title']

    passages = []
    part = 0
    while str(part) in doc:
        content = doc[str(part)]['content']
        dpr = doc[str(part)]['embedding']
        db.session.execute(db.insert_chunk_stmt, (title, part, content, dpr))
        passages.append(content)
        part += 1
    embeddings_flat, counts = encoder.encode_passages(passages)

    # split up embeddings_flat by counts, a list of the number of tokens in each passage
    start_indices = [0] + list(itertools.accumulate(counts[:-1]))
    embeddings_by_part = [embeddings_flat[start:start+count] for start, count in zip(start_indices, counts)]
    for part, embeddings in enumerate(embeddings_by_part):
        execute_concurrent_with_args(db.session, db.insert_colbert_stmt, [(title, part, i, e) for i, e in enumerate(embeddings)])
    print(f'Loaded {title} in {part} parts and {len(embeddings_flat)} embeddings')


def main():
    print("Waiting for Cassandra schema")
    get_threadlocals() # let one thread create the table + index
    time.sleep(1)

    print("Inserting data")
    num_threads = 4
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        chunks_path = Path('../wiki50k/chunks')
        executor.map(load_file, chunks_path.iterdir())
    # for path in Path('../wiki50k/chunks').iterdir():
    #     load_file(path)

if __name__ == '__main__':
    main()
