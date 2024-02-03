import gzip
import itertools
import json
import random
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.auto import tqdm

from cassandra.concurrent import execute_concurrent_with_args
from db import DB

#
# Load .json.gz files into Cassandra
#

thread_local_storage = threading.local()
def get_threadlocals():
    if not hasattr(thread_local_storage, 'db_handle'):
        thread_local_storage.db_handle = DB()
    return thread_local_storage.db_handle


def load_file(compressed_path):
    db = get_threadlocals()

    # load json contents
    with gzip.open(compressed_path, 'rt', encoding='UTF-8') as f:
        doc = json.load(f)
    title = doc['title']

    part = 0
    n_embeddings = 0
    while str(part) in doc:
        content = doc[str(part)]['content']
        dpr = doc[str(part)]['embedding']
        db.session.execute(db.insert_chunk_stmt, (title, part, content, dpr))
        embeddings = doc[str(part)]['colbert_embedding']
        execute_concurrent_with_args(db.session, db.insert_colbert_stmt, [(title, part, i, e) for i, e in enumerate(embeddings)])
        part += 1
        n_embeddings += len(embeddings)

    print(f'Loaded {title} in {part} parts and {n_embeddings} embeddings')


def main():
    print("Waiting for Cassandra schema")
    get_threadlocals() # let one thread create the table + index
    time.sleep(1)

    print("Inserting data")
    num_threads = 16
    chunks_path = Path('/home/jonathan/datasets/wiki50k-chunks')
    # for compressed_path in chunks_path.iterdir():
    #     load_file(compressed_path)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks and hold their futures in a list
        futures = [executor.submit(load_file, path) for path in chunks_path.iterdir()]

        # Iterate over the futures as they complete (whether successfully or due to exceptions)
        for future in as_completed(futures):
            try:
                # Accessing the result of the future will raise any exceptions caught during execution
                future.result()
            except Exception as exc:
                print(f'An exception occurred: {exc}')

if __name__ == '__main__':
    main()
