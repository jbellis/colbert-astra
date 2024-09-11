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
# !!! This was created for a different schema than the one created for BEIR
#

thread_local_storage = threading.local()
def get_threadlocals():
    if not hasattr(thread_local_storage, 'db_handle'):
        thread_local_storage.db_handle = DB()
    return thread_local_storage.db_handle


def load_file(compressed_path):
    db = get_threadlocals()

    loaded_path = compressed_path.with_suffix('.loaded')
    if loaded_path.exists():
        print(f'{compressed_path} already loaded')
        return

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

    # create an empty file to mark this as completed
    loaded_path.touch()
    print(f'Loaded {title} in {part} parts and {n_embeddings} embeddings')


def main():
    print("Waiting for Cassandra schema")
    get_threadlocals() # let one thread create the table + index
    time.sleep(1)

    print("Inserting data")
    chunks_path = Path('/home/jonathan/datasets/wiki50k-chunks')
    num_threads = 4 # this seems to be the ceiling of what we can leverage w/ GIL in the way
    # for compressed_path in chunks_path.iterdir():
    #     load_file(compressed_path)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks and hold their futures in a list
        L = [path for path in chunks_path.iterdir() if path.suffix == '.gz'][:1000]
        futures = [executor.submit(load_file, path) for path in L]
        # Iterate over the futures as they complete (whether successfully or due to exceptions)
        for future in as_completed(futures):
            try:
                # Accessing the result of the future will raise any exceptions caught during execution
                future.result()
            except Exception as exc:
                print(f'An exception occurred: {exc}')

if __name__ == '__main__':
    main()
