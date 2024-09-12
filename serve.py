from db import db
from openai import OpenAI
import torch
from datetime import datetime
from colbert.infra.config import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.modeling.colbert import ColBERT
from cassandra.concurrent import execute_concurrent_with_args
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster


_cf = ColBERTConfig(checkpoint='checkpoints/colbertv2.0')
_cp = Checkpoint(_cf.checkpoint, colbert_config=_cf)
ColBERT.try_load_torch_extensions(False) # enable segmented_maxsim even if gpu is detected

def pool_query_embeddings(query_embeddings, pool_factor=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_embeddings = query_embeddings.to(device)
    
    # Compute cosine similarity
    similarities = torch.mm(query_embeddings, query_embeddings.t())
    # Convert similarities to distances
    distances = 1 - similarities.cpu().numpy()
    # Create hierarchical clusters
    Z = linkage(distances, method='ward')

    # Determine the number of clusters
    max_clusters = max(query_embeddings.shape[0] // pool_factor, 1)
    cluster_labels = fcluster(Z, t=max_clusters, criterion='maxclust')
    
    # Pool embeddings within each cluster
    pooled_embeddings = []
    for cluster_id in range(1, max_clusters + 1):
        cluster_indices = torch.where(torch.tensor(cluster_labels == cluster_id, device=device))[0]
        if cluster_indices.numel() > 0:
            pooled_embedding = query_embeddings[cluster_indices].mean(dim=0)
            pooled_embeddings.append(pooled_embedding)
    
    return torch.stack(pooled_embeddings)

def encode(q):
    query_embeddings = _cp.queryFromText([q])[0]  # Get embeddings for a single query
    pooled_embeddings = pool_query_embeddings(query_embeddings)
    return pooled_embeddings.unsqueeze(0)  # Add batch dimension back


DENSE_MODEL = "text-embedding-3-small"
client = OpenAI(api_key=open('secrets/openai_key', 'r').read().splitlines()[0])

def ada_embedding_of(text: str):
    res = client.embeddings.create(
        input=[text],
        model=DENSE_MODEL
    )
    return res.data[0].embedding

def get_top_ids_ada(query, limit):
    qv = ada_embedding_of(query)
    rows = db.session.execute(db.query_ada_stmt, [qv, qv, limit])
    return {row.id: row.similarity for row in rows}


# construct a single-dimension tensor with all the embeddings packed into it
# i.e. a tensor of [d1.e1, d1.e2, d2.e1, d2.e2, d2.e3, dN.e1, .. dN.eM] for doc parts 1..N and embeddings 1..M
# since each doc can have a different number of embeddings, we also return a tensor of how many embeddings correspond to each doc
def load_data_and_construct_tensors(L, db):
    all_embeddings = []
    lengths = []

    results = execute_concurrent_with_args(db.session, db.query_colbert_parts_stmt, [(chunk_id,) for chunk_id in L])
    for success, result in results:
        if not success:
            raise Exception('Failed to retrieve embeddings')
        embeddings_for_chunk = [torch.tensor(row.bert_embedding) for row in result]
        packed_one_chunk = torch.stack(embeddings_for_chunk)
        all_embeddings.append(packed_one_chunk)
        lengths.append(packed_one_chunk.shape[0])

    D_packed = torch.cat(all_embeddings)
    D_lengths = torch.tensor(lengths, dtype=torch.long)

    return D_packed, D_lengths


MAX_ASTRA_LIMIT = 1000
def get_top_ids_colbert(query, n_ann_docs, n_colbert_candidates):
    Q = encode(query)
    query_encodings = Q[0]

    # compute the max score for each term for each doc
    chunks_per_query = {}
    for qv in query_encodings:
        qv = qv.cpu()
        qv_list = list(qv)
        limit = n_ann_docs
        rows = []
        while limit <= MAX_ASTRA_LIMIT:
            rows = list(db.session.execute(db.query_colbert_ann_stmt, [qv_list, qv_list, limit]))
            distinct_chunks = set(row.chunk_id for row in rows)
            if len(distinct_chunks) >= max(2, n_ann_docs / 4):
                break
            limit *= 2
        for row in rows:
            key = (row.chunk_id, qv)
            chunks_per_query[key] = max(chunks_per_query.get(key, -1), row.similarity)
    if not chunks_per_query:
        return {}  # empty database

    chunks = {}
    for (chunk_id, qv), similarity in chunks_per_query.items():
        chunks[chunk_id] = chunks.get(chunk_id, 0) + similarity
    candidates = sorted(chunks, key=chunks.get, reverse=True)[:n_colbert_candidates]

    # fully score each document in the top candidates
    Q = Q.squeeze(0).cpu() # transform Q from 2D to 1D
    D_packed, D_lengths = load_data_and_construct_tensors(candidates, db)
    # Calculate raw scores using matrix multiplication
    raw_scores = D_packed @ Q.to(dtype=D_packed.dtype).T
    # Apply optimized maxsim to obtain final per-document scores
    final_scores = ColBERT.segmented_maxsim(raw_scores, D_lengths)
    # map the flat list back to the document part keys
    scores = {chunk_id: final_scores[i].item() for i, chunk_id in enumerate(candidates)}
    return scores
    # Return the top n_docs chunk_ids with their scores
    # FIXME move this into retrieve_colbert
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_docs])


def retrieve_colbert(query, n_docs=5):
    top_chunks = get_top_ids_colbert(query, n_docs)
    if not top_chunks:
        return []  # empty database

    # load the source chunk for the top results
    results = execute_concurrent_with_args(db.session, db.query_part_by_pk_stmt, [(chunk_id,) for chunk_id in top_chunks.keys()])
    assert results
    if not all(success for success, _ in results):
        raise Exception('Failed to retrieve chunk')
    return [dict(_id=row.id, title=row.title, body=row.body, score=top_chunks[row.id])
            for row in (rs.one() for _, rs in results)]


def format_stdout(L):
    return '\n'.join(f"{i+1}. {row['title']}\n{row['body']}" for i, row in enumerate(L))

if __name__ == '__main__':
    while True:
        try:
            query = input('Enter a query: ')
        except (EOFError, KeyboardInterrupt):
            break
        print('\n# Retrieving from ColBERT #\n')
        print(format_stdout(retrieve_colbert(query)))

        # print('\n\n# Retrieving from ADA #\n')
        # print(format_stdout(retrieve_ada(query)))

        print()
