from db import db
from openai import OpenAI
import torch
from colbert.infra.config import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.modeling.colbert import ColBERT


_cf = ColBERTConfig(checkpoint='checkpoints/colbertv2.0')
_cp = Checkpoint(_cf.checkpoint, colbert_config=_cf)
ColBERT.try_load_torch_extensions(False) # enable segmented_maxsim even if gpu is detected
encode = lambda q: _cp.queryFromText([q])


client = OpenAI(api_key=open('openai.key', 'r').read().splitlines()[0])
def ada_embedding_of(text: str):
    res = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return res.data[0].embedding

def retrieve_ada(query):
    qv = ada_embedding_of(query)
    rows = db.session.execute(db.query_ada_stmt, [qv])
    return [{'title': row.title, 'body': row.body} for row in rows]


def maxsim(qv, embeddings):
    return max(qv @ e for e in embeddings)

# construct a single-dimension tensor with all the embeddings packed into it
# i.e. a tensor of [d1.e1, d1.e2, d2.e1, d2.e2, d2.e3, dN.e1, .. dN.eM] for doc parts 1..N and embeddings 1..M
# since each doc can have a different number of embeddings, we also return a tensor of how many embeddings correspond to each doc
def load_data_and_construct_tensors(L, db):
    all_embeddings = []
    lengths = []

    for title, part in L:
        rows = db.session.execute(db.query_colbert_parts_stmt, [title, part])
        embeddings_for_part = [torch.tensor(row.bert_embedding) for row in rows]
        packed_one_part = torch.stack(embeddings_for_part)
        all_embeddings.append(packed_one_part)
        lengths.append(packed_one_part.shape[0])

    D_packed = torch.cat(all_embeddings)
    D_lengths = torch.tensor(lengths, dtype=torch.long)

    return D_packed, D_lengths


MAX_ASTRA_LIMIT = 1000
def retrieve_colbert(query):
    n_docs = 5 # number of documents that we return
    Q = encode(query)
    query_encodings = Q[0]

    # compute the max score for each term for each doc
    docparts_per_query = {}
    # the number of results per term, k, should be larger than n_docs,
    # but the larger n_docs is the less we need to expand it
    raw_k = max(1.0, 0.979 + 4.021 * n_docs ** 0.761) # f(1) = 5.0, f(100) = 1.1, f(1000) = 1.0
    k = min(MAX_ASTRA_LIMIT, int(raw_k))
    for qv in query_encodings:
        qv = qv.cpu()
        # loop until we find at least K/2 distinct values
        limit = k
        rows = []
        while limit <= min(MAX_ASTRA_LIMIT, 8*n_docs):
            # pull resultset into a list so we can iterate over it twice
            rows = list(db.session.execute(db.query_colbert_ann_stmt, [list(qv), limit]))
            distinct_documents = set((row.title, row.part) for row in rows) # this needs to use the primary key
            if len(distinct_documents) >= k / 2:
                break
            limit *= 2
        # record the scores
        for row in rows:
            key = (row.title, row.part, qv)
            value = qv @ torch.tensor(row.bert_embedding)
            docparts_per_query[key] = max(docparts_per_query.get(key, -1), value)
    if not docparts_per_query:
        return [] # empty database

    # group by document part, summing the score
    docparts = {}
    for (title, part, qv), score in docparts_per_query.items():
        docparts[(title, part)] = docparts.get((title, part), 0) + score
    # limit the candidates for the full, expensive scoring to 2x the count requested
    candidates = sorted(docparts, key=docparts.get, reverse=True)[:2*n_docs]

    # fully score each document in the top candidates
    scores = {}
    Q = Q.squeeze(0).cpu() # transform Q from 2D to 1D
    D_packed, D_lengths = load_data_and_construct_tensors(candidates, db)
    # Calculate raw scores using matrix multiplication
    raw_scores = D_packed @ Q.to(dtype=D_packed.dtype).T
    # Apply optimized maxsim to obtain final per-document scores
    final_scores = ColBERT.segmented_maxsim(raw_scores, D_lengths)
    # map the flat list back to the document part keys
    for i, (title, part) in enumerate(candidates):
        scores[(title, part)] = final_scores[i]

    # load the source chunk for the top results
    docs_by_score = sorted(scores, key=scores.get, reverse=True)[:k]
    results = []
    for title, part in docs_by_score:
        rs = db.session.execute(db.query_part_by_pk_stmt, [title, part])
        results.append({'title': title, 'body': rs.one().body})
    return results


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
