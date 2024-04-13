from db import DB
from openai import OpenAI
from torch import tensor
from colbert.infra.config import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint


db = DB()

_cf = ColBERTConfig(checkpoint='checkpoints/colbertv2.0')
_cp = Checkpoint(_cf.checkpoint, colbert_config=_cf)
encode = lambda q: _cp.queryFromText([q])[0]


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

def retrieve_colbert(query):
    K = 5
    query_encodings = encode(query)
    # find the most relevant documents
    docparts_per_query = {}
    for qv in query_encodings:
        rows = db.session.execute(db.query_colbert_ann_stmt, [list(qv), K])
        # set the value to the max of any existing value and the new one
        for row in rows:
            key = (row.title, row.part, qv)
            value = qv @ tensor(row.bert_embedding)
            docparts_per_query[key] = max(docparts_per_query.get(key, -1), value)

    # group by document part, summing the score
    docparts = {}
    for (title, part, qv), score in docparts_per_query.items():
        docparts[(title, part)] = docparts.get((title, part), 0) + score
    # top 2K document parts
    L = sorted(docparts, key=docparts.get, reverse=True)[:2*K]

    # fully score each document in the top 2K
    scores = {}
    for title, part in L:
        rows = db.session.execute(db.query_colbert_parts_stmt, [title, part])
        embeddings_for_part = [tensor(row.bert_embedding) for row in rows]
        scores[(title, part)] = sum(maxsim(qv, embeddings_for_part) for qv in query_encodings)
    # load the source chunk for the top K
    docs_by_score = sorted(scores, key=scores.get, reverse=True)[:K]
    L = []
    for title, part in docs_by_score:
        rs = db.session.execute(db.query_part_by_pk_stmt, [title, part])
        L.append({'title': title, 'body': rs.one().body})
    return L


def format_stdout(L):
    return '\n'.join(f"{i+1}. {row['title']}\n{row['body']}" for i, row in enumerate(L))

if __name__ == '__main__':
    while True:
        query = input('Enter a query: ')
        print('\n# Retrieving from ColBERT #\n')
        print(format_stdout(retrieve_colbert(query)))

        print('\n\n# Retrieving from ADA #\n')
        print(format_stdout(retrieve_ada(query)))

        print()
