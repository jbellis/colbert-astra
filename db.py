from typing import Any, Dict, List
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement

keyspace = "colbertv2"

class DB:
    def __init__(self, **kwargs):
        self.cluster = Cluster(**kwargs)
        self.session = self.cluster.connect()
        self.session.default_timeout = 60

        insert_chunk_cql = f"""
        INSERT INTO {keyspace}.chunks (title, part, body, ada002_embedding)
        VALUES (?, ?, ?, ?)
        """
        self.insert_chunk_stmt = self.session.prepare(insert_chunk_cql)

        insert_colbert_cql = f"""
        INSERT INTO {keyspace}.colbert_embeddings (title, part, embedding_id, bert_embedding)
        VALUES (?, ?, ?, ?)
        """
        self.insert_colbert_stmt = self.session.prepare(insert_colbert_cql)

        query_ada_cql = f"""
        SELECT title, body
        FROM {keyspace}.chunks
        ORDER BY ada002_embedding ANN OF ?
        LIMIT ?
        """
        self.query_ada_stmt = self.session.prepare(query_ada_cql)

        query_colbert_ann_cql = f"""
        SELECT title, part, bert_embedding
        FROM {keyspace}.colbert_embeddings
        ORDER BY bert_embedding ANN OF ?
        LIMIT ?
        """
        self.query_colbert_ann_stmt = self.session.prepare(query_colbert_ann_cql)

        query_colbert_parts_cql = f"""
        SELECT title, part, bert_embedding
        FROM {keyspace}.colbert_embeddings
        WHERE title = ? AND part = ?
        """
        self.query_colbert_parts_stmt = self.session.prepare(query_colbert_parts_cql)

        query_part_by_pk = f"""
        SELECT body
        FROM {keyspace}.chunks
        WHERE title = ? AND part = ?
        """
        self.query_part_by_pk_stmt = self.session.prepare(query_part_by_pk)
