from typing import Any, Dict, List
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement


class DB:
    def __init__(self, **kwargs):
        self.cluster = Cluster(**kwargs)
        self.session = self.cluster.connect()
        self.session.default_timeout = 60

        insert_chunk_cql = f"""
        INSERT INTO colbert.chunks (title, part, body, ada002_embedding)
        VALUES (?, ?, ?, ?)
        """
        self.insert_chunk_stmt = self.session.prepare(insert_chunk_cql)

        insert_colbert_cql = f"""
        INSERT INTO colbert.colbert_embeddings (title, part, embedding_id, bert_embedding)
        VALUES (?, ?, ?, ?)
        """
        self.insert_colbert_stmt = self.session.prepare(insert_colbert_cql)
