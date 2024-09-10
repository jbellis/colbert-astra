import os
from typing import Any, Dict, List
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement

keyspace = "colbertv2"

class DB:
    def __init__(self, cluster):
        self.cluster = cluster
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
        SELECT title, part, similarity_dot_product(?, bert_embedding) as similarity
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
        SELECT title, body
        FROM {keyspace}.chunks
        WHERE title = ? AND part = ?
        """
        self.query_part_by_pk_stmt = self.session.prepare(query_part_by_pk)


astra_token = os.environ.get('ASTRA_DB_TOKEN')
if astra_token:
    print('Connecting to Astra')
    cwd = os.path.dirname(os.path.realpath(__file__))
    cloud_config = {
      'secure_connect_bundle': os.path.join(cwd, 'secrets', 'secure-connect-beir.zip')
    }
    auth_provider = PlainTextAuthProvider('token', astra_token)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    db = DB(cluster)
else:
    print('Connecting to local Cassandra')
    db = DB(Cluster())
