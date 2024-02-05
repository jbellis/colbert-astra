# Colbert on Astra #

POC of ColBERT search, compared with vanilla DPR.

# Requirements #

* Assumes you have Cassandra (vsearch branch) running locally.  Should "just work" with Astra given minor changes to db.py.
* Dataset with DPR ada002 embeddings already computed, this code does not do that (but adding it would just be a few lines)

# Usage #

1. cqlsh < create.cqlsh

2a. hack up compute-and-load.py to load your chunks.  currently it expects json files that look like this:
  {
    'title': $title,
    '0': { 'content': $raw_text, 'embedding': $ada002_embedding },
    '1': { 'content': $raw_text, 'embedding': $ada002_embedding },
    ...
  }

  If you don't have pre-chunked documents, or you don't have or don't want to save a single dense embedding for comparison,
  then adjust it accordingly.

2b. alternatively, hack up compute.py and load.py instead.  `compute` computes the colbert embeddings and augments the json file with them, and `load` sends those to Cassandra.  I did this because I wanted to compute the embeddings on a fast gpu machine.

3. `python serve_httpy.py` and navigate to http://localhost:5000
