# Benchmarking Retrieval Models on Forums: BM25 vs. Semantic vs. Hybrid Search

[Project Proposal](https://textdata.org/submissions/6aca700fa8567d79c2f70b63)
   
## Dataset
1. [AskUbuntu](https://github.com/taolei87/askubuntu)

## Setup
```
git clone https://github.com/taolei87/askubuntu.git data/askubuntu
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Semantic Search
`semantic_search.py` embeds each question (title + body) with a sentence-transformers model (`all-MiniLM-L6-v2` by default) and scores candidates by cosine similarity.

```
python semantic_search.py                    # reranks dev + test, writes results/semantic_{dev,test}.txt
python semantic_search.py --model sentence-transformers/all-mpnet-base-v2
python semantic_search.py --query "wifi not working after upgrade"   # search the full corpus
```

The output files use the same format as the BM25 files in the dataset (`dev.txt` / `test.txt`):
```
query_id \t similar_ids \t candidate_ids \t scores
```
The BM25 baseline from the dataset is included as `results/bm25_dev.txt` and `results/bm25_test.txt` (copied unchanged from the dataset's `dev.txt` / `test.txt`). Candidates are sorted by score, highest first, so any metric code written against the BM25 files works on these unchanged. For the hybrid approach, both files cover the same 20 candidates per query, so the scores can be combined per candidate. `data_loader.py` has helpers to load the corpus, read these files, and write rankings in this format.

Note: some queries have no similar questions among the candidates (11 in dev, 14 in test). These are kept in the output to match the BM25 files and are usually skipped during evaluation.

## Hybrid Search
`hybrid_search.py` combines the BM25 and semantic scores for the same 20 candidates per query.

```
python hybrid_search.py                  # weighted fusion, alpha tuned on dev
python hybrid_search.py --alpha 0.3      # fixed BM25 weight
python hybrid_search.py --method rrf     # reciprocal rank fusion (k=60)
```

- **Weighted:** `alpha * bm25 + (1 - alpha) * semantic`, with both scores min-max normalized per query. If `--alpha` isn't given, it tries 0.0 to 1.0 on dev and picks the best MAP.
- **RRF:** `1/(k + rank_bm25) + 1/(k + rank_semantic)`.

Output goes to `results/hybrid_{dev,test}.txt` and `results/hybrid_rrf_{dev,test}.txt` in the same format as the other results.
