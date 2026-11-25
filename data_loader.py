import gzip
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "askubuntu")


def load_corpus(path=None):
    path = path or os.path.join(DATA_DIR, "text_tokenized.txt.gz")
    corpus = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            qid = parts[0]
            title = parts[1] if len(parts) > 1 else ""
            body = parts[2] if len(parts) > 2 else ""
            corpus[qid] = (title, body)
    return corpus


def load_eval_file(split="dev"):
    """Reads dev.txt / test.txt into a list of dicts.

    Each line is: query_id \t similar_ids \t candidate_ids \t bm25_scores
    """
    path = os.path.join(DATA_DIR, f"{split}.txt")
    queries = []
    with open(path) as f:
        for line in f:
            qid, similar, candidates, scores = line.rstrip("\n").split("\t")
            queries.append({
                "query_id": qid,
                "similar": similar.split(),
                "candidates": candidates.split(),
                "scores": [float(s) for s in scores.split()],
            })
    return queries


def write_rankings(rankings, path):
    """Writes rankings in the same 4-column format as the BM25 dev/test files,
    with candidates sorted by score (highest first)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rankings:
            pairs = sorted(zip(r["candidates"], r["scores"]), key=lambda x: -x[1])
            cands = " ".join(c for c, _ in pairs)
            scores = " ".join(f"{s:.6f}" for _, s in pairs)
            f.write(f"{r['query_id']}\t{' '.join(r['similar'])}\t{cands}\t{scores}\n")
