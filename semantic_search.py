import argparse
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from data_loader import load_corpus, load_eval_file, write_rankings

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SemanticSearcher:
    def __init__(self, corpus, model_name=DEFAULT_MODEL, batch_size=64):
        self.corpus = corpus
        self.model = SentenceTransformer(model_name, device=get_device())
        self.batch_size = batch_size
        self.cache = {}
        self.index_ids = None
        self.index_vecs = None

    def doc_text(self, qid):
        title, body = self.corpus[qid]
        return f"{title} . {body}"

    def encode(self, texts):
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        )

    def embed_ids(self, ids):
        missing = [i for i in dict.fromkeys(ids) if i not in self.cache]
        if missing:
            vecs = self.encode([self.doc_text(i) for i in missing])
            self.cache.update(zip(missing, vecs))
        return np.stack([self.cache[i] for i in ids])

    def rerank(self, query_id, candidate_ids):
        q = self.embed_ids([query_id])[0]
        docs = self.embed_ids(candidate_ids)
        return (docs @ q).tolist()

    def run_split(self, split):
        queries = load_eval_file(split)
        all_ids = [q["query_id"] for q in queries] + [c for q in queries for c in q["candidates"]]
        self.embed_ids(all_ids)

        results = []
        for q in queries:
            scores = self.rerank(q["query_id"], q["candidates"])
            results.append({
                "query_id": q["query_id"],
                "similar": q["similar"],
                "candidates": q["candidates"],
                "scores": scores,
            })
        return results

    def build_index(self, cache_path=None):
        if cache_path and os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=False)
            self.index_ids = data["ids"].tolist()
            self.index_vecs = data["vecs"]
            return
        self.index_ids = list(self.corpus.keys())
        self.index_vecs = self.encode([self.doc_text(i) for i in self.index_ids])
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            np.savez(cache_path, ids=np.array(self.index_ids), vecs=self.index_vecs)

    def search(self, text, k=10):
        """Free-text search over the whole corpus. Call build_index() first."""
        q = self.encode([text.lower()])[0]
        scores = self.index_vecs @ q
        top = np.argsort(-scores)[:k]
        return [(self.index_ids[i], float(scores[i])) for i in top]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--query", help="run a free-text search over the full corpus instead")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    corpus = load_corpus()
    searcher = SemanticSearcher(corpus, args.model)

    if args.query:
        name = args.model.split("/")[-1]
        searcher.build_index(os.path.join(args.out_dir, f"index_{name}.npz"))
        for qid, score in searcher.search(args.query, args.k):
            print(f"{score:.4f}  [{qid}]  {corpus[qid][0]}")
        return

    for split in args.splits:
        results = searcher.run_split(split)
        out = os.path.join(args.out_dir, f"semantic_{split}.txt")
        write_rankings(results, out)
        print(f"wrote {out} ({len(results)} queries)")


if __name__ == "__main__":
    main()
