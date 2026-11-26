import argparse
import os

from data_loader import write_rankings


def read_rankings(path):
    rankings = {}
    with open(path) as f:
        for line in f:
            qid, similar, candidates, scores = line.rstrip("\n").split("\t")
            rankings[qid] = {
                "similar": similar.split(),
                "scores": dict(zip(candidates.split(), map(float, scores.split()))),
            }
    return rankings


def min_max(scores):
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {c: 0.0 for c in scores}
    return {c: (s - lo) / (hi - lo) for c, s in scores.items()}


def ranks(scores):
    order = sorted(scores, key=lambda c: -scores[c])
    return {c: i + 1 for i, c in enumerate(order)}


def weighted_fusion(bm25, semantic, alpha):
    b, s = min_max(bm25), min_max(semantic)
    return {c: alpha * b[c] + (1 - alpha) * s[c] for c in bm25}


def rrf_fusion(bm25, semantic, k=60):
    rb, rs = ranks(bm25), ranks(semantic)
    return {c: 1 / (k + rb[c]) + 1 / (k + rs[c]) for c in bm25}


def fuse(bm25_runs, semantic_runs, method="weighted", alpha=0.5, k=60):
    results = []
    for qid, b in bm25_runs.items():
        s = semantic_runs[qid]["scores"]
        if method == "rrf":
            scores = rrf_fusion(b["scores"], s, k)
        else:
            scores = weighted_fusion(b["scores"], s, alpha)
        results.append({
            "query_id": qid,
            "similar": b["similar"],
            "candidates": list(scores),
            "scores": list(scores.values()),
        })
    return results


def mean_ap(results):
    # only used to pick alpha on dev
    total, n = 0.0, 0
    for r in results:
        relevant = set(r["similar"])
        if not relevant:
            continue
        order = sorted(zip(r["candidates"], r["scores"]), key=lambda x: -x[1])
        hits, ap = 0, 0.0
        for i, (c, _) in enumerate(order, 1):
            if c in relevant:
                hits += 1
                ap += hits / i
        total += ap / hits
        n += 1
    return total / n


def tune_alpha(results_dir):
    bm25 = read_rankings(os.path.join(results_dir, "bm25_dev.txt"))
    semantic = read_rankings(os.path.join(results_dir, "semantic_dev.txt"))
    best_alpha, best_map = 0.0, -1
    for i in range(11):
        alpha = i / 10
        score = mean_ap(fuse(bm25, semantic, "weighted", alpha))
        print(f"alpha={alpha:.1f}  dev MAP={score:.4f}")
        if score > best_map:
            best_alpha, best_map = alpha, score
    return best_alpha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["weighted", "rrf"], default="weighted")
    parser.add_argument("--alpha", type=float, help="BM25 weight; tuned on dev if not given")
    parser.add_argument("--k", type=int, default=60)
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    alpha = args.alpha
    if args.method == "weighted" and alpha is None:
        alpha = tune_alpha(args.results_dir)
        print(f"using alpha={alpha:.1f}")

    for split in args.splits:
        bm25 = read_rankings(os.path.join(args.results_dir, f"bm25_{split}.txt"))
        semantic = read_rankings(os.path.join(args.results_dir, f"semantic_{split}.txt"))
        results = fuse(bm25, semantic, args.method, alpha, args.k)
        suffix = "_rrf" if args.method == "rrf" else ""
        out = os.path.join(args.results_dir, f"hybrid{suffix}_{split}.txt")
        write_rankings(results, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
