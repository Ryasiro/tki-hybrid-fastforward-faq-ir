"""Evaluasi reproduksibel pada dataset_pedoman_akademik.json (43 entitas: 35 FAQ + 8 Passage).

Protokol ilmiah:
1. Pemisahan tegas dev (15 kueri) untuk penyetelan alpha dan test (28 kueri tak terlihat) untuk laporan final.
2. Evaluasi CPU nyata menggunakan sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.
3. Verifikasi 0 selisih peringkat Safe Early Stopping vs Exhaustive pada seluruh kueri test.
4. Benchmark latensi CPU (warm-up + pengulangan) dengan laporan p50/p95.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from faq_core import HybridFAQ, MODEL_NAME

ROOT = Path(__file__).resolve().parent


def compute_metrics(hits, relevance: dict[str, int], *, k: int) -> dict[str, float]:
    """Metrik IR per kueri: Recall@k, MRR@k, AP@k (penyebut min(k, |rel|)), nDCG@k."""
    if not relevance or max(relevance.values()) <= 0:
        raise ValueError("Ground-truth wajib memiliki >=1 dokumen relevan.")
    predicted = [x.id for x in hits[:k]]
    relevant = {d for d, g in relevance.items() if g > 0}
    found = [1 if d in relevant else 0 for d in predicted]
    recall = len(relevant.intersection(predicted)) / len(relevant)
    rank = next((i + 1 for i, v in enumerate(found) if v), None)
    rr = 1.0 / rank if rank is not None else 0.0
    cum, ap_sum = 0, 0.0
    for i, v in enumerate(found, 1):
        if v:
            cum += 1
            ap_sum += cum / i
    ap_denom = min(k, len(relevant))
    ap = ap_sum / ap_denom if ap_denom > 0 else 0.0

    def dcg(grades: list[int]) -> float:
        return float(sum((2**g - 1) / np.log2(i + 2) for i, g in enumerate(grades)))

    dcg_val = dcg([relevance.get(d, 0) for d in predicted])
    idcg_val = dcg(sorted([g for g in relevance.values() if g > 0], reverse=True)[:k])
    return {
        "Recall@k": float(recall),
        "MRR@k": float(rr),
        "AP@k": float(ap),
        "nDCG@k": float(dcg_val / idcg_val) if idcg_val > 0 else 0.0,
    }


def evaluate_config(
    engine: HybridFAQ,
    cases: list[dict],
    *,
    alpha: float,
    k: int = 5,
    early_stopping: bool = True,
    query_vecs: dict[str, np.ndarray] | None = None,
) -> tuple[dict, list[dict]]:
    individual = []
    for case in cases:
        qvec = query_vecs.get(case["qid"]) if query_vecs is not None else None
        res = engine.search(
            case["query"],
            alpha=float(alpha),
            top_k=k,
            early_stopping=early_stopping,
            precomputed_query_vec=qvec,
        )
        metrics = compute_metrics(res.hits, case["relevance"], k=k)
        individual.append(
            {
                "qid": case["qid"],
                "split": case.get("split", ""),
                "category": case.get("category", ""),
                "query": case["query"],
                "alpha": float(alpha),
                "delta": engine.delta,
                "early_stopping": early_stopping,
                "target": ",".join(case["relevance"].keys()),
                "ranking": " > ".join(x.id for x in res.hits),
                "dense_lookups": res.evaluated_dense,
                "total_docs": res.total_documents,
                "lookup_saving_pct": 100.0 * (1.0 - res.evaluated_dense / res.total_documents)
                if alpha < 1
                else 100.0,
                "stopped_early": res.stopped_early,
                **metrics,
            }
        )
    n_docs = len(engine.corpus)
    avg_lookups = float(np.mean([r["dense_lookups"] for r in individual]))
    summary = {
        "alpha": float(alpha),
        "delta": engine.delta,
        "early_stopping": early_stopping,
        "n_queries": len(cases),
        "k": k,
        "raw_passages": engine.raw_passage_count,
        "coalesced_passages": engine.coalesced_count,
        "compression_pct": round(100.0 * engine.compression_ratio, 4),
        "Recall@k": float(np.mean([r["Recall@k"] for r in individual])),
        "MRR@k": float(np.mean([r["MRR@k"] for r in individual])),
        "MAP@k": float(np.mean([r["AP@k"] for r in individual])),
        "nDCG@k": float(np.mean([r["nDCG@k"] for r in individual])),
        "mean_dense_lookups": avg_lookups,
        "lookup_saving_pct": round(100.0 * (1.0 - avg_lookups / n_docs), 4) if alpha < 1 else 100.0,
        "early_stop_rate_pct": round(100.0 * float(np.mean([1.0 if r["stopped_early"] else 0.0 for r in individual])), 2),
    }
    return summary, individual


def benchmark_cpu_latency(
    engine: HybridFAQ,
    cases: list[dict],
    *,
    alpha: float,
    k: int = 5,
    early_stopping: bool = True,
    warmup: int = 10,
    repeats: int = 8,
) -> dict[str, float | int | bool]:
    for _ in range(warmup):
        engine.search(cases[0]["query"], alpha=alpha, top_k=k, early_stopping=early_stopping)
    core_ms, enc_ms, rank_ms = [], [], []
    for _ in range(repeats):
        for case in cases:
            res = engine.search(case["query"], alpha=alpha, top_k=k, early_stopping=early_stopping)
            core_ms.append(res.time_total_core_ms)
            enc_ms.append(res.time_query_encoder_ms)
            rank_ms.append(max(0.0, res.time_total_core_ms - res.time_query_encoder_ms))
    return {
        "alpha": float(alpha),
        "delta": engine.delta,
        "top_k": int(k),
        "early_stopping": bool(early_stopping),
        "warmup_runs": int(warmup),
        "repeats_per_query": int(repeats),
        "total_measured_queries": len(core_ms),
        "core_p50_ms": float(np.percentile(core_ms, 50)),
        "core_p95_ms": float(np.percentile(core_ms, 95)),
        "core_mean_ms": float(np.mean(core_ms)),
        "encoder_p50_ms": float(np.percentile(enc_ms, 50)),
        "encoder_p95_ms": float(np.percentile(enc_ms, 95)),
        "ranking_only_p50_ms": float(np.percentile(rank_ms, 50)),
        "ranking_only_p95_ms": float(np.percentile(rank_ms, 95)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "hasil_evaluasi"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--delta", type=float, default=0.035)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    import sentence_transformers
    import torch

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    encoder = SentenceTransformer(MODEL_NAME, device="cpu")
    corpus = json.loads((ROOT / "dataset_pedoman_akademik.json").read_text(encoding="utf-8"))
    all_queries = json.loads((ROOT / "queries_eval.json").read_text(encoding="utf-8"))
    dev_queries = [q for q in all_queries if q["split"] == "dev"]
    test_queries = [q for q in all_queries if q["split"] == "test"]

    engine = HybridFAQ(corpus, encoder, delta=args.delta, max_passage_words=120, negation_safe=False)
    q_embs = np.asarray(
        encoder.encode([q["query"] for q in all_queries], normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )
    query_vecs = {q["qid"]: q_embs[i] for i, q in enumerate(all_queries)}

    # 1. Dev sweep alpha
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    dev_alpha_rows = []
    for a in alphas:
        summ, _ = evaluate_config(engine, dev_queries, alpha=a, k=args.top_k, early_stopping=True, query_vecs=query_vecs)
        dev_alpha_rows.append(summ)

    hybrid_dev = [r for r in dev_alpha_rows if 0.0 < r["alpha"] < 1.0]
    best_dev = max(hybrid_dev, key=lambda r: (r["nDCG@k"], r["MRR@k"], r["Recall@k"], -abs(r["alpha"] - 0.5)))
    best_alpha = float(best_dev["alpha"])

    # 2. Test evaluation (28 unseen queries)
    neg_engine = HybridFAQ(
        corpus,
        encoder,
        delta=args.delta,
        max_passage_words=120,
        negation_safe=True,
        precomputed_passages_embeddings=engine.raw_embeddings,
    )

    configs = [
        ("BM25 Murni (alpha=1.0)", 1.0, False, False),
        ("Dense Murni (alpha=0.0)", 0.0, False, False),
        ("Hybrid Default (alpha=0.5, ES)", 0.5, True, False),
        (f"Hybrid Exhaustive (alpha*={best_alpha})", best_alpha, False, False),
        (f"Hybrid Safe Early Stopping (alpha*={best_alpha})", best_alpha, True, False),
        (f"Hybrid Negation-Safe (alpha*={best_alpha})", best_alpha, True, True),
    ]

    test_summary_rows = []
    test_detail_all = []
    per_config_details: dict[str, list[dict]] = {}
    for label, a, es, use_neg in configs:
        eng = neg_engine if use_neg else engine
        summ, indiv = evaluate_config(eng, test_queries, alpha=a, k=args.top_k, early_stopping=es, query_vecs=query_vecs)
        summ3, _ = evaluate_config(eng, test_queries, alpha=a, k=3, early_stopping=es, query_vecs=query_vecs)
        summ_row = {
            "model": label,
            **summ,
            "Recall@3": summ3["Recall@k"],
            "MRR@3": summ3["MRR@k"],
            "nDCG@3": summ3["nDCG@k"],
        }
        test_summary_rows.append(summ_row)
        per_config_details[label] = indiv
        for row in indiv:
            test_detail_all.append({"model": label, **row})

    ex_label = f"Hybrid Exhaustive (alpha*={best_alpha})"
    es_label = f"Hybrid Safe Early Stopping (alpha*={best_alpha})"
    es_mismatches = sum(
        1
        for r_ex, r_es in zip(per_config_details[ex_label], per_config_details[es_label])
        if r_ex["ranking"] != r_es["ranking"]
    )

    # 3. Breakdown per kategori
    categories = sorted({q["category"] for q in test_queries})
    cat_rows = []
    for cat in categories:
        row = {"category": cat, "n_queries": sum(1 for q in test_queries if q["category"] == cat)}
        for short, full_label in [("BM25", "BM25 Murni (alpha=1.0)"), ("Dense", "Dense Murni (alpha=0.0)"), ("Hybrid", es_label)]:
            subset = [r for r in per_config_details[full_label] if r["category"] == cat]
            row[f"{short}_Recall@{args.top_k}"] = round(float(np.mean([r["Recall@k"] for r in subset])), 4)
            row[f"{short}_MRR@{args.top_k}"] = round(float(np.mean([r["MRR@k"] for r in subset])), 4)
            row[f"{short}_nDCG@{args.top_k}"] = round(float(np.mean([r["nDCG@k"] for r in subset])), 4)
        cat_rows.append(row)

    # 4. Latency CPU benchmark (28 * 8 = 224 queries)
    lat_ex = benchmark_cpu_latency(engine, test_queries, alpha=best_alpha, k=args.top_k, early_stopping=False, warmup=args.warmup, repeats=args.repeats)
    lat_es = benchmark_cpu_latency(engine, test_queries, alpha=best_alpha, k=args.top_k, early_stopping=True, warmup=args.warmup, repeats=args.repeats)
    lat_bm = benchmark_cpu_latency(engine, test_queries, alpha=1.0, k=args.top_k, early_stopping=False, warmup=args.warmup, repeats=args.repeats)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "dev_sweep_alpha.csv", dev_alpha_rows)
    write_csv(out / "test_ringkasan.csv", test_summary_rows)
    write_csv(out / "test_per_kueri.csv", test_detail_all)
    write_csv(out / "test_per_kategori.csv", cat_rows)

    metadata = {
        "execution_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "dataset_pedoman_akademik.json (43 entitas: 35 FAQ + 8 Passage Pedoman)",
        "encoder": MODEL_NAME,
        "force_cpu": True,
        "torch_num_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "numpy": np.__version__,
        "machine": platform.platform(),
        "processor": platform.processor(),
        "corpus_n": len(corpus),
        "faq_count": sum(1 for x in corpus if x.get("type") == "faq"),
        "passage_entry_count": sum(1 for x in corpus if x.get("type") == "passage"),
        "multi_passage_docs": engine.multi_passage_docs,
        "raw_passages": engine.raw_passage_count,
        "coalesced_passages": engine.coalesced_count,
        "compression_pct": round(100.0 * engine.compression_ratio, 4),
        "avgdl_tokens": round(engine.bm25.avgdl, 2),
        "dev_query_n": len(dev_queries),
        "test_query_n": len(test_queries),
        "top_k": args.top_k,
        "delta": args.delta,
        "selected_alpha_from_dev": best_alpha,
        "early_stopping_vs_exhaustive_mismatches_on_test": es_mismatches,
        "latency_cpu_exhaustive": lat_ex,
        "latency_cpu_early_stopping": lat_es,
        "latency_cpu_bm25_only": lat_bm,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metadata": metadata, "test_summary": test_summary_rows, "by_category": cat_rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
