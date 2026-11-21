"""Unit & integration test untuk pipeline dataset_pedoman_akademik.json (43 entitas)."""
import csv
import json
from pathlib import Path
import numpy as np
import pytest

from faq_core import (
    BM25PositiveIDF,
    HybridFAQ,
    normalize_l2,
    preprocess_lexical,
    sequential_coalesce,
    split_passages,
)
from evaluate import compute_metrics, evaluate_config

ROOT = Path(__file__).resolve().parent
CORPUS = json.loads((ROOT / "dataset_pedoman_akademik.json").read_text(encoding="utf-8"))
QUERIES = json.loads((ROOT / "queries_eval.json").read_text(encoding="utf-8"))


class SyntheticEncoder:
    def encode(self, sentences, **kwargs):
        out = []
        for text in sentences:
            seed = sum((i + 1) * ord(c) for i, c in enumerate(text[:200])) % (2**32)
            v = np.random.default_rng(seed).normal(size=24)
            v /= np.linalg.norm(v)
            out.append(v.astype("float32"))
        return np.asarray(out)


def test_dataset_json_csv_and_factual_consistency():
    assert len(CORPUS) == 43
    ids = [d["id"] for d in CORPUS]
    assert len(ids) == len(set(ids))
    with (ROOT / "dataset_pedoman_akademik.csv").open(encoding="utf-8-sig") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 43
    assert [r["id"] for r in csv_rows] == ids
    by_id = {d["id"]: d for d in CORPUS}
    # Regresi faktual terhadap Buku Pedoman UNSRI 2026/2027
    assert "KPM" in by_id["ADM_001"]["answer"]
    assert "berbeda dari SKPI" in by_id["ADM_004"]["answer"]
    assert "85%" in by_id["AKD_001"]["answer"]
    assert "1 semester" in by_id["CUTI_002"]["answer"] and "diperhitungkan" in by_id["CUTI_002"]["answer"]
    assert "USEPT" in by_id["TA_003"]["answer"]
    assert "52 SKS" in by_id["PEDOMAN_BAB5_P1"]["answer"]
    assert "pendamping ijazah" in by_id["PEDOMAN_BAB4_P3"]["answer"].lower()
    assert all("Bab VII" not in d["domain"] for d in CORPUS)
    assert all(d.get("source_status") in {"terverifikasi_pedoman_unsri_2026_2027", "kurasi_operasional_unit"} for d in CORPUS)


def test_queries_dev_test_disjoint_and_valid():
    corpus_ids = {d["id"] for d in CORPUS}
    dev = [q for q in QUERIES if q["split"] == "dev"]
    test = [q for q in QUERIES if q["split"] == "test"]
    assert len(dev) == 20
    assert len(test) == 30
    assert set(q["qid"] for q in dev).isdisjoint(q["qid"] for q in test)
    assert set(q["query"] for q in dev).isdisjoint(q["query"] for q in test)
    for q in QUERIES:
        for doc_id, grade in q["relevance"].items():
            assert doc_id in corpus_ids
            assert grade in (1, 2)


def test_preprocessing_and_bm25_positive_idf():
    assert preprocess_lexical("Tidak ingat kode akses!") == ["ingat", "kode", "akses"]
    assert preprocess_lexical("Tidak ingat kode akses!", negation_safe=True) == ["tidak", "ingat", "kode", "akses"]
    bm = BM25PositiveIDF([preprocess_lexical(d["question"] + " " + d["answer"]) for d in CORPUS])
    assert all(v > 0 for v in bm.idf.values())
    tik_idx = next(i for i, d in enumerate(CORPUS) if d["id"] == "TIK_001")
    assert bm.get_scores(preprocess_lexical("tidak ingat kode akses masuk sistem"))[tik_idx] == 0.0


def test_coalesce_and_split_passages():
    v = normalize_l2(np.array([[1.0, 0.0, 0.0], [0.998, 0.05, 0.0], [0.0, 1.0, 0.0]], dtype="float32"))
    assert sequential_coalesce(v, 0.02).shape == (2, 3)
    assert sequential_coalesce(v, 0.0).shape == (3, 3)
    assert len(split_passages(" ".join(["kata"] * 250), max_words=120)) == 3


@pytest.mark.parametrize("alpha", [0.0, 0.2, 0.5, 0.8, 0.95])
@pytest.mark.parametrize("k", [1, 3, 5])
def test_exact_early_stopping_invariant(alpha, k):
    engine = HybridFAQ(CORPUS, SyntheticEncoder(), delta=0.035)
    for q in QUERIES[:10]:
        r_es = engine.search(q["query"], alpha=alpha, top_k=k, early_stopping=True)
        r_ex = engine.search(q["query"], alpha=alpha, top_k=k, early_stopping=False)
        assert [h.id for h in r_es.hits] == [h.id for h in r_ex.hits]
        for a, b in zip(r_es.hits, r_ex.hits):
            assert a.hybrid == pytest.approx(b.hybrid, abs=1e-7)


def test_no_match_and_metrics():
    engine = HybridFAQ(CORPUS, SyntheticEncoder())
    assert engine.search("zzzzxxxyyy tidakada", alpha=1.0, top_k=5).hits == []
    assert len(engine.search("zzzzxxxyyy tidakada", alpha=0.0, top_k=5).hits) == 5

    class H:
        def __init__(self, id):
            self.id = id

    m = compute_metrics([H("x"), H("d2"), H("d1")], {"d1": 2, "d2": 1}, k=3)
    assert m["Recall@k"] == 1.0
    assert m["MRR@k"] == 0.5
    assert m["AP@k"] == pytest.approx((0.5 + 2 / 3) / 2)
    summ, _ = evaluate_config(engine, QUERIES[:5], alpha=0.5, k=5)
    assert summ["n_queries"] == 5
