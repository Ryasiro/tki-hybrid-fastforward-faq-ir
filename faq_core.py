"""Mesin IR Hybrid Fast-Forward untuk Dataset FAQ & Pedoman Akademik (43 entitas).

Komponen inti (independen dari Streamlit):
- Prapemrosesan leksikal Bahasa Indonesia (default & negation_safe)
- BM25PositiveIDF: IDF = ln(1 + (N - df + 0.5)/(df + 0.5)), k1=1.5, b=0.75
- Pemecahan multi-passage (default max_passage_words=120) & normalisasi L2
- Sequential Coalescing per dokumen (centroid dinormalisasi ulang)
- Agregasi Max-Passage (maxP) berbasis dot product pada Fast-Forward Index
- Interpolasi konveks linier & Safe Early Stopping Top-K (batas atas kosinus global = 1.0)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from heapq import heappush, heapreplace
from math import log1p
from time import perf_counter
from typing import Any, Protocol
import re
import numpy as np

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

STOPWORDS_DEFAULT = frozenset(
    "yang di ke dari untuk pada dengan atau dan ini itu jika tidak bisa apa bagaimana kapan saya anda".split()
)
STOPWORDS_NEGATION_SAFE = STOPWORDS_DEFAULT.difference({"tidak"})


def preprocess_lexical(text: str, *, negation_safe: bool = False) -> list[str]:
    stopwords = STOPWORDS_NEGATION_SAFE if negation_safe else STOPWORDS_DEFAULT
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    return [token for token in cleaned.split() if token not in stopwords]


class BM25PositiveIDF:
    """BM25 dengan IDF positif: ln(1 + (N - df + 0.5) / (df + 0.5))."""

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        if not tokenized_corpus:
            raise ValueError("Korpus tidak boleh kosong.")
        if k1 <= 0 or not (0 <= b <= 1):
            raise ValueError("Parameter BM25 tidak valid.")
        self.k1, self.b = float(k1), float(b)
        self.corpus_size = len(tokenized_corpus)
        self.doc_freqs = [Counter(doc) for doc in tokenized_corpus]
        self.doc_len = np.asarray([len(doc) for doc in tokenized_corpus], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean())
        if self.avgdl == 0:
            raise ValueError("Semua dokumen kosong setelah preprocessing.")
        df = Counter(term for counts in self.doc_freqs for term in counts)
        self.idf = {
            term: log1p((self.corpus_size - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float64)
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / self.avgdl)
        for term in query_tokens:
            if term not in self.idf:
                continue
            tf = np.fromiter(
                (doc.get(term, 0) for doc in self.doc_freqs),
                dtype=np.float64,
                count=self.corpus_size,
            )
            scores += self.idf[term] * (tf * (self.k1 + 1.0) / (tf + norm))
        return scores


def normalize_l2(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if not np.all(np.isfinite(v)):
        raise ValueError("Embedding mengandung NaN/Inf.")
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("Embedding bernorma nol; tidak dapat dinormalisasi.")
    return (v / norm).astype(np.float32)


def split_passages(text: str, max_words: int = 120) -> list[str]:
    if max_words < 1:
        raise ValueError("max_words harus >= 1.")
    tokens = text.split()
    if not tokens:
        raise ValueError("Teks dokumen tidak boleh kosong.")
    return [" ".join(tokens[i : i + max_words]) for i in range(0, len(tokens), max_words)]


def sequential_coalesce(vectors: np.ndarray, delta: float) -> np.ndarray:
    """Gabungkan vektor passage bersebelahan dalam satu dokumen jika distance < delta."""
    if not 0 <= delta < 1:
        raise ValueError("delta harus berada dalam [0, 1).")
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) == 0:
        raise ValueError("vectors harus berupa matriks passage nonkosong.")
    vectors = normalize_l2(vectors)
    if delta == 0:
        return vectors.copy()
    merged = []
    accumulator = vectors[0].astype(np.float64).copy()
    count = 1
    for vec in vectors[1:]:
        centroid = accumulator / count
        centroid /= np.linalg.norm(centroid)
        distance = 1.0 - float(np.dot(centroid, vec))
        if distance < delta:
            accumulator += vec
            count += 1
        else:
            merged.append(accumulator / count)
            accumulator = vec.astype(np.float64).copy()
            count = 1
    merged.append(accumulator / count)
    return normalize_l2(np.vstack(merged))


class EncoderProtocol(Protocol):
    def encode(self, sentences: Any, **kwargs: Any) -> np.ndarray: ...


@dataclass(frozen=True)
class Hit:
    rank: int
    idx: int
    id: str
    doc_type: str
    domain: str
    title: str
    question: str
    answer: str
    keywords: tuple[str, ...]
    source_status: str
    source_reference: str
    raw_bm25: float
    norm_bm25: float
    dense: float | None
    hybrid: float


@dataclass
class SearchResult:
    query: str
    tokens: list[str]
    alpha: float
    hits: list[Hit]
    evaluated_dense: int
    total_documents: int
    stopped_early: bool
    time_query_encoder_ms: float
    time_total_core_ms: float
    zero_lexical_match: bool


class HybridFAQ:
    """Fast-Forward Index prakomputasi, Sequential Coalescing, MaxPassage, dan Safe Early Stopping."""

    def __init__(
        self,
        corpus: list[dict[str, Any]],
        encoder: EncoderProtocol,
        *,
        delta: float = 0.035,
        max_passage_words: int = 120,
        negation_safe: bool = False,
        precomputed_passages_embeddings: np.ndarray | None = None,
    ):
        if not corpus:
            raise ValueError("Korpus kosong.")
        ids = [r["id"] for r in corpus]
        if len(set(ids)) != len(ids):
            raise ValueError("ID dokumen tidak unik.")
        self.corpus = corpus
        self.encoder = encoder
        self.delta = float(delta)
        self.max_passage_words = int(max_passage_words)
        self.negation_safe = bool(negation_safe)
        # Hanya gabungkan question + answer (Persamaan Concat(q*, a)); tidak menyuntikkan keywords
        self.document_texts = [f"{x['question']} {x['answer']}" for x in corpus]
        self.bm25 = BM25PositiveIDF(
            [preprocess_lexical(d, negation_safe=negation_safe) for d in self.document_texts]
        )
        self.passages_by_doc = [split_passages(d, max_passage_words) for d in self.document_texts]
        all_passages = [s for parts in self.passages_by_doc for s in parts]
        if precomputed_passages_embeddings is not None:
            encoded = np.asarray(precomputed_passages_embeddings, dtype=np.float32)
        else:
            encoded = np.asarray(
                self.encoder.encode(
                    all_passages, normalize_embeddings=True, show_progress_bar=False
                ),
                dtype=np.float32,
            )
        if encoded.ndim != 2 or encoded.shape[0] != len(all_passages):
            raise ValueError("Dimensi hasil encoder tidak sesuai jumlah passage.")
        self.raw_embeddings = normalize_l2(encoded)
        self.raw_passage_count = len(all_passages)
        self.multi_passage_docs = sum(1 for parts in self.passages_by_doc if len(parts) > 1)
        self.dim = int(self.raw_embeddings.shape[1])
        self.forward_index: dict[str, np.ndarray] = {}
        self.rebuild_coalesced_index(delta=self.delta)

    def rebuild_coalesced_index(self, *, delta: float) -> None:
        self.delta = float(delta)
        self.forward_index = {}
        off = 0
        for row, passages in zip(self.corpus, self.passages_by_doc):
            vecs = self.raw_embeddings[off : off + len(passages)]
            self.forward_index[row["id"]] = sequential_coalesce(vecs, delta=self.delta)
            off += len(passages)
        self.coalesced_count = sum(len(v) for v in self.forward_index.values())

    @property
    def compression_ratio(self) -> float:
        return 1.0 - self.coalesced_count / self.raw_passage_count

    def search(
        self,
        query: str,
        *,
        alpha: float = 0.5,
        top_k: int = 5,
        early_stopping: bool = True,
        precomputed_query_vec: np.ndarray | None = None,
    ) -> SearchResult:
        if not query.strip():
            raise ValueError("Kueri kosong.")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha harus dalam rentang [0, 1].")
        if top_k < 1:
            raise ValueError("top_k harus positif.")
        t0 = perf_counter()
        tokens = preprocess_lexical(query, negation_safe=self.negation_safe)
        raw = self.bm25.get_scores(tokens)
        max_raw = float(np.max(raw))
        norm = raw / max_raw if max_raw > 0 else np.zeros_like(raw)
        zero_match = bool(max_raw <= 0)
        n = len(self.corpus)
        q_encoder_ms = 0.0
        qvec = None
        if alpha < 1:
            if precomputed_query_vec is not None:
                qvec = normalize_l2(np.asarray(precomputed_query_vec, dtype=np.float32).reshape(1, -1))[0]
            else:
                tq = perf_counter()
                qvec_arr = np.asarray(
                    self.encoder.encode([query], normalize_embeddings=True, show_progress_bar=False),
                    dtype=np.float32,
                )
                q_encoder_ms = 1000.0 * (perf_counter() - tq)
                if qvec_arr.shape != (1, self.dim):
                    raise ValueError("Dimensi vektor kueri tidak sesuai indeks.")
                qvec = normalize_l2(qvec_arr)[0]

        def make_hit(rank: int, i: int, dense_val: float | None, hybrid_val: float) -> Hit:
            row = self.corpus[i]
            kws = row.get("keywords", [])
            if isinstance(kws, str):
                kws = [k.strip() for k in kws.split(",") if k.strip()]
            return Hit(
                rank=rank,
                idx=i,
                id=row["id"],
                doc_type=row.get("type", "faq"),
                domain=row.get("domain", row.get("category", "")),
                title=row.get("title", row["question"]),
                question=row["question"],
                answer=row["answer"],
                keywords=tuple(kws),
                source_status=row.get("source_status", ""),
                source_reference=row.get("source_reference", ""),
                raw_bm25=float(raw[i]),
                norm_bm25=float(norm[i]),
                dense=dense_val,
                hybrid=float(hybrid_val),
            )

        if alpha == 1:
            ranked = [i for i in sorted(range(n), key=lambda i: (-norm[i], i)) if raw[i] > 0]
            selected = ranked[:top_k]
            hits = [make_hit(k, i, None, float(norm[i])) for k, i in enumerate(selected, 1)]
            return SearchResult(
                query, tokens, alpha, hits, 0, n, False, q_encoder_ms,
                1000.0 * (perf_counter() - t0), zero_match,
            )

        order = sorted(range(n), key=lambda i: (-norm[i], i))
        heap: list[tuple[float, int, int, float]] = []
        evaluated = 0
        stopped = False
        for pos, i in enumerate(order):
            if early_stopping and len(heap) >= top_k and pos > 0:
                upper_bound = alpha * float(norm[i]) + (1.0 - alpha) * 1.0
                if upper_bound < heap[0][0] - 1e-12:
                    stopped = True
                    break
            vecs = self.forward_index[self.corpus[i]["id"]]
            dense = float(np.max(np.clip(vecs @ qvec, -1.0, 1.0)))
            score = alpha * float(norm[i]) + (1.0 - alpha) * dense
            item = (score, -i, i, dense)
            evaluated += 1
            if len(heap) < top_k:
                heappush(heap, item)
            elif (score, -i) > (heap[0][0], heap[0][1]):
                heapreplace(heap, item)

        ranked = sorted(heap, key=lambda e: (-e[0], e[2]))
        hits = [make_hit(k, i, float(dense), float(score)) for k, (score, _, i, dense) in enumerate(ranked, 1)]
        return SearchResult(
            query, tokens, alpha, hits, evaluated, n, stopped, q_encoder_ms,
            1000.0 * (perf_counter() - t0), zero_match,
        )
