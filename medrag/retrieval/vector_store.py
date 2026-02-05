from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from medrag.retrieval.embedder import HashingEmbedder

# ✅ NEW
import jieba
from rank_bm25 import BM25Okapi


@dataclass
class Doc:
    id: str
    title: str
    text: str
    metadata: Dict[str, Any]


def _tokenize_zh(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # 你也可以在这里加 stopwords / 归一化 / 英文 split 等
    return [t for t in jieba.lcut(text) if t.strip()]


def _rrf_fusion(
    ranked_lists: List[List[int]],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> Dict[int, float]:
    """
    ranked_lists: 多个“doc index 的排序列表”（rank 1 最相关）
    return: doc_index -> fused_score
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    assert len(weights) == len(ranked_lists)

    scores: Dict[int, float] = {}
    for w, lst in zip(weights, ranked_lists):
        for rank, idx in enumerate(lst, start=1):
            scores[idx] = scores.get(idx, 0.0) + w * (1.0 / (k + rank))
    return scores


class VectorStore:
    def __init__(self, documents_path: str):
        with open(documents_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.docs: List[Doc] = []
        for item in raw:
            md = item.get("metadata") or {}
            self.docs.append(
                Doc(
                    id=item["id"],
                    title=item["title"],
                    text=item["text"],
                    metadata=md,
                )
            )

        # ===== Vector index =====
        self.embedder = HashingEmbedder(dim=512)
        corpus = [f"{d.title}。{d.text}" for d in self.docs]
        self.embs = self.embedder.encode(corpus, normalize_embeddings=True).astype(np.float32)

        # ===== BM25 index (NEW) =====
        self._bm25_corpus_tokens: List[List[str]] = [_tokenize_zh(c) for c in corpus]
        self._bm25 = BM25Okapi(self._bm25_corpus_tokens)

        self._id2doc = {d.id: d for d in self.docs}

    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        d = self._id2doc.get(doc_id)
        if not d:
            return None
        return {
            "id": d.id,
            "title": d.title,
            "text": d.text,
            "metadata": d.metadata,
        }

    def _candidate_indices(
        self,
        filters: Dict[str, Optional[str]],
        allow_generic_population: bool,
        fallback_to_all: bool,
    ) -> List[int]:
        drug = filters.get("drug")
        population = filters.get("population")

        cand_idx: List[int] = []
        for i, d in enumerate(self.docs):
            md = d.metadata or {}
            drugs = md.get("drug", []) or []
            pops = md.get("population", []) or []

            ok = True
            if drug:
                ok = ok and (drug in drugs)
            if population:
                if allow_generic_population:
                    ok = ok and ((population in pops) or (len(pops) == 0))
                else:
                    ok = ok and (population in pops)

            if ok:
                cand_idx.append(i)

        if len(cand_idx) == 0 and fallback_to_all:
            cand_idx = list(range(len(self.docs)))

        return cand_idx

    def _vector_rank(
        self,
        query: str,
        cand_idx: List[int],
        topk: int,
    ) -> Tuple[List[int], Dict[int, float]]:
        q = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)[0]
        sub = self.embs[cand_idx]
        scores = (sub @ q).astype(np.float32)

        topk = min(topk, len(cand_idx))
        order = np.argsort(-scores)[:topk]

        ranked_real_idx: List[int] = []
        score_map: Dict[int, float] = {}
        for p in order:
            real_i = cand_idx[int(p)]
            ranked_real_idx.append(real_i)
            score_map[real_i] = float(scores[int(p)])
        return ranked_real_idx, score_map

    def _bm25_rank(
        self,
        query: str,
        cand_idx: List[int],
        topk: int,
    ) -> Tuple[List[int], Dict[int, float]]:
        q_tokens = _tokenize_zh(query)
        if not q_tokens:
            return [], {}

        # BM25Okapi.get_scores 返回的是“全量 doc 的分数”
        all_scores = np.array(self._bm25.get_scores(q_tokens), dtype=np.float32)

        # 只取候选集合的分数再排序
        cand_scores = all_scores[cand_idx]
        topk = min(topk, len(cand_idx))
        order = np.argsort(-cand_scores)[:topk]

        ranked_real_idx: List[int] = []
        score_map: Dict[int, float] = {}
        for p in order:
            real_i = cand_idx[int(p)]
            ranked_real_idx.append(real_i)
            score_map[real_i] = float(cand_scores[int(p)])
        return ranked_real_idx, score_map

    def search(
        self,
        query: str,
        topk: int = 12,
        filters: Dict[str, Optional[str]] | None = None,
        allow_generic_population: bool = True,
        fallback_to_all: bool = True,
        # ✅ NEW: hybrid 控制参数
        mode: str = "hybrid",  # "hybrid" | "vector" | "bm25"
        bm25_topk: int = 60,
        vector_topk: int = 60,
        rrf_k: int = 60,
        weights: Tuple[float, float] = (1.2, 1.0),  # (bm25_weight, vector_weight)
    ) -> List[Dict[str, Any]]:
        filters = filters or {}

        cand_idx = self._candidate_indices(
            filters=filters,
            allow_generic_population=allow_generic_population,
            fallback_to_all=fallback_to_all,
        )

        if len(cand_idx) == 0:
            return []

        # --- vector only ---
        if mode == "vector":
            v_rank, v_score = self._vector_rank(query, cand_idx, topk)
            hits: List[Dict[str, Any]] = []
            for real_i in v_rank:
                d = self.docs[real_i]
                hits.append(
                    {
                        "id": d.id,
                        "title": d.title,
                        "text": d.text,
                        "metadata": d.metadata,
                        "vec_score": float(v_score.get(real_i, 0.0)),
                        "bm25_score": 0.0,
                        "fused_score": float(v_score.get(real_i, 0.0)),  # 兼容
                        "rerank_score": 0.0,
                        "source": "vector",
                    }
                )
            return hits

        # --- bm25 only ---
        if mode == "bm25":
            b_rank, b_score = self._bm25_rank(query, cand_idx, topk)
            hits: List[Dict[str, Any]] = []
            for real_i in b_rank:
                d = self.docs[real_i]
                hits.append(
                    {
                        "id": d.id,
                        "title": d.title,
                        "text": d.text,
                        "metadata": d.metadata,
                        "vec_score": 0.0,
                        "bm25_score": float(b_score.get(real_i, 0.0)),
                        "fused_score": float(b_score.get(real_i, 0.0)),  # 兼容
                        "rerank_score": 0.0,
                        "source": "bm25",
                    }
                )
            return hits

        # --- hybrid (default) ---
        b_rank, b_score = self._bm25_rank(query, cand_idx, bm25_topk)
        v_rank, v_score = self._vector_rank(query, cand_idx, vector_topk)

        fused = _rrf_fusion([b_rank, v_rank], k=rrf_k, weights=[weights[0], weights[1]])
        fused_sorted = sorted(fused.items(), key=lambda x: x[1], reverse=True)[: min(topk, len(fused))]

        hits: List[Dict[str, Any]] = []
        for real_i, fused_score in fused_sorted:
            d = self.docs[real_i]
            hits.append(
                {
                    "id": d.id,
                    "title": d.title,
                    "text": d.text,
                    "metadata": d.metadata,
                    "vec_score": float(v_score.get(real_i, 0.0)),
                    "bm25_score": float(b_score.get(real_i, 0.0)),
                    "fused_score": float(fused_score),
                    "rerank_score": 0.0,  # reranker 会填
                    "source": "hybrid",
                }
            )
        return hits
