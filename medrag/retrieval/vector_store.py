from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Iterable

import numpy as np

from medrag.retrieval.embedder import HashingEmbedder


@dataclass
class Doc:
    id: str
    title: str
    text: str
    metadata: Dict[str, Any]


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


class VectorStore:
    def __init__(self, documents_path: str):
        # 兼容两种格式：
        # 1) JSON list（旧版 documents.json）
        # 2) JSONL（新版 chunks.jsonl）
        docs: List[Dict[str, Any]] = []

        # 先尝试当 JSON 读（list）
        try:
            with open(documents_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                docs = raw
            else:
                # 不是 list，就当 JSONL 处理
                docs = list(_iter_jsonl(documents_path))
        except json.JSONDecodeError:
            # JSON 解析失败 -> JSONL
            docs = list(_iter_jsonl(documents_path))

        self.docs: List[Doc] = []
        for item in docs:
            md = item.get("metadata") or {}

            # 新版 chunks.jsonl: chunk_id/doc_id
            if "chunk_id" in item and "id" not in item:
                _id = item["chunk_id"]
            else:
                _id = item.get("id") or item.get("chunk_id")  # 兜底

            # 把 doc_id 带进 metadata，方便上层统一使用
            if "doc_id" in item and "doc_id" not in md:
                md = dict(md)
                md["doc_id"] = item["doc_id"]

            self.docs.append(
                Doc(
                    id=str(_id),
                    title=item.get("title", ""),
                    text=item.get("text", ""),
                    metadata=md,
                )
            )

        self.embedder = HashingEmbedder(dim=512)
        corpus = [f"{d.title}。{d.text}" for d in self.docs]
        self.embs = self.embedder.encode(corpus, normalize_embeddings=True).astype(np.float32)

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

    def search(
        self,
        query: str,
        topk: int = 12,
        filters: Dict[str, Optional[str]] | None = None,
        allow_generic_population: bool = True,
        fallback_to_all: bool = True,
        # 下面这些参数如果 hybrid/bm25 已接入就保留
        mode: str = "vector",
        bm25_topk: int = 80,
        vector_topk: int = 80,
        rrf_k: int = 60,
        weights: tuple[float, float] = (1.0, 1.0),
    ) -> List[Dict[str, Any]]:
        # 先走现有的 metadata 过滤逻辑（drug/population）
        filters = filters or {}
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

        # 为了不破坏现有工程，这里保留 vector 作为稳定兜底。
        q = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)[0]
        sub = self.embs[cand_idx]
        scores = (sub @ q).astype(np.float32)

        topk = min(topk, len(cand_idx))
        order = np.argsort(-scores)[:topk]

        hits: List[Dict[str, Any]] = []
        for p in order:
            real_i = cand_idx[int(p)]
            d = self.docs[real_i]
            hits.append(
                {
                    "id": d.id,
                    "title": d.title,
                    "text": d.text,
                    "metadata": d.metadata,
                    "vec_score": float(scores[int(p)]),
                    "bm25_score": 0.0,
                    "fused_score": float(scores[int(p)]),
                    "rerank_score": 0.0,
                    "source": "vector",
                }
            )
        return hits
