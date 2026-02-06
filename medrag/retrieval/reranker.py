from __future__ import annotations

from typing import List, Dict, Any, Optional

# 你现有的 SimpleReranker（保持不变也行）
class SimpleReranker:
    def rerank(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        qset = set(query.replace(" ", ""))

        def score(h: Dict[str, Any]) -> float:
            t = (h.get("title", "") + h.get("text", "")).replace(" ", "")
            tset = set(t)
            overlap = len(qset & tset)
            return overlap / (len(qset) + 1e-6)

        out = []
        for h in hits:
            h2 = dict(h)
            h2["rerank_score"] = float(score(h2))
            out.append(h2)

        out.sort(key=lambda x: x["rerank_score"], reverse=True)
        return out


def build_reranker(kind: str = "simple", **kwargs):
    """
    kind:
      - "simple"
      - "cross_encoder"
    kwargs: CrossEncoderConfig 参数
    """
    kind = (kind or "simple").lower().strip()
    if kind == "simple":
        return SimpleReranker()

    if kind == "cross_encoder":
        try:
            from medrag.retrieval.cross_encoder_reranker import CrossEncoderReranker, CrossEncoderConfig
            cfg = CrossEncoderConfig(**kwargs)
            return CrossEncoderReranker(cfg)
        except Exception as e:
            # fallback：避免 demo 因依赖/模型问题直接崩
            print(f"[WARN] CrossEncoder init failed, fallback to SimpleReranker. err={e}")
            return SimpleReranker()

    raise ValueError(f"Unknown reranker kind: {kind}")
