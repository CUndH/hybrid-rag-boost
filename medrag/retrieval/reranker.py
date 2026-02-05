from __future__ import annotations

from typing import List, Dict, Any
import math


class SimpleReranker:
    """
    教学 + Hybrid 友好的 reranker：

    rerank_score =
        lexical_overlap_score
        × (1 + alpha * prior_score)

    prior_score 优先级：
        fused_score > vec_score > bm25_score
    """

    def __init__(self, alpha: float = 0.8):
        """
        alpha:
            0   → 完全等价于你现在的 reranker
            0.5 → 轻度尊重召回排序
            1.0 → 强调 hybrid 先验（推荐 0.6~1.0）
        """
        self.alpha = alpha

    def rerank(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        qset = set(query.replace(" ", ""))

        def lexical_score(h: Dict[str, Any]) -> float:
            t = (h.get("title", "") + h.get("text", "")).replace(" ", "")
            tset = set(t)
            overlap = len(qset & tset)
            return overlap / (len(qset) + 1e-6)

        def prior_score(h: Dict[str, Any]) -> float:
            """
            hybrid 先验分数，做一个轻度压缩，避免某一路极端大
            """
            if "fused_score" in h and h["fused_score"] > 0:
                s = h["fused_score"]
            elif "vec_score" in h and h["vec_score"] > 0:
                s = h["vec_score"]
            elif "bm25_score" in h and h["bm25_score"] > 0:
                s = h["bm25_score"]
            else:
                return 0.0

            # log 压缩，防止 fused_score 拉爆
            return math.log1p(s)

        out = []
        for h in hits:
            h2 = dict(h)

            lex = lexical_score(h2)
            prior = prior_score(h2)

            h2["rerank_score"] = float(
                lex * (1.0 + self.alpha * prior)
            )

            # 👉 调试非常有用（可选）
            h2["_debug"] = {
                "lexical": lex,
                "prior": prior,
                "alpha": self.alpha,
            }

            out.append(h2)

        out.sort(key=lambda x: x["rerank_score"], reverse=True)
        return out
