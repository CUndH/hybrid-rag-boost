from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import math


@dataclass
class CrossEncoderConfig:
    model_name_or_path: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"               # "cpu" | "cuda"
    batch_size: int = 16
    max_length: int = 384
    use_sigmoid: bool = False         # 有些模型输出已是相关性logit；sigmoid 仅用于压缩到(0,1)
    # 只对召回池前多少条进行CE重排（通常 40~120）
    rerank_topn: int = 80


class CrossEncoderReranker:
    """
    Cross-Encoder reranker (transformers):
    - 输入：query + (title+text)
    - 输出：rerank_score（越大越相关）
    """

    def __init__(self, cfg: Optional[CrossEncoderConfig] = None):
        self.cfg = cfg or CrossEncoderConfig()
        self._loaded = False

        # 延迟加载：避免 CLI 启动就卡住/报错
        self.tokenizer = None
        self.model = None

    def _lazy_load(self) -> None:
        if self._loaded:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as e:
            raise RuntimeError(
                "CrossEncoderReranker requires torch + transformers. "
                "Install: pip install torch transformers"
            ) from e

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.cfg.model_name_or_path)
        self.model.eval()

        if self.cfg.device == "cuda":
            self.model.to("cuda")

        self._loaded = True

    def _score_pairs(self, pairs: List[tuple[str, str]]) -> List[float]:
        import torch

        assert self.tokenizer is not None and self.model is not None

        scores: List[float] = []
        bs = max(1, int(self.cfg.batch_size))

        with torch.no_grad():
            for i in range(0, len(pairs), bs):
                batch = pairs[i : i + bs]
                qs = [q for q, _ in batch]
                ps = [p for _, p in batch]

                enc = self.tokenizer(
                    qs,
                    ps,
                    padding=True,
                    truncation=True,
                    max_length=int(self.cfg.max_length),
                    return_tensors="pt",
                )

                if self.cfg.device == "cuda":
                    enc = {k: v.to("cuda") for k, v in enc.items()}

                out = self.model(**enc)
                # 常见：logits shape = [B, 1] 或 [B, 2]
                logits = out.logits

                if logits.dim() == 2 and logits.size(-1) == 1:
                    batch_scores = logits.squeeze(-1)
                elif logits.dim() == 2 and logits.size(-1) == 2:
                    # 取“相关”类（通常是 index=1）
                    batch_scores = logits[:, 1]
                else:
                    batch_scores = logits.view(-1)

                batch_scores = batch_scores.detach().float().cpu().tolist()

                if self.cfg.use_sigmoid:
                    batch_scores = [1.0 / (1.0 + math.exp(-s)) for s in batch_scores]

                scores.extend([float(s) for s in batch_scores])

        return scores

    def rerank(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not hits:
            return []

        self._lazy_load()

        # 只重排 topn，剩余保持原顺序拼在后面（节省算力）
        topn = min(int(self.cfg.rerank_topn), len(hits))
        head = hits[:topn]
        tail = hits[topn:]

        pairs: List[tuple[str, str]] = []
        for h in head:
            passage = (h.get("title", "") + " " + h.get("text", "")).strip()
            pairs.append((query, passage))

        ce_scores = self._score_pairs(pairs)

        out_head: List[Dict[str, Any]] = []
        for h, s in zip(head, ce_scores):
            h2 = dict(h)
            h2["rerank_score"] = float(s)
            h2["rerank_model"] = self.cfg.model_name_or_path
            out_head.append(h2)

        out_head.sort(key=lambda x: x["rerank_score"], reverse=True)

        # 尾部不做CE，保持原召回/融合排序（也可以给个很小的 rerank_score）
        out_tail = [dict(h, rerank_score=float(h.get("rerank_score", 0.0))) for h in tail]

        return out_head + out_tail
