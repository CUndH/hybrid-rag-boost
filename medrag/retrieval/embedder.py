from __future__ import annotations

import hashlib
from typing import List
import numpy as np


class HashingEmbedder:
    """
    离线可复现 embedding：
    - 字符 2-gram hashing -> 固定维度向量
    - L2 normalize
    用途：demo 可跑 + 可解释 + 不依赖外网模型
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _hash(self, s: str) -> int:
        return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)

    def encode(self, texts: List[str], normalize_embeddings: bool = True) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            t = t.replace(" ", "")
            grams = [t[j : j + 2] for j in range(max(0, len(t) - 1))]
            for g in grams:
                idx = self._hash(g) % self.dim
                vecs[i, idx] += 1.0

        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
            vecs = vecs / norms
        return vecs
