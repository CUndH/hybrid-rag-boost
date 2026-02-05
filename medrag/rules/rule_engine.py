from __future__ import annotations

import json
from typing import Dict, Any, List, Optional

from medrag.retrieval.vector_store import VectorStore


class RuleEngine:
    """
    轻量“图谱/规则层”：
    - rules.json 里定义：
        when: drug=..., population=...
        then: force_evidence_ids=[...], message=...
    - apply() 返回 forced evidence（带 source=forced, rule=...）
    """

    def __init__(self, rules_path: str, store: VectorStore):
        with open(rules_path, "r", encoding="utf-8") as f:
            self.rules = json.load(f)
        self.store = store

    def apply(self, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        drugs = entities.get("drugs") or []
        pops = entities.get("populations") or []

        forced: List[Dict[str, Any]] = []
        for r in self.rules:
            if self._match(r.get("when", {}), drugs, pops):
                for eid in r.get("force_evidence_ids", []):
                    doc = self.store.get_by_id(eid)
                    if doc:
                        forced.append(
                            {
                                **doc,
                                "vec_score": 1.0,      # forced 不靠向量分
                                "rerank_score": 1.0,   # forced 优先
                                "source": "forced",
                                "rule": r.get("message", "forced by rule"),
                            }
                        )
        # 去重
        seen = set()
        uniq = []
        for e in forced:
            if e["id"] in seen:
                continue
            seen.add(e["id"])
            uniq.append(e)
        return uniq

    def _match(self, cond: Dict[str, Any], drugs: List[str], pops: List[str]) -> bool:
        drug = cond.get("drug")
        pop = cond.get("population")

        ok = True
        if drug:
            ok = ok and (drug in drugs)
        if pop:
            ok = ok and (pop in pops)
        return ok
