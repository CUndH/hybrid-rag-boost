from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

from medrag.config import default_paths
from medrag.nlp.entity_extractor import EntityExtractor
from medrag.retrieval.vector_store import VectorStore
from medrag.retrieval.reranker import SimpleReranker
from medrag.rules.rule_engine import RuleEngine


@dataclass
class PipelineConfig:
    allow_generic_population: bool = True
    fallback_to_all: bool = True
    drug_filter_min_conf: float = 0.65
    pop_filter_min_conf: float = 0.55

    # ✅ NEW: hybrid recall knobs
    recall_mode: str = "hybrid"   # "hybrid" | "vector" | "bm25"
    recall_pool_k: int = 60       # 先召回多少条给 rerank（建议 40~120）
    bm25_pool_k: int = 80         # bm25 子召回池大小
    vector_pool_k: int = 80       # vector 子召回池大小
    rrf_k: int = 60               # RRF 的 k
    bm25_weight: float = 1.2
    vector_weight: float = 1.0


class HybridRAGPipeline:
    """
    统一融合版：
    1) 实体识别（drug/population）
    2) metadata 过滤 + 向量召回（粗召回 topN）
    3) rerank 精排
    4) 规则引擎强制补全（禁用/慎用等强约束）
    5) 合并证据：forced 优先，其次 rerank topK
    """

    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        paths = default_paths()

        self.entities = EntityExtractor(paths.aliases_path)
        self.store = VectorStore(paths.documents_path)
        self.reranker = SimpleReranker()
        self.rules = RuleEngine(paths.rules_path, self.store)

    def answer(self, query: str, topk_recall: int = 12, topk_final: int = 5) -> Dict[str, Any]:
        ents = self.entities.extract(query)

        # 1) 规则兜底：先拿 forced evidence（强约束）
        forced = self.rules.apply(ents)

        # ✅ NEW: 召回池大小：优先用 cfg.recall_pool_k（更适合 rerank）
        recall_k = self.cfg.recall_pool_k or topk_recall

        # 2) hybrid 召回（带 metadata 过滤）
        recall = self.store.search(
            query=query,
            topk=recall_k,  # ✅ 给 rerank 更大候选池
            filters={
                "drug": ents["drugs"][0] if ents["drugs"] else None,
                "population": ents["populations"][0] if ents["populations"] else None,
            },
            allow_generic_population=self.cfg.allow_generic_population,
            fallback_to_all=self.cfg.fallback_to_all,

            # ✅ NEW: hybrid params（要求你 VectorStore.search 支持这些参数）
            mode=self.cfg.recall_mode,
            bm25_topk=self.cfg.bm25_pool_k,
            vector_topk=self.cfg.vector_pool_k,
            rrf_k=self.cfg.rrf_k,
            weights=(self.cfg.bm25_weight, self.cfg.vector_weight),
        )

        # 3) rerank（只对 recall）
        reranked = self.reranker.rerank(query, recall)

        # 4) 合并证据（去重）：forced 优先
        forced_ids = {e["id"] for e in forced}
        final: List[Dict[str, Any]] = []
        for e in forced:
            final.append(e)

        for e in reranked:
            if e["id"] in forced_ids:
                continue
            final.append(e)
            if len(final) >= topk_final:
                break

        # A+: 无实体场景的 section 白名单...
        if not ents.get("drugs") and not ents.get("populations"):
            allow = {"symptom", "general_principle"}
            filtered = [e for e in final if (e.get("metadata", {}) or {}).get("section") in allow]
            if filtered:
                final = filtered

        draft = self._draft_answer_template(query, ents, forced, final)

        return {
            "query": query,
            "entities": ents,
            "forced": [f["rule"] for f in forced] if forced else [],
            "final_evidence": final,
            "draft_answer": draft,
            # ✅ 可选：调试时看看 hybrid 召回的分数结构
            # "recall_debug": recall,
        }

    def _draft_answer_template(
        self,
        query: str,
        ents: Dict[str, Any],
        forced: List[Dict[str, Any]],
        final: List[Dict[str, Any]],
    ) -> str:
        drug = ents["drugs"][0] if ents.get("drugs") else "（未识别药物）"
        pop = ents["populations"][0] if ents.get("populations") else "（未识别人群）"

        has_contra = any(
            e.get("metadata", {}).get("section") == "contraindication"
            for e in forced
        )
        has_warn = any(
            e.get("metadata", {}).get("risk_level") in ("warning", "caution")
            for e in forced
        )

        if has_contra:
            return f"针对 {pop} 的用药问题：{drug} 存在明确禁忌或禁用提示，请勿自行用药，应立即咨询医生。"

        if has_warn:
            return f"{drug} 可用于相关适应症，但对 {pop} 存在慎用提示，应由医生评估风险并进行监测。"

        return f"基于当前证据，建议围绕 {drug} 与 {pop} 的适用性进一步核对禁忌与注意事项，并咨询医生确认。"
