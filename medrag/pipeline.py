from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

from medrag.config import default_paths
from medrag.nlp.entity_extractor import EntityExtractor
from medrag.nlp.intent_classifier import QueryIntentClassifier
from medrag.retrieval.vector_store import VectorStore
from medrag.rules.rule_engine import RuleEngine
from medrag.retrieval.reranker import build_reranker


@dataclass
class PipelineConfig:
    # --- existing knobs ---
    allow_generic_population: bool = True
    fallback_to_all: bool = True
    drug_filter_min_conf: float = 0.65
    pop_filter_min_conf: float = 0.55

    # --- NEW: recall baseline (bigger pool for rerank) ---
    recall_pool_k: int = 60

    # --- NEW: safety-critical strategy (BM25-biased) ---
    safety_mode: str = "hybrid"  # "hybrid" | "vector" | "bm25"
    safety_bm25_topk: int = 100
    safety_vector_topk: int = 80
    safety_rrf_k: int = 60
    safety_weights: tuple[float, float] = (1.3, 1.0)  # (bm25, vector)

    # --- NEW: medical-general strategy (vector-biased) ---
    general_mode: str = "hybrid"
    general_bm25_topk: int = 60
    general_vector_topk: int = 100
    general_rrf_k: int = 60
    general_weights: tuple[float, float] = (1.0, 1.2)

    # --- NEW: non-medical behavior ---
    non_medical_policy: str = "fallback"  # "fallback" | "refuse"
    
    reranker_kind: str = "simple"  # "simple" | "cross_encoder"
    ce_model_name_or_path: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ce_device: str = "cpu"
    ce_batch_size: int = 16
    ce_max_length: int = 384
    ce_rerank_topn: int = 80

    # --- final fusion knobs ---
    fusion_alpha_safety: float = 0.85
    fusion_alpha_general: float = 0.70

    rule_weight: float = 0.6     # w_rule
    risk_weight: float = 0.3     # w_risk

    # 如果 CE 不可用/失败，是否退回只用 prior+rule+risk
    allow_ce_fallback: bool = True


class HybridRAGPipeline:
    """
    统一融合版：
    1) 实体识别（drug/population）
    2) 意图/风险识别：safety_critical / medical_general / non_medical
    3) metadata 过滤 + (BM25 + Vector) hybrid 召回（粗召回 pool）
    4) rerank 精排
    5) 规则引擎强制补全（禁用/慎用等强约束）
    6) 合并证据：forced 优先，其次 rerank topK
    """

    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        paths = default_paths()

        self.entities = EntityExtractor(paths.aliases_path)
        self.store = VectorStore(paths.documents_path)
        self.rules = RuleEngine(paths.rules_path, self.store)
        self.intent = QueryIntentClassifier()
        self.reranker = build_reranker(
            self.cfg.reranker_kind,
            model_name_or_path=self.cfg.ce_model_name_or_path,
            device=self.cfg.ce_device,
            batch_size=self.cfg.ce_batch_size,
            max_length=self.cfg.ce_max_length,
            rerank_topn=self.cfg.ce_rerank_topn,
        )

    def answer(self, query: str, topk_recall: int = 12, topk_final: int = 5) -> Dict[str, Any]:
        ents = self.entities.extract(query)

        # 1) 规则兜底：先拿 forced evidence（强约束）
        forced = self.rules.apply(ents)

        # 2) 意图/风险判断（决定召回策略）
        intent_res = self.intent.classify(query, ents, forced_rules=forced)

        # 3) 召回池（给 rerank 更大空间）
        recall_k = self.cfg.recall_pool_k or topk_recall

        # 4) 选择策略参数（safety vs general）
        if intent_res.intent == "safety_critical":
            mode = self.cfg.safety_mode
            bm25_topk = self.cfg.safety_bm25_topk
            vector_topk = self.cfg.safety_vector_topk
            rrf_k = self.cfg.safety_rrf_k
            weights = self.cfg.safety_weights
        else:
            mode = self.cfg.general_mode
            bm25_topk = self.cfg.general_bm25_topk
            vector_topk = self.cfg.general_vector_topk
            rrf_k = self.cfg.general_rrf_k
            weights = self.cfg.general_weights

        # 5) 非医疗：可选择直接拒答
        if intent_res.intent == "non_medical" and self.cfg.non_medical_policy == "refuse":
            return {
                "query": query,
                "entities": ents,
                "intent": intent_res.__dict__,
                "forced": [f["rule"] for f in forced] if forced else [],
                "final_evidence": [],
                "draft_answer": "这个问题看起来不属于医疗用药/医学咨询范围。你可以换个与用药、人群、禁忌、剂量相关的问题试试。",
            }

        # 6) 召回（hybrid/vector/bm25 由 mode 控制）
        recall = self.store.search(
            query=query,
            topk=recall_k,
            filters={
                "drug": ents["drugs"][0] if ents.get("drugs") else None,
                "population": ents["populations"][0] if ents.get("populations") else None,
            },
            allow_generic_population=self.cfg.allow_generic_population,
            fallback_to_all=self.cfg.fallback_to_all,
            mode=mode,
            bm25_topk=bm25_topk,
            vector_topk=vector_topk,
            rrf_k=rrf_k,
            weights=weights,
        )

        # 7) rerank（只对 recall）
        reranked = self.reranker.rerank(query, recall)
        # ✅ NEW: final fusion (CE/prior/rule/risk) re-sort
        reranked = self._final_fuse_sort(reranked, intent_res)

        # 8) 合并证据（去重）：forced 优先
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

        # A+: 无实体场景白名单，避免把用药风险条款混入“症状类问题”
        if not ents.get("drugs") and not ents.get("populations"):
            allow = {"symptom", "general_principle"}
            filtered = [e for e in final if (e.get("metadata", {}) or {}).get("section") in allow]
            if filtered:
                final = filtered

        draft = self._draft_answer_template(query, ents, forced, final)

        return {
            "query": query,
            "entities": ents,
            "intent": intent_res.__dict__,
            "forced": [f["rule"] for f in forced] if forced else [],
            "final_evidence": final,
            "draft_answer": draft,
        }

    def _minmax_norm(self, xs: List[float]) -> List[float]:
        if not xs:
            return []
        mn = min(xs)
        mx = max(xs)
        if mx - mn < 1e-12:
            return [0.0 for _ in xs]
        return [(x - mn) / (mx - mn) for x in xs]

    def _rule_strength(self, h: Dict[str, Any]) -> float:
        md = (h.get("metadata") or {})
        sec = md.get("section")
        risk = md.get("risk_level")

        # 你可以按你数据再细调
        if sec == "contraindication":
            return 1.0
        if risk in ("warning", "caution"):
            return 0.6
        return 0.0

    def _risk_scalar(self, risk: str) -> float:
        if risk == "high":
            return 1.0
        if risk == "medium":
            return 0.4
        return 0.0

    def _final_fuse_sort(self, hits: List[Dict[str, Any]], intent_res: Any) -> List[Dict[str, Any]]:
        """
        输入：已经 rerank 过（可能是 CE，也可能是 simple）的 hits
        输出：根据最终融合公式重排后的 hits（每条带 final_score）
        """
        if not hits:
            return []

        # prior：优先 fused_score，其次 vec_score/bm25_score
        prior_raw = []
        ce_raw = []
        rule_strength = []
        for h in hits:
            fs = float(h.get("fused_score", 0.0) or 0.0)
            if fs <= 0:
                # 兼容非hybrid
                fs = float(h.get("vec_score", 0.0) or 0.0) + float(h.get("bm25_score", 0.0) or 0.0)
            prior_raw.append(float(__import__("math").log1p(max(1e-9, fs))))

            ce_raw.append(float(h.get("rerank_score", 0.0) or 0.0))
            rule_strength.append(float(self._rule_strength(h)))

        prior_n = self._minmax_norm(prior_raw)
        ce_n = self._minmax_norm(ce_raw)

        # alpha 按 intent 走
        if getattr(intent_res, "intent", None) == "safety_critical":
            alpha = float(self.cfg.fusion_alpha_safety)
        else:
            alpha = float(self.cfg.fusion_alpha_general)

        # risk 乘子
        r = self._risk_scalar(getattr(intent_res, "risk", "low"))
        risk_mult = 1.0 + float(self.cfg.risk_weight) * r

        out = []
        for i, h in enumerate(hits):
            base = alpha * ce_n[i] + (1.0 - alpha) * prior_n[i]
            rule_mult = 1.0 + float(self.cfg.rule_weight) * rule_strength[i]
            final_score = base * rule_mult * risk_mult

            h2 = dict(h)
            h2["final_score"] = float(final_score)
            h2["_final_debug"] = {
                "alpha": alpha,
                "ce_n": ce_n[i],
                "prior_n": prior_n[i],
                "rule_strength": rule_strength[i],
                "rule_mult": rule_mult,
                "risk_mult": risk_mult,
                "base": base,
            }
            out.append(h2)

        out.sort(key=lambda x: x["final_score"], reverse=True)
        return out

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
