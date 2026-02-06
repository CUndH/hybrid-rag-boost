from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class IntentResult:
    intent: str                 # "safety_critical" | "medical_general" | "non_medical"
    risk: str                   # "high" | "medium" | "low"
    reasons: List[str]


class QueryIntentClassifier:
    """
    轻量意图/风险识别（无需模型）：
    - 结合实体识别结果 + query 关键词
    - 目的：为 pipeline 选择不同策略（召回模式/权重/白名单等）
    """

    # 你可以持续增补这些词表
    SAFETY_KEYWORDS = (
        "能不能", "可以吗", "可不可以", "能否", "可以用", "能用", "能吃", "能服",
        "禁忌", "禁用", "慎用", "风险", "不良反应", "副作用", "相互作用", "冲突",
        "剂量", "用量", "mg", "毫克", "减量", "调整剂量",
    )

    POP_KEYWORDS = (
        "孕", "怀孕", "孕妇", "妊娠", "哺乳", "儿童", "小儿", "老人", "老年",
        "肝", "肾", "肝功能", "肾功能", "透析",
    )

    MEDICAL_KEYWORDS = (
        "症状", "原因", "原理", "机制", "诊断", "检查", "治疗", "用药", "指南", "适应症",
        "高血压", "糖尿病", "发烧", "咳嗽", "头痛", "过敏",
    )

    def classify(self, query: str, ents: Dict[str, Any], forced_rules: List[Dict[str, Any]] | None = None) -> IntentResult:
        q = (query or "").strip()
        forced_rules = forced_rules or []
        reasons: List[str] = []

        has_drug = bool(ents.get("drugs"))
        has_pop = bool(ents.get("populations"))

        # 规则触发本身就是强信号
        has_forced = len(forced_rules) > 0
        if has_forced:
            reasons.append("forced_rules_triggered")

        # 关键词信号
        has_safety_kw = any(k in q for k in self.SAFETY_KEYWORDS)
        if has_safety_kw:
            reasons.append("safety_keywords")

        has_pop_kw = any(k in q for k in self.POP_KEYWORDS)
        if has_pop_kw:
            reasons.append("population_keywords")

        has_med_kw = any(k in q for k in self.MEDICAL_KEYWORDS)
        if has_med_kw:
            reasons.append("medical_keywords")

        # 判定逻辑（可解释、可调）
        # 1) 高风险：药物+人群 或 强安全关键词 或 规则触发
        if has_forced or (has_drug and (has_pop or has_safety_kw or has_pop_kw)) or (has_safety_kw and (has_drug or has_pop_kw)):
            return IntentResult(intent="safety_critical", risk="high", reasons=reasons)

        # 2) 泛医疗：没有明确药，但属于医疗问题（或有人群但没识别药）
        if has_med_kw or has_pop or has_pop_kw:
            return IntentResult(intent="medical_general", risk="medium", reasons=reasons)

        # 3) 非医疗/闲聊
        return IntentResult(intent="non_medical", risk="low", reasons=reasons)
