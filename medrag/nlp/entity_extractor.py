from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

from medrag.nlp.normalize import normalize_for_match, normalize_text


@dataclass
class Match:
    canonical: str
    alias: str
    start: int
    end: int
    score: float
    kind: str  # "drug" / "population"


class EntityExtractor:
    """
    升级版实体识别（不靠大模型）：
    - alias 表（支持 list 或 {aliases, priority}）
    - 处理剂量噪音（5mg / 片 / ml 等）
    - 支持多实体、长词优先、重叠消解
    - 输出置信度（用于决定是否强过滤）
    """

    def __init__(self, aliases_path: str):
        with open(aliases_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.drug_terms = self._build_terms(raw.get("drugs", {}), kind="drug")
        self.pop_terms = self._build_terms(raw.get("populations", {}), kind="population")

        # 合并词表：用于统一扫描
        self.all_terms = sorted(self.drug_terms + self.pop_terms, key=lambda x: (-len(x[0]), -x[2], x[1]))
        # term tuple: (alias_norm, canonical, priority, kind, alias_raw)

    def _build_terms(self, block: Dict[str, Any], kind: str) -> List[Tuple[str, str, int, str, str]]:
        terms: List[Tuple[str, str, int, str, str]] = []
        for canonical, v in block.items():
            if isinstance(v, list):
                aliases = v
                priority = 10 if kind == "population" else 5
            else:
                aliases = v.get("aliases", [])
                priority = int(v.get("priority", 10 if kind == "population" else 5))

            # canonical 也作为 alias
            aliases = list(aliases) + [canonical]
            for a in aliases:
                a_raw = str(a)
                a_norm = normalize_for_match(a_raw)
                if not a_norm:
                    continue
                terms.append((a_norm, canonical, priority, kind, a_raw))
        # 去重（同一个 alias_norm 可能重复）
        uniq = {}
        for t in terms:
            key = (t[0], t[1], t[3])
            uniq[key] = t
        return list(uniq.values())

    def extract(self, query: str) -> Dict[str, Any]:
        """
        返回：
        {
          "drugs": [...],
          "populations": [...],
          "drug_conf": 0~1,
          "population_conf": 0~1,
          "matches": [ {kind, canonical, alias, score, span} ...]
        }
        """
        q_norm = normalize_for_match(query)
        if not q_norm:
            return {"drugs": [], "populations": [], "drug_conf": 0.0, "population_conf": 0.0, "matches": []}

        matches = self._scan_matches(q_norm)

        # 重叠消解：长词优先 + 分数优先
        kept = self._non_overlapping(matches)

        drugs, drug_conf = self._collect(kept, kind="drug", q_norm=q_norm)
        pops, pop_conf = self._collect(kept, kind="population", q_norm=q_norm)

        return {
            "drugs": drugs,
            "populations": pops,
            "drug_conf": drug_conf,
            "population_conf": pop_conf,
            "matches": [
                {
                    "kind": m.kind,
                    "canonical": m.canonical,
                    "alias": m.alias,
                    "score": round(m.score, 4),
                    "span": [m.start, m.end],
                }
                for m in kept
            ],
        }

    def _scan_matches(self, q_norm: str) -> List[Match]:
        found: List[Match] = []
        for alias_norm, canonical, priority, kind, alias_raw in self.all_terms:
            # 找所有出现位置
            start = 0
            while True:
                idx = q_norm.find(alias_norm, start)
                if idx == -1:
                    break
                end = idx + len(alias_norm)
                score = self._score(alias_norm, canonical, priority, kind, q_norm)
                found.append(Match(canonical=canonical, alias=alias_raw, start=idx, end=end, score=score, kind=kind))
                start = idx + 1
        # 按 score 降序，span 长度降序
        found.sort(key=lambda m: (-m.score, -(m.end - m.start), m.start))
        return found

    def _score(self, alias_norm: str, canonical: str, priority: int, kind: str, q_norm: str) -> float:
        """
        一个可解释的置信度启发式：
        - 命中长度越长越可信
        - priority 越高越可信（你可调）
        - 药物命中给更高权重（相对人群）
        """
        L = len(alias_norm)
        base = min(1.0, 0.2 + L / 10.0)  # 长词更高
        p = min(1.0, 0.5 + priority / 20.0)
        kind_boost = 1.0 if kind == "drug" else 0.9
        # 最终
        return float(min(1.0, base * p * kind_boost))

    def _non_overlapping(self, matches: List[Match]) -> List[Match]:
        taken = [False] * 20000  # 足够长；q_norm 通常很短
        kept: List[Match] = []
        for m in matches:
            # 检查 span 是否与已选重叠
            overlap = False
            for i in range(m.start, m.end):
                if i < len(taken) and taken[i]:
                    overlap = True
                    break
            if overlap:
                continue
            kept.append(m)
            for i in range(m.start, m.end):
                if i < len(taken):
                    taken[i] = True
        return kept

    def _collect(self, kept: List[Match], kind: str, q_norm: str) -> Tuple[List[str], float]:
        items: List[Match] = [m for m in kept if m.kind == kind]
        if not items:
            return [], 0.0

        # canonical 去重保序
        seen = set()
        out = []
        scores = []
        for m in items:
            if m.canonical in seen:
                continue
            seen.add(m.canonical)
            out.append(m.canonical)
            scores.append(m.score)

        # 置信度：取 top1（简单但够用）；也可用平均/最大
        conf = float(max(scores)) if scores else 0.0
        return out, conf
