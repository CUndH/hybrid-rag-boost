from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Iterable

from medrag.nlp.normalize import normalize_for_match


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
    - aliases 支持多种格式（dict/list，key兼容）
    - 处理剂量噪音（由 normalize_for_match 负责）
    - 支持多实体、长词优先、重叠消解
    - 输出置信度（用于决定是否强过滤）
    """

    def __init__(self, aliases_path: str):
        with open(aliases_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # ✅ 兼容顶层 key 命名：drugs/populations 或 drug/population
        drugs_block = (
            raw.get("drugs")
            if isinstance(raw, dict)
            else None
        )
        pops_block = (
            raw.get("populations")
            if isinstance(raw, dict)
            else None
        )

        if drugs_block is None and isinstance(raw, dict):
            drugs_block = raw.get("drug")
        if pops_block is None and isinstance(raw, dict):
            pops_block = raw.get("population")

        drugs_block = drugs_block or {}
        pops_block = pops_block or {}

        self.drug_terms = self._build_terms(drugs_block, kind="drug")
        self.pop_terms = self._build_terms(pops_block, kind="population")

        # 合并词表：用于统一扫描
        # term tuple: (alias_norm, canonical, priority, kind, alias_raw)
        self.all_terms = sorted(
            self.drug_terms + self.pop_terms,
            key=lambda x: (-len(x[0]), -x[2], x[1])
        )

    # -----------------------------
    # aliases parsing / term build
    # -----------------------------
    def _build_terms(self, block: Any, kind: str) -> List[Tuple[str, str, int, str, str]]:
        """
        支持三种 block 形式：
        A) dict: { canonical: [aliases...] }
        B) dict: { canonical: { "aliases": [...], "priority": 7 } }
        C) list: [ { "alias": "...", "canonical": "...", "priority": 7 }, ... ]
        """
        default_priority = 10 if kind == "population" else 5
        terms: List[Tuple[str, str, int, str, str]] = []

        # ---- C) list form ----
        if isinstance(block, list):
            for it in block:
                if not isinstance(it, dict):
                    continue
                canonical = it.get("canonical") or it.get("target") or it.get("name")
                alias = it.get("alias")
                if not canonical or not alias:
                    continue
                priority = int(it.get("priority", default_priority))
                self._add_alias_terms(
                    terms=terms,
                    canonical=str(canonical),
                    aliases=[str(alias)],
                    priority=priority,
                    kind=kind,
                )
            return self._dedup_terms(terms)

        # ---- A/B) dict form ----
        if not isinstance(block, dict):
            return []

        for canonical, v in block.items():
            priority = default_priority
            aliases: List[str] = []

            if isinstance(v, list):
                aliases = [str(x) for x in v]
                priority = default_priority
            elif isinstance(v, dict):
                aliases = [str(x) for x in (v.get("aliases") or [])]
                priority = int(v.get("priority", default_priority))
            elif isinstance(v, str):
                # 兼容有人写成 {canonical: "alias"} 的情况
                aliases = [v]
                priority = default_priority
            else:
                # 其他类型忽略
                continue

            # canonical 也作为 alias（有助于英文/拼音）
            aliases = list(aliases) + [str(canonical)]

            self._add_alias_terms(
                terms=terms,
                canonical=str(canonical),
                aliases=aliases,
                priority=priority,
                kind=kind,
            )

        return self._dedup_terms(terms)

    def _add_alias_terms(
        self,
        terms: List[Tuple[str, str, int, str, str]],
        canonical: str,
        aliases: List[str],
        priority: int,
        kind: str,
    ) -> None:
        for a in aliases:
            a_raw = str(a)
            a_norm = normalize_for_match(a_raw)
            if not a_norm:
                continue
            terms.append((a_norm, canonical, int(priority), kind, a_raw))

    def _dedup_terms(self, terms: List[Tuple[str, str, int, str, str]]) -> List[Tuple[str, str, int, str, str]]:
        # 去重（同一个 alias_norm 可能重复）
        uniq = {}
        for t in terms:
            key = (t[0], t[1], t[3])
            # 若重复，保留 priority 更高的
            if key not in uniq or t[2] > uniq[key][2]:
                uniq[key] = t
        return list(uniq.values())

    # -----------------------------
    # main extraction
    # -----------------------------
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

        # 重叠消解：按 score/长度优先挑选
        kept = self._non_overlapping(matches, q_len=len(q_norm))

        drugs, drug_conf = self._collect(kept, kind="drug")
        pops, pop_conf = self._collect(kept, kind="population")

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
                score = self._score(alias_norm=alias_norm, priority=priority, kind=kind)
                found.append(
                    Match(
                        canonical=canonical,
                        alias=alias_raw,
                        start=idx,
                        end=end,
                        score=score,
                        kind=kind,
                    )
                )
                start = idx + 1

        # 按 score 降序，span 长度降序，再按出现位置
        found.sort(key=lambda m: (-m.score, -(m.end - m.start), m.start))
        return found

    def _score(self, alias_norm: str, priority: int, kind: str) -> float:
        """
        可解释置信度启发式：
        - 命中长度越长越可信
        - priority 越高越可信
        - 药物命中给更高权重（相对人群）
        """
        L = len(alias_norm)
        base = min(1.0, 0.2 + L / 10.0)  # 长词更高
        p = min(1.0, 0.5 + priority / 20.0)
        kind_boost = 1.0 if kind == "drug" else 0.9
        return float(min(1.0, base * p * kind_boost))

    def _non_overlapping(self, matches: List[Match], q_len: int) -> List[Match]:
        # ✅ 动态长度，避免固定 20000 的隐患/浪费
        taken = [False] * (q_len + 1)
        kept: List[Match] = []

        for m in matches:
            # 检查 span 是否与已选重叠
            overlap = False
            for i in range(m.start, m.end):
                if 0 <= i < len(taken) and taken[i]:
                    overlap = True
                    break
            if overlap:
                continue

            kept.append(m)
            for i in range(m.start, m.end):
                if 0 <= i < len(taken):
                    taken[i] = True

        return kept

    def _collect(self, kept: List[Match], kind: str) -> Tuple[List[str], float]:
        items: List[Match] = [m for m in kept if m.kind == kind]
        if not items:
            return [], 0.0

        # canonical 去重保序
        seen = set()
        out: List[str] = []
        scores: List[float] = []

        for m in items:
            if m.canonical in seen:
                continue
            seen.add(m.canonical)
            out.append(m.canonical)
            scores.append(m.score)

        conf = float(max(scores)) if scores else 0.0
        return out, conf
