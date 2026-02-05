from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from medrag.pipeline import HybridRAGPipeline  # noqa


@dataclass
class CaseResult:
    case_id: str
    query: str

    entity_ok: bool
    conf_ok: bool
    forced_ok: bool
    evidence_ok: bool
    sections_ok: bool
    negative_ok: bool

    got_entities: Dict[str, Any]
    got_forced: List[str]
    got_evidence_ids: List[str]
    got_sections: List[str]
    score_stats: Dict[str, float]

    miss_entities: Dict[str, List[str]]
    miss_conf: Dict[str, Tuple[float, float]]
    miss_evidence_any: List[str]
    miss_sections: List[str]
    hit_negative_sections: List[str]
    miss_forced_contains: List[str]

    suggestions: List[str]


def _load_cases(path: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cases.append(json.loads(line))
    return cases


def _subset_ok(expected: List[str], got: List[str]) -> Tuple[bool, List[str]]:
    miss = [x for x in expected if x not in got]
    return (len(miss) == 0), miss


def _any_ok(expected_any: List[str], got: List[str]) -> Tuple[bool, List[str]]:
    if not expected_any:
        return True, []
    ok = any(x in got for x in expected_any)
    miss = [] if ok else expected_any[:]
    return ok, miss


def _contains_all(needles: List[str], haystack: str) -> Tuple[bool, List[str]]:
    miss = [n for n in needles if n and (n not in haystack)]
    return (len(miss) == 0), miss


def _collect_sections(evidence: List[Dict[str, Any]]) -> List[str]:
    secs = []
    for e in evidence:
        md = e.get("metadata") or {}
        sec = md.get("section")
        if sec:
            secs.append(sec)
    seen = set()
    out = []
    for s in secs:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _score_stats(evidence: List[Dict[str, Any]]) -> Dict[str, float]:
    if not evidence:
        return {"n": 0, "vec_max": 0.0, "vec_avg": 0.0, "rerank_max": 0.0, "rerank_avg": 0.0}

    vec = [float(e.get("vec_score", 0.0)) for e in evidence]
    rr = [float(e.get("rerank_score", 0.0)) for e in evidence]
    n = len(evidence)
    return {
        "n": n,
        "vec_max": max(vec),
        "vec_avg": sum(vec) / n,
        "rerank_max": max(rr),
        "rerank_avg": sum(rr) / n,
    }


def _make_suggestions(
    *,
    entity_ok: bool,
    conf_ok: bool,
    forced_ok: bool,
    evidence_ok: bool,
    sections_ok: bool,
    negative_ok: bool,
    miss_entities: Dict[str, List[str]],
    miss_conf: Dict[str, Tuple[float, float]],
    miss_evidence_any: List[str],
    miss_sections: List[str],
    hit_negative_sections: List[str],
    miss_forced_contains: List[str],
) -> List[str]:
    sug: List[str] = []

    if not entity_ok:
        if miss_entities.get("drugs"):
            sug.append("实体识别缺药物：建议在 aliases.json 增加该药物别名（含中英文/商品名/剂量写法）。")
        if miss_entities.get("populations"):
            sug.append("实体识别缺人群：建议在 aliases.json 增加人群同义词（孕期/妊娠/怀孕等）。")

    if entity_ok and not conf_ok:
        for k, (got, need) in miss_conf.items():
            sug.append(f"置信度不足({k}): got={got:.2f} < need={need:.2f}。建议提高该实体 alias priority 或增加更长别名（长词更稳）。")

    if not forced_ok:
        sug.append("规则未触发或文案不匹配：检查 rules.json 的 when 条件（drug/population）是否能匹配，以及 message 是否包含关键字。")

    if not evidence_ok:
        sug.append("未命中关键证据：检查 documents.json 是否缺少该条款 chunk，或 metadata.drug/population/section 标注不完整导致过滤/排序失败。")

    if not sections_ok:
        sug.append("关键 section 缺失：建议补充对应 section 的 chunk（如 contraindication/pregnancy_warning），并确保 metadata.section 填对。")

    if not negative_ok:
        sug.append(f"出现不该出现的 section：{hit_negative_sections}。建议：①降低这些 chunk 的相似度噪音（标题更精确）；②增加 BM25/metadata 约束；③调小 topk_final。")

    if not sug:
        sug.append("✅ 全通过：可以开始扩充评测集（更多别名/多药/反例）并逐步引入 BM25 融合召回。")

    return sug


def run_eval(
    cases_path: str,
    topk_recall: int = 12,
    topk_final: int = 5,
    verbose: bool = False,
) -> int:
    pipe = HybridRAGPipeline()

    cases = _load_cases(cases_path)
    if not cases:
        print(f"No cases found in {cases_path}")
        return 2

    results: List[CaseResult] = []

    for c in cases:
        cid = c.get("id", "UNKNOWN")
        q = c["query"]

        out = pipe.answer(q, topk_recall=topk_recall, topk_final=topk_final)

        ents = out.get("entities", {}) or {}
        got_drugs = ents.get("drugs", []) or []
        got_pops = ents.get("populations", []) or []
        got_drug_conf = float(ents.get("drug_conf") or 0.0)
        got_pop_conf = float(ents.get("population_conf") or 0.0)

        got_forced = out.get("forced", []) or []
        forced_text = "\n".join(got_forced)

        evidence = out.get("final_evidence", []) or []
        got_ids = [e.get("id") for e in evidence if e.get("id")]
        got_secs = _collect_sections(evidence)
        stats = _score_stats(evidence)

        # expectations
        exp_ents = c.get("expect_entities", {}) or {}
        exp_drugs = exp_ents.get("drugs", []) or []
        exp_pops = exp_ents.get("populations", []) or []

        exp_min_conf = c.get("expect_min_conf", {}) or {}
        need_drug_conf = float(exp_min_conf.get("drug", 0.0))
        need_pop_conf = float(exp_min_conf.get("population", 0.0))

        exp_forced_contains = c.get("expect_forced_contains", []) or []
        exp_evidence_any = c.get("expect_evidence_ids_any", []) or []
        exp_secs = c.get("expect_sections_contains", []) or []
        exp_secs_excl = c.get("expect_sections_exclude", []) or []

        drugs_ok, miss_drugs = _subset_ok(exp_drugs, got_drugs)
        pops_ok, miss_pops = _subset_ok(exp_pops, got_pops)
        entity_ok = drugs_ok and pops_ok

        conf_ok = (got_drug_conf >= need_drug_conf) and (got_pop_conf >= need_pop_conf)
        miss_conf: Dict[str, Tuple[float, float]] = {}
        if got_drug_conf < need_drug_conf:
            miss_conf["drug"] = (got_drug_conf, need_drug_conf)
        if got_pop_conf < need_pop_conf:
            miss_conf["population"] = (got_pop_conf, need_pop_conf)

        forced_ok, miss_forced = _contains_all(exp_forced_contains, forced_text)
        evidence_ok, miss_any = _any_ok(exp_evidence_any, got_ids)
        sections_ok, miss_secs = _subset_ok(exp_secs, got_secs)

        hit_negative = [s for s in exp_secs_excl if s in got_secs]
        negative_ok = len(hit_negative) == 0

        suggestions = _make_suggestions(
            entity_ok=entity_ok,
            conf_ok=conf_ok,
            forced_ok=forced_ok,
            evidence_ok=evidence_ok,
            sections_ok=sections_ok,
            negative_ok=negative_ok,
            miss_entities={"drugs": miss_drugs, "populations": miss_pops},
            miss_conf=miss_conf,
            miss_evidence_any=miss_any,
            miss_sections=miss_secs,
            hit_negative_sections=hit_negative,
            miss_forced_contains=miss_forced,
        )

        results.append(
            CaseResult(
                case_id=cid,
                query=q,
                entity_ok=entity_ok,
                conf_ok=conf_ok,
                forced_ok=forced_ok,
                evidence_ok=evidence_ok,
                sections_ok=sections_ok,
                negative_ok=negative_ok,
                got_entities={"drugs": got_drugs, "populations": got_pops, "drug_conf": got_drug_conf, "population_conf": got_pop_conf},
                got_forced=got_forced,
                got_evidence_ids=got_ids,
                got_sections=got_secs,
                score_stats=stats,
                miss_entities={"drugs": miss_drugs, "populations": miss_pops},
                miss_conf=miss_conf,
                miss_evidence_any=miss_any,
                miss_sections=miss_secs,
                hit_negative_sections=hit_negative,
                miss_forced_contains=miss_forced,
                suggestions=suggestions,
            )
        )

        if verbose:
            print(f"\n[{cid}] {q}")
            print(f"  entities: {results[-1].got_entities}  ok={entity_ok} conf_ok={conf_ok}")
            print(f"  forced: {got_forced}  ok={forced_ok}")
            print(f"  evidence_ids: {got_ids}  ok={evidence_ok}")
            print(f"  sections: {got_secs}  ok={sections_ok} negative_ok={negative_ok}")
            print(f"  score_stats: {stats}")

            if not entity_ok:
                print(f"  MISS entities: {results[-1].miss_entities}")
            if not conf_ok:
                print(f"  MISS conf: {miss_conf}")
            if not forced_ok:
                print(f"  MISS forced_contains: {miss_forced}")
            if not evidence_ok:
                print(f"  MISS evidence_any: {miss_any}")
            if not sections_ok:
                print(f"  MISS sections: {miss_secs}")
            if not negative_ok:
                print(f"  HIT negative sections: {hit_negative}")
            print("  suggestions:")
            for s in suggestions:
                print(f"    - {s}")

    # aggregate metrics
    n = len(results)
    def rate(xs): return f"{sum(xs)}/{n} = {sum(xs)/n:.1%}"

    entity_pass = rate([r.entity_ok for r in results])
    conf_pass = rate([r.conf_ok for r in results])
    forced_pass = rate([r.forced_ok for r in results])
    evidence_pass = rate([r.evidence_ok for r in results])
    sections_pass = rate([r.sections_ok for r in results])
    negative_pass = rate([r.negative_ok for r in results])

    overall = [r.entity_ok and r.conf_ok and r.forced_ok and r.evidence_ok and r.sections_ok and r.negative_ok for r in results]
    overall_pass = rate(overall)

    avg_n = sum(r.score_stats["n"] for r in results) / n
    avg_vec = sum(r.score_stats["vec_avg"] for r in results) / n
    avg_rr = sum(r.score_stats["rerank_avg"] for r in results) / n

    # section distribution
    sec_count: Dict[str, int] = {}
    for r in results:
        for s in r.got_sections:
            sec_count[s] = sec_count.get(s, 0) + 1
    top_secs = sorted(sec_count.items(), key=lambda x: (-x[1], x[0]))[:10]

    print("\n=== EVAL SUMMARY (A+) ===")
    print(f"cases: {n}")
    print(f"entity_pass:    {entity_pass}")
    print(f"conf_pass:      {conf_pass}")
    print(f"forced_pass:    {forced_pass}")
    print(f"evidence_pass:  {evidence_pass}")
    print(f"sections_pass:  {sections_pass}")
    print(f"negative_pass:  {negative_pass}")
    print(f"overall_pass:   {overall_pass}")
    print("\n=== STATS ===")
    print(f"avg_final_evidence_n: {avg_n:.2f}")
    print(f"avg_vec_score:        {avg_vec:.4f}")
    print(f"avg_rerank_score:     {avg_rr:.4f}")
    print("\n=== TOP SECTIONS (coverage) ===")
    for s, k in top_secs:
        print(f"- {s}: {k}/{n}")

    fails = [r for r in results if not (r.entity_ok and r.conf_ok and r.forced_ok and r.evidence_ok and r.sections_ok and r.negative_ok)]
    if fails:
        print("\n=== FAILURES ===")
        for r in fails:
            print(f"- {r.case_id}: {r.query}")
            if not r.entity_ok:
                print(f"    entity MISS: {r.miss_entities}   got={r.got_entities}")
            if not r.conf_ok:
                print(f"    conf MISS: {r.miss_conf}   got={r.got_entities}")
            if not r.forced_ok:
                print(f"    forced MISS: need_contains={r.miss_forced_contains}   got={r.got_forced}")
            if not r.evidence_ok:
                print(f"    evidence MISS: need_any={r.miss_evidence_any}   got_ids={r.got_evidence_ids}")
            if not r.sections_ok:
                print(f"    sections MISS: need={r.miss_sections}   got={r.got_sections}")
            if not r.negative_ok:
                print(f"    negative HIT: {r.hit_negative_sections}   got={r.got_sections}")
            print("    suggestions:")
            for s in r.suggestions:
                print(f"      - {s}")

    return 0 if not fails else 1


if __name__ == "__main__":
    cases_path = os.path.join(REPO_ROOT, "eval", "cases.jsonl")
    verbose = "--verbose" in sys.argv
    rc = run_eval(cases_path=cases_path, topk_recall=12, topk_final=5, verbose=verbose)
    raise SystemExit(rc)
