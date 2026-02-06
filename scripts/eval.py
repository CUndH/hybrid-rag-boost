from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Tuple, Optional

from medrag.pipeline import HybridRAGPipeline, PipelineConfig


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def recall_at_k(pred_ids: List[str], gold_ids: List[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    topk = set(pred_ids[:k])
    hit = any(g in topk for g in gold_ids)
    return 1.0 if hit else 0.0


def mrr_at_k(pred_ids: List[str], gold_ids: List[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    for rank, pid in enumerate(pred_ids[:k], start=1):
        if pid in gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(pred_ids: List[str], gold_ids: List[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    # binary relevance
    def dcg(ids: List[str]) -> float:
        s = 0.0
        for i, pid in enumerate(ids[:k], start=1):
            rel = 1.0 if pid in gold_ids else 0.0
            if rel > 0:
                s += rel / ( (i + 1) ** 0.0 )  # 也可用 log2(i+1)，这里简化
                # 如果你想标准 ndcg： rel / log2(i+1)
        return s

    # 用标准 log2
    import math
    def dcg_log(ids: List[str]) -> float:
        s = 0.0
        for i, pid in enumerate(ids[:k], start=1):
            rel = 1.0 if pid in gold_ids else 0.0
            if rel > 0:
                s += rel / math.log2(i + 1)
        return s

    ideal = gold_ids[:k]
    idcg = dcg_log(ideal)
    if idcg <= 1e-12:
        return 0.0
    return dcg_log(pred_ids) / idcg


def is_safety_section(section: Optional[str]) -> bool:
    return section in ("contraindication", "warning")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=os.path.join("medrag", "data", "eval", "eval.jsonl"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--topk_recall", type=int, default=60)
    ap.add_argument("--topk_final", type=int, default=5)
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "vector", "bm25"])
    ap.add_argument("--reranker", default="simple", choices=["simple", "cross_encoder"])
    ap.add_argument("--log", default=os.path.join("medrag", "data", "eval", "run_logs.jsonl"))
    args = ap.parse_args()

    # 统一强制 safety/general 使用同一 mode，便于做 A/B
    cfg = PipelineConfig(
        recall_pool_k=args.topk_recall,
        safety_mode=args.mode,
        general_mode=args.mode,
        non_medical_policy="fallback",
        reranker_kind=args.reranker,
        # 你若用 CE，可在这里继续配置：ce_device/ce_rerank_topn 等
    )
    pipe = HybridRAGPipeline(cfg)

    cases = load_jsonl(args.eval)

    # metrics accum
    n_effective = 0
    sum_recall = 0.0
    sum_mrr = 0.0
    sum_ndcg = 0.0

    intent_total = 0
    intent_hit = 0
    risk_total = 0
    risk_hit = 0

    # safety coverage: 在 safety_critical 样本上，topK 是否出现 contraindication/warning
    safety_total = 0
    safety_cov = 0

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, "w", encoding="utf-8") as flog:
        for c in cases:
            q = c["query"]
            gold_ids = c.get("gold_ids") or []
            expect = c.get("expect") or {}

            t0 = time.time()
            out = pipe.answer(q, topk_recall=args.topk_recall, topk_final=args.topk_final)
            dt = time.time() - t0

            evidence = out.get("final_evidence") or []
            pred_ids = [e.get("id") for e in evidence if e.get("id")]

            # retrieval metrics（只对有 gold 的样本）
            if gold_ids:
                n_effective += 1
                sum_recall += recall_at_k(pred_ids, gold_ids, args.k)
                sum_mrr += mrr_at_k(pred_ids, gold_ids, args.k)
                sum_ndcg += ndcg_at_k(pred_ids, gold_ids, args.k)

            # intent/risk accuracy（可选）
            intent = (out.get("intent") or {}).get("intent")
            risk = (out.get("intent") or {}).get("risk")

            if "intent" in expect:
                intent_total += 1
                if intent == expect["intent"]:
                    intent_hit += 1

            if "risk" in expect:
                risk_total += 1
                if risk == expect["risk"]:
                    risk_hit += 1

            # safety coverage（仅 safety_critical）
            if intent == "safety_critical":
                safety_total += 1
                topk = evidence[:args.k]
                has_safety = any(is_safety_section((e.get("metadata") or {}).get("section")) for e in topk)
                if has_safety:
                    safety_cov += 1

            # log per case for replay
            flog.write(json.dumps({
                "id": c.get("id"),
                "query": q,
                "expect": expect,
                "pred": {
                    "intent": intent,
                    "risk": risk,
                    "top_ids": pred_ids[:args.k],
                    "top_sections": [((e.get("metadata") or {}).get("section")) for e in evidence[:args.k]],
                },
                "gold_ids": gold_ids,
                "latency_s": round(dt, 4),
                "mode": args.mode,
                "reranker": args.reranker,
            }, ensure_ascii=False) + "\n")

    # print summary
    print("\n=== EVAL SUMMARY ===")
    print(f"mode={args.mode}  reranker={args.reranker}  K={args.k}")
    if n_effective > 0:
        print(f"Recall@{args.k}: {sum_recall/n_effective:.3f}  ({n_effective} cases w/ gold)")
        print(f"MRR@{args.k}:    {sum_mrr/n_effective:.3f}")
        print(f"nDCG@{args.k}:   {sum_ndcg/n_effective:.3f}")
    else:
        print("No gold cases found; retrieval metrics skipped.")

    if intent_total > 0:
        print(f"Intent Acc:      {intent_hit/intent_total:.3f}  ({intent_total} labeled)")
    if risk_total > 0:
        print(f"Risk Acc:        {risk_hit/risk_total:.3f}  ({risk_total} labeled)")

    if safety_total > 0:
        print(f"SafetyCov@{args.k}: {safety_cov/safety_total:.3f}  ({safety_total} safety_critical queries)")

    print(f"Logs saved to: {args.log}")


if __name__ == "__main__":
    main()
