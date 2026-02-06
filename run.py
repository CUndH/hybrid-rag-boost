from __future__ import annotations

from rich import print
from rich.console import Console
from rich.text import Text

from medrag.pipeline import HybridRAGPipeline, PipelineConfig

console = Console()

DEBUG = True  # True: 打印融合调试信息；False: 更简洁


def _safe_get(d, key, default=None):
    try:
        return d.get(key, default)
    except Exception:
        return default


def _fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return f"{x}"


def main():
    print("[bold cyan]Med Hybrid RAG (CLI)[/bold cyan]")
    print("示例：")
    print("  - 孕期高血压能不能用络活喜5mg？")
    print("  - 孕妇高血压能不能用依那普利？")
    print("  - 高血压平时注意什么？")
    print("输入 quit 退出。\n")

    # ✅ 推荐默认配置：意图分流 + hybrid 召回 +（你若启用CE，在 PipelineConfig 里配置）
    pipe = HybridRAGPipeline(
        PipelineConfig(
            recall_pool_k=60,

            # 非医疗问题：demo阶段建议 fallback（别太硬）
            non_medical_policy="fallback",

            # safety / general 召回策略（你可按需调）
            safety_mode="hybrid",
            safety_bm25_topk=100,
            safety_vector_topk=80,
            safety_rrf_k=60,
            safety_weights=(1.3, 1.0),

            general_mode="hybrid",
            general_bm25_topk=60,
            general_vector_topk=100,
            general_rrf_k=60,
            general_weights=(1.0, 1.2),

            # 如果你已经能用 cross-encoder，再打开下面这些（否则保持 simple）
            reranker_kind="cross_encoder",
            ce_model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2",
            ce_device="cpu",  # 有GPU改成 "cuda"
            ce_batch_size=8,
            ce_max_length=256,
            ce_rerank_topn=20,

            # 若你已实现“最终融合公式”，可在 PipelineConfig 里加这些参数并在 pipeline 内生效
            fusion_alpha_safety=0.85,
            fusion_alpha_general=0.70,
            rule_weight=0.6,
            risk_weight=0.3,
        )
    )

    while True:
        q = input("You> ").strip()
        if not q:
            continue
        if q.lower() in ("q", "quit", "exit"):
            break

        out = pipe.answer(q, topk_recall=12, topk_final=5)

        # 1) Query & Intent
        print("\n[bold cyan]=== Query ===[/bold cyan]")
        print(q)

        intent = out.get("intent", {}) or {}
        print("\n[bold cyan]=== Intent & Risk ===[/bold cyan]")
        print(
            f"intent={intent.get('intent')}  "
            f"risk={intent.get('risk')}  "
            f"reasons={intent.get('reasons')}"
        )

        # 2) Entities & Matches
        ents = out.get("entities", {}) or {}
        print("\n[bold cyan]=== Entities ===[/bold cyan]")
        print({
            "drugs": ents.get("drugs"),
            "populations": ents.get("populations"),
            "drug_conf": ents.get("drug_conf"),
            "population_conf": ents.get("population_conf"),
        })

        matches = ents.get("matches") or []
        if matches:
            print("\n[bold cyan]--- Entity Matches ---[/bold cyan]")
            for m in matches:
                try:
                    print(
                        f"- {m['kind']:10s} "
                        f"{m['canonical']}  "
                        f"(alias='{m['alias']}', score={float(m['score']):.2f}, span={m.get('span')})"
                    )
                except Exception:
                    print(f"- {m}")

        # 3) Forced Rules
        print("\n[bold cyan]=== Forced (Rules) ===[/bold cyan]")
        forced_rules = out.get("forced") or []
        if forced_rules:
            for f in forced_rules:
                print(f"- {f}")
        else:
            print("(none)")

        # 4) Recall Strategy (from cfg)
        print("\n[bold cyan]=== Recall Strategy ===[/bold cyan]")
        is_safety = intent.get("intent") == "safety_critical"
        mode = pipe.cfg.safety_mode if is_safety else pipe.cfg.general_mode
        print(f"mode={mode}  recall_pool_k={pipe.cfg.recall_pool_k}")

        if is_safety:
            print(
                f"bm25_topk={pipe.cfg.safety_bm25_topk}  vector_topk={pipe.cfg.safety_vector_topk}  "
                f"rrf_k={pipe.cfg.safety_rrf_k}  weights={pipe.cfg.safety_weights}"
            )
        else:
            print(
                f"bm25_topk={pipe.cfg.general_bm25_topk}  vector_topk={pipe.cfg.general_vector_topk}  "
                f"rrf_k={pipe.cfg.general_rrf_k}  weights={pipe.cfg.general_weights}"
            )

        # 5) Final Evidence
        print("\n[bold cyan]=== Final Evidence (Ranked) ===[/bold cyan]")
        evidence = out.get("final_evidence") or []
        if not evidence:
            print("(none)")
        else:
            for i, h in enumerate(evidence, 1):
                md = (h.get("metadata") or {})
                dbg = (h.get("_final_debug") or {})

                title = h.get("title", "")
                doc_id = h.get("id", "")
                sec = md.get("section")

                # 分数（兼容字段缺失）
                final_score = h.get("final_score", None)
                rerank_score = h.get("rerank_score", 0.0)
                fused_score = h.get("fused_score", 0.0)
                vec_score = h.get("vec_score", 0.0)
                bm25_score = h.get("bm25_score", 0.0)
                source = h.get("source", "-")
                rmodel = h.get("rerank_model", "-")

                print(f"[bold]{i}.[/bold] {title}  (id={doc_id})")
                print(
                    f"    src={source}  section={sec}  rmodel={rmodel}"
                )

                # 展示融合前后的关键分数
                if final_score is not None:
                    print(
                        f"    score: final={_fmt(final_score)} | "
                        f"ce={_fmt(rerank_score)} | prior(fused)={_fmt(fused_score)} | "
                        f"vec={_fmt(vec_score)} bm25={_fmt(bm25_score)}"
                    )
                else:
                    print(
                        f"    score: rerank={_fmt(rerank_score)} | fused={_fmt(fused_score)} | "
                        f"vec={_fmt(vec_score)} bm25={_fmt(bm25_score)}"
                    )

                # 如果你实现了最终融合公式，会有 debug 字段
                if DEBUG and dbg:
                    print(
                        f"    boost: rule_mult={dbg.get('rule_mult', 1.0):.2f} | "
                        f"risk_mult={dbg.get('risk_mult', 1.0):.2f} | "
                        f"alpha={dbg.get('alpha')}"
                    )
                    print(
                        f"    parts: ce_n={dbg.get('ce_n', 0.0):.4f} | "
                        f"prior_n={dbg.get('prior_n', 0.0):.4f} | "
                        f"rule_strength={dbg.get('rule_strength', 0.0):.2f} | "
                        f"base={dbg.get('base', 0.0):.4f}"
                    )

                text = h.get("text", "")
                if text:
                    print(f"    text: {text}")

        # 6) Draft Answer
        print("\n[bold cyan]=== Draft Answer ===[/bold cyan]")
        print(out.get("draft_answer"))

        print("-" * 90)


if __name__ == "__main__":
    main()
