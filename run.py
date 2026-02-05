from rich import print
from medrag.pipeline import HybridRAGPipeline,PipelineConfig


def main():
    print("[bold cyan]Med Hybrid RAG (CLI)[/bold cyan]")
    print("示例：")
    print("  - 孕期高血压能不能用络活喜5mg？")
    print("  - 孕妇高血压能不能用依那普利？")
    print("输入 quit 退出。\n")

    pipe = HybridRAGPipeline(
        PipelineConfig(
            recall_mode="hybrid",   # 可改成 "vector" / "bm25" 做对比
            recall_pool_k=60,
            bm25_pool_k=80,
            vector_pool_k=80,
            bm25_weight=1.2,
            vector_weight=1.0,
        )
    )

    while True:
        q = input("You> ").strip()
        if not q:
            continue
        if q.lower() in ("q", "quit", "exit"):
            break

        out = pipe.answer(q, topk_recall=12, topk_final=5)

        print("\n[bold]=== Entities ===[/bold]")
        print({k: out["entities"][k] for k in ["drugs","populations","drug_conf","population_conf"]})

        print("\n[bold]=== Matches ===[/bold]")
        for m in out["entities"].get("matches", []):
            print(f"- {m['kind']}: {m['canonical']}  via='{m['alias']}'  score={m['score']} span={m['span']}")

        print("\n[bold]=== Forced (Rules) ===[/bold]")
        if out["forced"]:
            for f in out["forced"]:
                print(f"- {f}")
        else:
            print("(none)")

        print("\n[bold]=== Evidence (Final) ===[/bold]")
        for i, h in enumerate(out["final_evidence"], 1):
            print(
                f"{i}. [{h['id']}] {h['title']} "
                f"src={h['source']} vec={h['vec_score']:.4f} rerank={h['rerank_score']:.4f} "
                f"meta(drug={h['metadata'].get('drug')}, pop={h['metadata'].get('population')}, sec={h['metadata'].get('section')})"
            )
            print(f"   {h['text']}")

        print("\n[bold]=== Draft Answer (Template) ===[/bold]")
        print(out["draft_answer"])
        print("-" * 80)


if __name__ == "__main__":
    main()