# Med Hybrid RAG (CLI)

面向医疗场景的融合检索增强（Hybrid RAG）命令行 demo：从问句理解到多路召回、精排与规则兜底的一体化流水线。

## 核心功能与特点

- **实体与意图识别**：药品/人群实体抽取 + 轻量意图分类（用药安全 / 一般医学 / 非医疗），无额外模型依赖。
- **多路召回**：BM25 + 向量检索，按意图切换策略（安全类偏 BM25、一般类偏向量），RRF 融合。
- **精排**：支持简单打分或 Cross-Encoder 精排（如 `ms-marco-MiniLM-L-6-v2`），可配置 top-k。
- **规则兜底**：规则引擎对禁用/慎用等强约束优先补全证据，再与精排结果融合。
- **可评测**：内置 `eval.jsonl` 评测集与 `scripts.eval`，支持 Recall@k、MRR、NDCG 等指标。

## 安装

```bash
pip install -r requirements.txt
```

项目会从 Hugging Face 下载模型。若网络受限，可设置镜像：

```bash
# PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

## 运行

在项目根目录执行：

```bash
python run.py
```

按提示输入问题（如「孕期高血压能不能用络活喜5mg？」），输入 `quit` 退出。

## 评测

使用 `medrag/data/eval/eval.jsonl` 作为评测集，在根目录运行：

```bash
python -m scripts.eval --mode hybrid --reranker cross_encoder --k 5
```

结果输出到 `medrag/data/eval/run_logs.jsonl`。
