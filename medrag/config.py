from dataclasses import dataclass
import os


@dataclass
class Paths:
    base_dir: str  # === medrag/ ===

    @property
    def data_dir(self) -> str:
        # medrag/data
        return os.path.join(self.base_dir, "data")

    @property
    def processed_dir(self) -> str:
        # medrag/data/processed
        return os.path.join(self.data_dir, "processed")

    @property
    def aliases_path(self) -> str:
        return os.path.join(self.data_dir, "aliases.json")

    # ✅ 关键：指向 chunk-level 数据
    @property
    def documents_path(self) -> str:
        return os.path.join(self.processed_dir, "chunks.jsonl")

    # ✅ 可选（现在不用，但以后一定会用）
    @property
    def docs_path(self) -> str:
        return os.path.join(self.processed_dir, "docs.jsonl")

    @property
    def rules_path(self) -> str:
        return os.path.join(self.data_dir, "rules.json")


def default_paths() -> Paths:
    # here == medrag/
    here = os.path.dirname(__file__)
    return Paths(base_dir=here)
