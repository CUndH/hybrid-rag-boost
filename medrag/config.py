from dataclasses import dataclass
import os


@dataclass
class Paths:
    base_dir: str

    @property
    def data_dir(self) -> str:
        return os.path.join(self.base_dir, "data")

    @property
    def aliases_path(self) -> str:
        return os.path.join(self.data_dir, "aliases.json")

    @property
    def documents_path(self) -> str:
        return os.path.join(self.data_dir, "documents.json")

    @property
    def rules_path(self) -> str:
        return os.path.join(self.data_dir, "rules.json")


def default_paths() -> Paths:
    here = os.path.dirname(__file__)
    return Paths(base_dir=here)
