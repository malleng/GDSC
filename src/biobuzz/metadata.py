import typing as t
import json
from pathlib import Path

import pandas as pd


class MetadataLoader:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.labels_path = self.data_dir / "labels.json"
        self.loaded = False

    def load(self) -> None:
        self.labels = json.loads(self.labels_path.read_bytes())
        self.loaded = True

    def get_classes(self) -> t.List[str]:
        if not self.loaded:
            raise ValueError("Metadata has not yet been loaded")
        return list(self.labels.keys())

    def get_metadata(self) -> t.Tuple[pd.Series, ...]:
        if not self.loaded:
            raise ValueError("Metadata has not yet been loaded")
        audio_paths = list(self.data_dir.glob("**/*.wav"))
        df = pd.DataFrame({"filename": audio_paths})
        df["target"] = df.filename.map(lambda f: self.labels[f.name.split("_")[0]])
        df["fold"] = df.filename.map(lambda f: f.parent.name)
        df["filename"] = df.filename.map(lambda f: str(f))
        return df.filename, df.target, df.fold
