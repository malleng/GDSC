import typing as t
import json
from pathlib import Path

import pandas as pd


class MetadataLoader:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.labels_path = self.data_dir / "labels.json"
        self.metadata_path = self.data_dir / "metadata.csv"
        self.loaded = False

    def load(self) -> None:
        self.labels = json.loads(self.labels_path.read_bytes())
        self.loaded = True

    def get_classes(self) -> t.List[str]:
        if not self.loaded:
            raise ValueError("Metadata has not yet been loaded")
        return list(self.labels.keys())

    def get_metadata(self) -> t.Tuple[pd.Series, pd.Series, pd.Series]:
        if not self.loaded:
            raise ValueError("Metadata has not yet been loaded")
        df = pd.read_csv(self.metadata_path)
        # Replace "validation" with "val" and keep rest of the values untouched
        df["subset"] = df.subset.map({"validation": "val"}).fillna(df.subset).astype("string")
        df["filename"] = df.apply(
            lambda row: self.data_dir / row.subset / row.file_name, axis=1
        ).astype("string")
        return df.filename, df.label, df.subset

    def get_test_filenames(self) -> pd.Series:
        """Search for test files in the data directory and return their full paths."""
        test_files_iterator = self.data_dir.glob("test/*.wav")
        return pd.Series(test_files_iterator).astype("string")
