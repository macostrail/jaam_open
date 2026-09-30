import pandas as pd

from .config import VARIABLES_DIR


class Variables:
    def __init__(self, version: str = "jaam_2021"):
        self.version = version
        self.phase1_vars = self._read_variable_list("phase1_vars.csv")
        self.phase2_vars = self._read_variable_list("phase2_vars.csv")
        self.phase3_vars = self._read_variable_list("phase3_vars.csv")
        self.numeric_or_categoric_df = pd.read_csv(
            VARIABLES_DIR / "numeric_categoric_2021.csv",
            encoding="utf-8",
        )

    def _read_variable_list(self, filename: str) -> list[str]:
        df = pd.read_csv(VARIABLES_DIR / filename, encoding="utf-8")
        if self.version not in df.columns:
            return []
        return df[self.version].dropna().astype(str).to_list()
