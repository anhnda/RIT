import numpy as np
from pandas import DataFrame


class Outliner:
    """IQR-based outlier winsorizer.

    Replaces values outside [Q1 - 1.5·IQR, Q3 + 1.5·IQR] with NaN.
    Fit on training data; apply the same bounds to validation/test data.
    """

    def __init__(self):
        self.fitted = False

    def fit(self, df: DataFrame):
        self.Q1  = df.quantile(0.25)
        self.Q3  = df.quantile(0.75)
        self.IQR = self.Q3 - self.Q1
        self.fitted = True

    def transform(self, df: DataFrame) -> DataFrame:
        df = df.copy()
        lower = self.Q1 - 1.5 * self.IQR
        upper = self.Q3 + 1.5 * self.IQR
        df[(df < lower) | (df > upper)] = np.nan
        return df

    def fit_transform(self, df: DataFrame) -> DataFrame:
        self.fit(df)
        return self.transform(df)
