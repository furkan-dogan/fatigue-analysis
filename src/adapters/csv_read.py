"""Legacy optional CSV loading; missing exports render as empty tables."""

import pandas as pd


def read_optional_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()
