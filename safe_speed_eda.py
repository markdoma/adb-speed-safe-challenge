import pandas as pd

from safe_speed_schema import NUMERIC_FIELDS


def column_overview(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "missing": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "unique_values": df.nunique(dropna=True),
        }
    ).sort_values(["missing_pct", "unique_values"], ascending=[False, False])


def value_counts_table(df: pd.DataFrame, column: str, limit: int = 20) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame()
    return df[column].value_counts(dropna=False).head(limit).rename_axis(column).reset_index(name="segments")


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in NUMERIC_FIELDS if c in df.columns]
    if df.empty or not numeric_cols:
        return pd.DataFrame()
    return df[numeric_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T


def duplicate_id_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["OBJECTID", "OvertureID"]:
        if col in df.columns:
            rows.append({"field": col, "duplicate_rows": int(df[col].duplicated(keep=False).sum())})
    return pd.DataFrame(rows)


def speed_field_preview(df: pd.DataFrame, rows: int = 10) -> pd.DataFrame:
    speed_cols = [c for c in ["SpeedLimit", "MedianSpeed", "F85thPercentileSpeed", "PercentOverLimit"] if c in df.columns]
    if df.empty or not speed_cols:
        return pd.DataFrame()
    return df[speed_cols].head(rows)


def geometry_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    geometry_cols = [
        c
        for c in ["OBJECTID", "english_ro", "RoadClass", "LandUse", "Shape_Length", "_geometry_type", "_coordinate_count"]
        if c in df.columns
    ]
    if df.empty or not geometry_cols:
        return pd.DataFrame(), pd.DataFrame()

    summary = df[geometry_cols].describe(include="all")
    longest = (
        df.sort_values("Shape_Length", ascending=False)[geometry_cols].head(10)
        if "Shape_Length" in df.columns
        else pd.DataFrame()
    )
    return summary, longest
