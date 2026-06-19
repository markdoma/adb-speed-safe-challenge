from pathlib import Path

import pandas as pd

from safe_speed_data import coerce_numeric_columns
from safe_speed_schema import NUMERIC_FIELDS


DEFAULT_PRIORITY_WEIGHTS = {
    "weighted_over_limit_pressure": 0.45,
    "speed_gap_85th": 0.30,
    "PercentOverLimit": 0.15,
    "WeightedSample": 0.10,
}


def percentile_rank(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.rank(pct=True, na_option="keep")


def add_speed_risk_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = coerce_numeric_columns(df, NUMERIC_FIELDS)

    out["speed_gap_85th"] = (
        out["F85thPercentileSpeed"] - out["SpeedLimit"]
        if {"F85thPercentileSpeed", "SpeedLimit"}.issubset(out.columns)
        else pd.NA
    )
    out["speed_gap_median"] = (
        out["MedianSpeed"] - out["SpeedLimit"] if {"MedianSpeed", "SpeedLimit"}.issubset(out.columns) else pd.NA
    )
    out["weighted_over_limit_pressure"] = (
        out["PercentOverLimit"] * out["WeightedSample"]
        if {"PercentOverLimit", "WeightedSample"}.issubset(out.columns)
        else pd.NA
    )

    if {"NumberOverLimit", "Shape_Length"}.issubset(out.columns):
        km = out["Shape_Length"] / 1000
        out["over_limit_per_km"] = out["NumberOverLimit"] / km.replace(0, pd.NA)
    else:
        out["over_limit_per_km"] = pd.NA

    return out


def summarize_speed_risk(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [col for col in ["RoadClass", "LandUse"] if col in df.columns]
    if df.empty or not group_cols:
        return pd.DataFrame()

    value_cols = {
        "OBJECTID": "count",
        "Shape_Length": "sum",
        "WeightedSample": "sum",
        "PercentOverLimit": "mean",
        "speed_gap_85th": "median",
        "weighted_over_limit_pressure": "sum",
        "over_limit_per_km": "median",
    }
    value_cols = {k: v for k, v in value_cols.items() if k in df.columns}

    summary = df.groupby(group_cols, dropna=False).agg(value_cols).reset_index()
    summary = summary.rename(
        columns={
            "OBJECTID": "segment_count",
            "Shape_Length": "total_shape_length_m",
            "WeightedSample": "total_weighted_sample",
            "PercentOverLimit": "avg_percent_over_limit",
            "speed_gap_85th": "median_85th_speed_gap",
            "weighted_over_limit_pressure": "total_weighted_over_limit_pressure",
            "over_limit_per_km": "median_over_limit_per_km",
        }
    )
    sort_cols = [c for c in ["total_weighted_over_limit_pressure", "median_85th_speed_gap"] if c in summary.columns]
    return summary.sort_values(sort_cols, ascending=False) if sort_cols else summary


def add_priority_score(df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    out = df.copy()
    weights = weights or DEFAULT_PRIORITY_WEIGHTS
    components = []

    rank_columns = {
        "weighted_over_limit_pressure": "pressure_rank",
        "speed_gap_85th": "speed_gap_rank",
        "PercentOverLimit": "percent_over_rank",
        "WeightedSample": "traffic_rank",
    }
    for source_col, rank_col in rank_columns.items():
        weight = weights.get(source_col, 0)
        if source_col in out.columns and weight:
            components.append((rank_col, percentile_rank(out[source_col]), weight))

    score = pd.Series(0.0, index=out.index)
    total_weight = 0.0
    for name, values, weight in components:
        out[name] = values
        score = score + values.fillna(0) * weight
        total_weight += weight

    out["priority_score"] = score / total_weight if total_weight else pd.NA
    return out


def export_priority_segments(prioritized: pd.DataFrame, output_dir: Path) -> Path | None:
    if prioritized.empty:
        return None

    export_cols = [
        c
        for c in [
            "priority_score",
            "OBJECTID",
            "OvertureID",
            "english_ro",
            "RoadClass",
            "LandUse",
            "SpeedLimit",
            "MedianSpeed",
            "F85thPercentileSpeed",
            "speed_gap_median",
            "speed_gap_85th",
            "PercentOverLimit",
            "NumberOverLimit",
            "WeightedSample",
            "weighted_over_limit_pressure",
            "over_limit_per_km",
            "Percentile",
            "PercentileBand",
            "SampleSizeTotal",
            "Shape_Length",
            "StreetImageLink",
            "vru_exposure_score",
            "safety_priority_score",
            "is_urban_land_use",
            "road_class_vru_weight",
            "manual_vru_indicator_score",
            "has_manual_vru_evidence",
            "has_mapillary_evidence",
            "mapillary_image_count",
            "mapillary_image_id",
            "mapillary_thumbnail_url",
            "has_street_image_coords",
        ]
        if c in prioritized.columns
    ]

    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "safe_speed_priority_segments.csv"
    prioritized.sort_values("priority_score", ascending=False)[export_cols].to_csv(out_path, index=False)
    return out_path
