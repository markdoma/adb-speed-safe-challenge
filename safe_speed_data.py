import json
from pathlib import Path

import pandas as pd

from safe_speed_schema import KEY_FIELDS, NUMERIC_FIELDS, SCHEMA


def load_geojson_features(path: Path) -> list[dict]:
    """Load features from a GeoJSON FeatureCollection or a single Feature."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("type") == "FeatureCollection":
        return data.get("features", [])
    if data.get("type") == "Feature":
        return [data]
    raise ValueError(f"Unsupported GeoJSON root type: {data.get('type')}")


def count_coordinates(coords) -> int:
    if coords is None:
        return 0
    if isinstance(coords, (list, tuple)) and coords and isinstance(coords[0], (int, float)):
        return 1
    if isinstance(coords, (list, tuple)):
        return sum(count_coordinates(item) for item in coords)
    return 0


def feature_properties_frame(features: list[dict]) -> pd.DataFrame:
    rows = []
    for feature in features:
        props = dict(feature.get("properties") or {})
        geom = feature.get("geometry") or {}
        props["_geometry_type"] = geom.get("type")
        props["_coordinate_count"] = count_coordinates(geom.get("coordinates"))
        rows.append(props)
    return pd.DataFrame(rows)


def coerce_numeric_columns(df: pd.DataFrame, columns: list[str] = NUMERIC_FIELDS) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_roads(path: Path) -> tuple[list[dict], pd.DataFrame]:
    if not path.exists():
        return [], pd.DataFrame()

    features = load_geojson_features(path)
    roads_raw = feature_properties_frame(features)
    roads_raw = coerce_numeric_columns(roads_raw)
    return features, roads_raw


def filter_valid_analysis_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if "AnalysisStatus" not in df.columns:
        return df.copy()
    return df[df["AnalysisStatus"].eq("Valid")].copy()


def validate_schema(df: pd.DataFrame, expected_fields: list[str] = KEY_FIELDS) -> pd.DataFrame:
    rows = []
    for field in expected_fields:
        rows.append(
            {
                "field": field,
                "present": field in df.columns,
                "missing_values": int(df[field].isna().sum()) if field in df.columns else None,
                "non_null_values": int(df[field].notna().sum()) if field in df.columns else 0,
                "description": SCHEMA.get(field, ""),
            }
        )
    return pd.DataFrame(rows)
