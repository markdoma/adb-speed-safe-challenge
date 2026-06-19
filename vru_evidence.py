from pathlib import Path
import os
import re

import pandas as pd


VRU_REVIEW_COLUMNS = [
    "sidewalk_present",
    "crossing_present",
    "school_or_market_context",
    "transit_stop_context",
    "pedestrian_activity_visible",
    "speed_sign_visible",
    "notes",
]

VRU_SIGNAL_COLUMNS = [
    "sidewalk_present",
    "crossing_present",
    "school_or_market_context",
    "transit_stop_context",
    "pedestrian_activity_visible",
]


VRU_BOOLEAN_COLUMNS = [
    "sidewalk_present",
    "crossing_present",
    "school_or_market_context",
    "transit_stop_context",
    "pedestrian_activity_visible",
    "speed_sign_visible",
]

DEFAULT_VRU_EXPOSURE_WEIGHTS = {
    "is_urban_land_use": 0.35,
    "road_class_vru_weight": 0.20,
    "traffic_component": 0.15,
    "manual_vru_indicator_score": 0.25,
    "has_street_image_coords": 0.03,
    "has_mapillary_evidence": 0.02,
}


def setup_mapillary_client(project_root: Path):
    """Load Mapillary SDK and access token from env or a MAPILLARY_TOKEN file."""
    try:
        import mapillary.interface as mly
    except ModuleNotFoundError:
        print("Mapillary is not installed. Run `%pip install mapillary` or `pip install -r requirements.txt`, then restart the kernel.")
        return None

    token_candidates = [
        project_root / "MAPILLARY_TOKEN",
        Path.cwd() / "MAPILLARY_TOKEN",
        Path.cwd().parent / "MAPILLARY_TOKEN",
    ]
    token_path = next((path for path in token_candidates if path.exists()), None)
    mapillary_token = os.environ.get("MAPILLARY_TOKEN")

    if not mapillary_token and token_path is not None:
        mapillary_token = token_path.read_text(encoding="utf-8").strip()

    if mapillary_token:
        mly.set_access_token(mapillary_token)
        print("Mapillary token loaded")
    else:
        print("Set MAPILLARY_TOKEN as an environment variable or create a MAPILLARY_TOKEN file in the project root.")

    return mly


def parse_street_image_link(value):
    """Parse comma-separated lon/lat pairs from StreetImageLink-like values."""
    if pd.isna(value):
        return []

    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(value))]
    return [{"lon": nums[i], "lat": nums[i + 1]} for i in range(0, len(nums) - 1, 2)]


def add_street_image_query_points(df: pd.DataFrame) -> pd.DataFrame:
    """Add query_lon/query_lat from the first StreetImageLink coordinate pair."""
    out = df.copy()

    if "StreetImageLink" not in out.columns:
        out["query_lon"] = pd.NA
        out["query_lat"] = pd.NA
        out["street_image_point_count"] = 0
        return out

    parsed_points = out["StreetImageLink"].apply(parse_street_image_link)
    out["street_image_point_count"] = parsed_points.apply(len)
    out["query_lon"] = parsed_points.apply(lambda points: points[0]["lon"] if points else pd.NA)
    out["query_lat"] = parsed_points.apply(lambda points: points[0]["lat"] if points else pd.NA)
    return out


def mapillary_features_to_list(response) -> list[dict]:
    """Normalize SDK/API responses into a list of GeoJSON-like features."""
    if response is None:
        return []
    if hasattr(response, "to_dict"):
        response = response.to_dict()
    if isinstance(response, dict):
        if isinstance(response.get("features"), list):
            return response["features"]
        if response.get("type") == "Feature":
            return [response]
    return []


def first_mapillary_image_metadata(mly, lon: float, lat: float, radius_m: int = 100) -> dict:
    """Fetch nearby Mapillary image metadata for one coordinate."""
    if mly is None:
        raise RuntimeError("Mapillary package is not available. Install mapillary and restart the kernel.")

    try:
        response = mly.get_image_close_to(
            longitude=lon,
            latitude=lat,
            radius=radius_m,
        )
    except Exception:
        response = mly.get_image_close_to(
            longitude=lon,
            latitude=lat,
            radius=radius_m,
            image_type="all",
        )

    features = mapillary_features_to_list(response)
    if not features:
        return {
            "mapillary_has_coverage": 0,
            "mapillary_image_count": 0,
        }

    feature = features[0]
    props = feature.get("properties") or {}
    geometry = feature.get("geometry") or {}
    coords = geometry.get("coordinates") or [pd.NA, pd.NA]
    image_id = props.get("id") or feature.get("id")

    thumbnail_url = pd.NA
    if image_id:
        try:
            thumbnail_url = mly.image_thumbnail(image_id=image_id, resolution=1024)
        except Exception:
            thumbnail_url = pd.NA

    return {
        "mapillary_has_coverage": 1,
        "mapillary_image_count": len(features),
        "mapillary_image_id": image_id,
        "mapillary_captured_at": props.get("captured_at"),
        "mapillary_compass_angle": props.get("compass_angle"),
        "mapillary_is_pano": props.get("is_pano"),
        "mapillary_image_lon": coords[0] if len(coords) > 0 else pd.NA,
        "mapillary_image_lat": coords[1] if len(coords) > 1 else pd.NA,
        "mapillary_thumbnail_url": thumbnail_url,
    }


def build_mapillary_vru_evidence_cache(
    df: pd.DataFrame,
    evidence_path: Path,
    mly,
    top_n: int = 50,
    radius_m: int = 100,
) -> pd.DataFrame:
    """Create a cache CSV with Mapillary image metadata plus blank review-label columns."""
    if df.empty:
        return pd.DataFrame()
    if "OBJECTID" not in df.columns:
        raise ValueError("DataFrame must include OBJECTID")

    if "query_lon" not in df.columns or "query_lat" not in df.columns:
        df = add_street_image_query_points(df)

    ranked = df.sort_values("priority_score", ascending=False).head(top_n) if "priority_score" in df.columns else df.head(top_n)
    rows = []

    for _, row in ranked.iterrows():
        query_lon = row.get("query_lon")
        query_lat = row.get("query_lat")
        base = {
            "OBJECTID": row.get("OBJECTID"),
            "english_ro": row.get("english_ro"),
            "RoadClass": row.get("RoadClass"),
            "LandUse": row.get("LandUse"),
            "priority_score": row.get("priority_score"),
            "query_lon": query_lon,
            "query_lat": query_lat,
        }

        if pd.notna(query_lon) and pd.notna(query_lat):
            try:
                base.update(first_mapillary_image_metadata(mly, query_lon, query_lat, radius_m=radius_m))
            except Exception as exc:
                base.update(
                    {
                        "mapillary_has_coverage": 0,
                        "mapillary_image_count": 0,
                        "mapillary_error": str(exc),
                    }
                )
        else:
            base.update(
                {
                    "mapillary_has_coverage": 0,
                    "mapillary_image_count": 0,
                    "mapillary_error": "No StreetImageLink coordinates",
                }
            )

        for col in VRU_REVIEW_COLUMNS:
            base.setdefault(col, pd.NA)
        rows.append(base)

    cache = pd.DataFrame(rows)
    cache.to_csv(evidence_path, index=False)
    return cache


def vru_label_to_number(value):
    """Normalize manual/CV review labels into 1, 0, or missing."""
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "present", "visible"}:
        return 1
    if text in {"0", "false", "f", "no", "n", "absent", "not visible"}:
        return 0
    return pd.NA


def road_class_vru_weight(value) -> float:
    """Approximate VRU-relevance by road class before external POI data is joined."""
    road_class = str(value).lower()
    weights = {
        "secondary": 1.00,
        "primary": 0.85,
        "trunk": 0.60,
        "motorway": 0.10,
    }
    return weights.get(road_class, 0.50)


def normalize_boolean_series(series: pd.Series) -> pd.Series:
    return series.map(vru_label_to_number).astype("Float64")


def percentile_rank(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)
    return numeric.rank(pct=True).fillna(0.0)


def load_vru_evidence(path: Path) -> pd.DataFrame:
    """Load optional Mapillary API cache and reviewed VRU labels."""
    if not path.exists():
        return pd.DataFrame()

    evidence = pd.read_csv(path)
    if "OBJECTID" not in evidence.columns:
        raise ValueError(f"{path} must contain an OBJECTID column")

    evidence = evidence.copy()
    evidence["OBJECTID"] = pd.to_numeric(evidence["OBJECTID"], errors="coerce").astype("Int64")
    for col in VRU_BOOLEAN_COLUMNS:
        if col in evidence.columns:
            evidence[col] = normalize_boolean_series(evidence[col])
    return evidence


def build_segment_vru_check(
    prioritized_df: pd.DataFrame,
    evidence_path: Path,
    signal_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Join prioritized OBJECTIDs/query coordinates to Mapillary review evidence."""
    signal_columns = signal_columns or VRU_SIGNAL_COLUMNS

    if prioritized_df.empty:
        return pd.DataFrame()
    if "OBJECTID" not in prioritized_df.columns:
        raise ValueError("prioritized_df must include OBJECTID")

    segment_cols = [
        c
        for c in [
            "OBJECTID",
            "english_ro",
            "RoadClass",
            "LandUse",
            "priority_score",
            "query_lon",
            "query_lat",
            "street_image_point_count",
        ]
        if c in prioritized_df.columns
    ]
    segments = prioritized_df[segment_cols].copy()
    segments["OBJECTID"] = pd.to_numeric(segments["OBJECTID"], errors="coerce").astype("Int64")

    if not evidence_path.exists():
        segments["mapillary_has_coverage"] = 0
        segments["vru_presence_status"] = "no_cache"
        segments["vru_check_action"] = "Build the Mapillary evidence cache first."
        return segments

    evidence = pd.read_csv(evidence_path)
    if "OBJECTID" not in evidence.columns:
        raise ValueError(f"{evidence_path} must contain OBJECTID")

    evidence = evidence.copy()
    evidence["OBJECTID"] = pd.to_numeric(evidence["OBJECTID"], errors="coerce").astype("Int64")
    for col in signal_columns + ["speed_sign_visible"]:
        if col in evidence.columns:
            evidence[col] = evidence[col].map(vru_label_to_number).astype("Float64")

    evidence_cols = [
        c
        for c in [
            "OBJECTID",
            "mapillary_has_coverage",
            "mapillary_image_count",
            "mapillary_image_id",
            "mapillary_image_lon",
            "mapillary_image_lat",
            "mapillary_thumbnail_url",
            *signal_columns,
            "speed_sign_visible",
            "notes",
        ]
        if c in evidence.columns
    ]
    check = segments.merge(evidence[evidence_cols], on="OBJECTID", how="left")

    available_signal_cols = [c for c in signal_columns if c in check.columns]
    if available_signal_cols:
        check["vru_signal_count"] = check[available_signal_cols].sum(axis=1, skipna=True).astype("Float64")
        check["vru_reviewed"] = check[available_signal_cols].notna().any(axis=1)
        check["vru_detected"] = check["vru_signal_count"].fillna(0).gt(0)
    else:
        check["vru_signal_count"] = 0
        check["vru_reviewed"] = False
        check["vru_detected"] = False

    coverage_source = (
        check["mapillary_has_coverage"]
        if "mapillary_has_coverage" in check.columns
        else pd.Series(0, index=check.index)
    )
    coverage = coverage_source.fillna(0).astype(float).gt(0)

    check["vru_presence_status"] = "unknown_no_mapillary_coverage"
    check.loc[coverage & ~check["vru_reviewed"], "vru_presence_status"] = "needs_manual_or_cv_review"
    check.loc[coverage & check["vru_reviewed"] & ~check["vru_detected"], "vru_presence_status"] = (
        "reviewed_no_vru_visible"
    )
    check.loc[coverage & check["vru_detected"], "vru_presence_status"] = "vru_visible"

    check["vru_check_action"] = check["vru_presence_status"].map(
        {
            "needs_manual_or_cv_review": "Open thumbnail/mapillary_image_id and label VRU evidence columns.",
            "reviewed_no_vru_visible": "Keep as no visible VRU unless better imagery is available.",
            "vru_visible": "Use as positive VRU evidence for this segment.",
            "unknown_no_mapillary_coverage": "Try another image source or field/POI evidence.",
        }
    )
    return check


def add_vru_exposure_features(
    df: pd.DataFrame,
    evidence_path: Path,
    vru_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    vru_weights = vru_weights or DEFAULT_VRU_EXPOSURE_WEIGHTS

    vru_generated_cols = [
        "is_urban_land_use",
        "road_class_vru_weight",
        "has_street_image_coords",
        "has_mapillary_evidence",
        "has_manual_vru_evidence",
        "manual_vru_indicator_score",
        "vru_exposure_score",
        "safety_priority_score",
    ]
    cache_prefixes = ("mapillary_", "query_")
    review_cols = VRU_BOOLEAN_COLUMNS + ["notes"]
    columns_to_drop = [
        col
        for col in out.columns
        if col in vru_generated_cols or col in review_cols or col.startswith(cache_prefixes) or col.endswith("_evidence")
    ]
    if columns_to_drop:
        out = out.drop(columns=columns_to_drop)

    out["is_urban_land_use"] = (
        out["LandUse"].astype(str).str.upper().eq("URBAN").astype(float) if "LandUse" in out.columns else 0.0
    )
    out["road_class_vru_weight"] = out["RoadClass"].map(road_class_vru_weight) if "RoadClass" in out.columns else 0.5
    out["has_street_image_coords"] = out["StreetImageLink"].notna().astype(float) if "StreetImageLink" in out.columns else 0.0

    evidence = load_vru_evidence(evidence_path)
    if not evidence.empty and "OBJECTID" in out.columns:
        duplicate_context_cols = {"english_ro", "RoadClass", "LandUse", "priority_score"}
        evidence_cols = [
            "OBJECTID",
            *[c for c in evidence.columns if c != "OBJECTID" and c not in duplicate_context_cols and c not in out.columns],
        ]
        out = out.merge(evidence[evidence_cols], on="OBJECTID", how="left")
        out["has_mapillary_evidence"] = (
            out.get("mapillary_has_coverage", 0).fillna(0).astype(float) if "mapillary_has_coverage" in out.columns else 0.0
        )
        out["has_manual_vru_evidence"] = 1.0
        evidence_cols_present = [c for c in VRU_BOOLEAN_COLUMNS if c in out.columns]
        if evidence_cols_present:
            out.loc[out[evidence_cols_present].isna().all(axis=1), "has_manual_vru_evidence"] = 0.0
    else:
        out["has_mapillary_evidence"] = 0.0
        out["has_manual_vru_evidence"] = 0.0

    present_evidence_cols = [c for c in VRU_BOOLEAN_COLUMNS if c in out.columns]
    if present_evidence_cols:
        exposure_cols = [c for c in present_evidence_cols if c != "speed_sign_visible"]
        out["manual_vru_indicator_score"] = out[exposure_cols].mean(axis=1, skipna=True).fillna(0.0) if exposure_cols else 0.0
        out["speed_sign_visible"] = out["speed_sign_visible"].fillna(0.0) if "speed_sign_visible" in out.columns else 0.0
    else:
        out["manual_vru_indicator_score"] = 0.0
        out["speed_sign_visible"] = 0.0

    traffic_component = percentile_rank(out["WeightedSample"]) if "WeightedSample" in out.columns else pd.Series(0.0, index=out.index)
    out["vru_exposure_score"] = (
        vru_weights.get("is_urban_land_use", 0) * out["is_urban_land_use"].fillna(0.0)
        + vru_weights.get("road_class_vru_weight", 0) * out["road_class_vru_weight"].fillna(0.0)
        + vru_weights.get("traffic_component", 0) * traffic_component.fillna(0.0)
        + vru_weights.get("manual_vru_indicator_score", 0) * out["manual_vru_indicator_score"].fillna(0.0)
        + vru_weights.get("has_street_image_coords", 0) * out["has_street_image_coords"].fillna(0.0)
        + vru_weights.get("has_mapillary_evidence", 0) * out.get("has_mapillary_evidence", 0).fillna(0.0)
    )

    if "priority_score" in out.columns:
        out["safety_priority_score"] = (0.75 * out["priority_score"].fillna(0.0)) + (
            0.25 * out["vru_exposure_score"].fillna(0.0)
        )

    return out


def vru_status_summary(vru_segment_check: pd.DataFrame) -> pd.DataFrame:
    """Count segments by VRU evidence status."""
    if vru_segment_check.empty or "vru_presence_status" not in vru_segment_check.columns:
        return pd.DataFrame(columns=["status", "segments"])
    return (
        vru_segment_check["vru_presence_status"]
        .value_counts(dropna=False)
        .rename_axis("status")
        .reset_index(name="segments")
    )
