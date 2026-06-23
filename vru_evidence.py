from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
import re

import pandas as pd


VRU_REVIEW_COLUMNS = [
    "sidewalk_present",
    "crossing_present",
    "school_context",
    "market_context",
    "school_or_market_context",
    "transit_stop_context",
    "pedestrian_activity_visible",
    "speed_sign_visible",
    "notes",
]

VRU_SIGNAL_COLUMNS = [
    "sidewalk_present",
    "crossing_present",
    "school_context",
    "market_context",
    "school_or_market_context",
    "transit_stop_context",
    "pedestrian_activity_visible",
]


VRU_BOOLEAN_COLUMNS = [
    "sidewalk_present",
    "crossing_present",
    "school_context",
    "market_context",
    "school_or_market_context",
    "transit_stop_context",
    "pedestrian_activity_visible",
    "speed_sign_visible",
]

MAPILLARY_EVIDENCE_COLUMNS = [
    "mapillary_has_coverage",
    "mapillary_image_count",
    "mapillary_api_value_count",
    "mapillary_api_values",
    *VRU_BOOLEAN_COLUMNS,
]

DEFAULT_VRU_EXPOSURE_WEIGHTS = {
    "is_urban_land_use": 0.20,
    "road_class_vru_weight": 0.12,
    "traffic_component": 0.13,
    "sidewalk_present": 0.08,
    "crossing_present": 0.12,
    "school_context": 0.07,
    "market_context": 0.07,
    "transit_stop_context": 0.06,
    "pedestrian_activity_visible": 0.10,
    "has_street_image_coords": 0.03,
    "has_mapillary_evidence": 0.02,
}

MAPILLARY_VALUE_KEYWORDS = {
    "sidewalk_present": {"sidewalk", "pavement", "curb", "kerb"},
    "crossing_present": {"crosswalk", "crossing", "zebra", "pedestrian-crossing", "pedestrians-crossing"},
    "school_context": {"school", "children", "student"},
    "market_context": {"market", "shop", "store", "vendor", "stall", "commercial"},
    "transit_stop_context": {"bus-stop", "bus stop", "bus", "public-transport", "train", "station", "transit"},
    "pedestrian_activity_visible": {
        "human--person",
        "person",
        "pedestrian",
        "bicyclist",
        "motorcyclist",
        "rider",
        "bicycle",
        "motorcycle",
    },
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


def bbox_around_point(lon: float, lat: float, radius_m: float) -> dict:
    """Create a small WGS84 bbox around a lon/lat point."""
    lat_delta = radius_m / 111_320
    lon_delta = radius_m / (111_320 * max(0.1, abs(math.cos(math.radians(lat)))))
    return {
        "west": lon - lon_delta,
        "south": lat - lat_delta,
        "east": lon + lon_delta,
        "north": lat + lat_delta,
    }


def records_from_mapillary_response(response) -> list[dict]:
    """Normalize Mapillary API/SDK responses into a list of records."""
    if response is None:
        return []
    if hasattr(response, "to_dict"):
        response = response.to_dict()
    if isinstance(response, str):
        try:
            import json

            response = json.loads(response)
        except Exception:
            return []
    if isinstance(response, dict):
        if isinstance(response.get("data"), list):
            return response["data"]
        if isinstance(response.get("features"), list):
            records = []
            for feature in response["features"]:
                props = dict(feature.get("properties") or {})
                props.setdefault("id", feature.get("id"))
                props.setdefault("geometry", feature.get("geometry"))
                records.append(props)
            return records
        return [response]
    if isinstance(response, list):
        return response
    return []


def classify_mapillary_values(values: list[str], keywords: dict[str, set[str]] = MAPILLARY_VALUE_KEYWORDS) -> dict:
    """Classify Mapillary detection/map-feature values into VRU evidence flags."""
    normalized_values = [str(value).lower() for value in values if pd.notna(value)]
    joined = " | ".join(normalized_values)
    out = {}
    for field, field_keywords in keywords.items():
        out[field] = int(any(keyword.lower() in joined for keyword in field_keywords))
    out["mapillary_api_values"] = ";".join(sorted(set(normalized_values)))
    out["mapillary_api_value_count"] = len(normalized_values)
    return out


def mapillary_values_from_records(records: list[dict]) -> list[str]:
    values = []
    for record in records:
        if isinstance(record, dict):
            value = record.get("value") or record.get("object_value") or record.get("class") or record.get("type")
            if value is not None:
                values.append(str(value))
    return values


def mapillary_feature_context(mly, lon: float, lat: float, radius_m: float = 100) -> dict:
    """Use Mapillary map-feature and traffic-sign API values as VRU context evidence."""
    if mly is None or pd.isna(lon) or pd.isna(lat):
        return {}

    bbox = bbox_around_point(float(lon), float(lat), radius_m)
    values = []
    errors = []
    calls = [
        ("map_points", getattr(mly, "map_feature_points_in_bbox", None)),
        ("traffic_signs", getattr(mly, "traffic_signs_in_bbox", None)),
    ]
    for label, func in calls:
        if func is None:
            continue
        try:
            values.extend(mapillary_values_from_records(records_from_mapillary_response(func(bbox=bbox))))
        except Exception as exc:
            errors.append(f"{label}:{exc}")

    context = classify_mapillary_values(values)
    context["mapillary_feature_error"] = "; ".join(errors) if errors else pd.NA
    return context


def mapillary_detection_context_for_image(mly, image_id) -> dict:
    """Use Mapillary image detection API values as VRU context evidence."""
    if mly is None or pd.isna(image_id):
        return {}
    try:
        normalized_id = int(float(str(image_id)))
        records = records_from_mapillary_response(mly.get_detections_with_image_id(image_id=normalized_id, fields=["value"]))
        return classify_mapillary_values(mapillary_values_from_records(records))
    except Exception as exc:
        return {"mapillary_detection_error": str(exc)}


def merge_mapillary_contexts(*contexts: dict) -> dict:
    """Merge multiple Mapillary-derived context dictionaries into one evidence row."""
    merged = {}
    value_sets = []
    errors = []
    for context in contexts:
        for key, value in (context or {}).items():
            if key in VRU_BOOLEAN_COLUMNS:
                merged[key] = max(float(merged.get(key, 0) or 0), float(value or 0))
            elif key == "mapillary_api_values" and pd.notna(value):
                value_sets.extend(str(value).split(";"))
            elif key.endswith("_error") and pd.notna(value):
                errors.append(f"{key}:{value}")
            elif key == "mapillary_api_value_count":
                merged[key] = (merged.get(key, 0) or 0) + (value or 0)
            else:
                merged[key] = value
    if value_sets:
        merged["mapillary_api_values"] = ";".join(sorted({v.strip() for v in value_sets if v.strip()}))
    if errors:
        merged["mapillary_api_errors"] = "; ".join(errors)
    return merged


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


def mapillary_image_metadata_rows(
    mly,
    lon: float,
    lat: float,
    radius_m: int = 100,
    max_images: int | None = None,
    include_thumbnails: bool = False,
) -> list[dict]:
    """Fetch nearby Mapillary image metadata rows for one coordinate."""
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
        return [{"mapillary_has_coverage": 0, "mapillary_image_count": 0}]

    selected_features = features[:max_images] if max_images else features
    rows = []
    for image_rank, feature in enumerate(selected_features, start=1):
        props = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        coords = geometry.get("coordinates") or [pd.NA, pd.NA]
        image_id = props.get("id") or feature.get("id")

        thumbnail_url = pd.NA
        if include_thumbnails and image_id:
            try:
                thumbnail_url = mly.image_thumbnail(image_id=image_id, resolution=1024)
            except Exception:
                thumbnail_url = pd.NA

        rows.append(
            {
                "mapillary_has_coverage": 1,
                "mapillary_image_count": len(features),
                "mapillary_images_cached": len(selected_features),
                "mapillary_image_rank": image_rank,
                "mapillary_image_id": image_id,
                "mapillary_captured_at": props.get("captured_at"),
                "mapillary_compass_angle": props.get("compass_angle"),
                "mapillary_is_pano": props.get("is_pano"),
                "mapillary_image_lon": coords[0] if len(coords) > 0 else pd.NA,
                "mapillary_image_lat": coords[1] if len(coords) > 1 else pd.NA,
                "mapillary_thumbnail_url": thumbnail_url,
            }
        )
    return rows


def first_mapillary_image_metadata(mly, lon: float, lat: float, radius_m: int = 100) -> dict:
    """Fetch the first nearby Mapillary image metadata row for backward compatibility."""
    return mapillary_image_metadata_rows(mly, lon, lat, radius_m=radius_m, max_images=1)[0]


def build_mapillary_vru_evidence_cache(
    df: pd.DataFrame,
    evidence_path: Path,
    mly,
    top_n: int = 50,
    radius_m: int = 100,
    max_images_per_segment: int | None = None,
    include_thumbnails: bool = False,
    use_mapillary_api_filters: bool = True,
    use_image_detections: bool = False,
    max_detection_images_per_segment: int | None = 1,
    overwrite: bool = False,
    batch_size: int | None = None,
    flush_every: int = 25,
    progress_every: int = 25,
    skip_missing_coords: bool = True,
    max_workers: int = 1,
) -> pd.DataFrame:
    """Create a per-image cache CSV with Mapillary metadata plus VRU signal columns."""
    if df.empty:
        return pd.DataFrame()
    if "OBJECTID" not in df.columns:
        raise ValueError("DataFrame must include OBJECTID")

    if "query_lon" not in df.columns or "query_lat" not in df.columns:
        df = add_street_image_query_points(df)

    ranked = df.sort_values("priority_score", ascending=False).head(top_n) if "priority_score" in df.columns else df.head(top_n)
    if skip_missing_coords:
        ranked = ranked[ranked["query_lon"].notna() & ranked["query_lat"].notna()]

    existing_cache = pd.DataFrame()
    processed_ids = set()
    if evidence_path.exists() and not overwrite:
        existing_cache = pd.read_csv(evidence_path)
        if skip_missing_coords and {"query_lon", "query_lat"}.issubset(existing_cache.columns):
            existing_cache = existing_cache[existing_cache["query_lon"].notna() & existing_cache["query_lat"].notna()]
        required_cache_cols = [c for c in MAPILLARY_EVIDENCE_COLUMNS if c not in VRU_REVIEW_COLUMNS]
        if all(c in existing_cache.columns for c in required_cache_cols):
            complete_mask = existing_cache["mapillary_has_coverage"].notna()
            if "mapillary_api_value_count" in existing_cache.columns:
                complete_mask = complete_mask & existing_cache["mapillary_api_value_count"].notna()
            existing_cache = existing_cache[complete_mask]
        else:
            existing_cache = pd.DataFrame()
        if "OBJECTID" in existing_cache.columns:
            processed_ids = set(existing_cache["OBJECTID"].dropna().astype(str))

    if processed_ids:
        ranked = ranked[~ranked["OBJECTID"].astype(str).isin(processed_ids)]
    if batch_size is not None:
        ranked = ranked.head(batch_size)

    total_segments = len(ranked)
    if total_segments == 0:
        if progress_every:
            print("Mapillary evidence progress: no new coordinate-eligible segment(s) to process")
        return existing_cache

    def process_segment(row) -> list[dict]:
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
                image_rows = mapillary_image_metadata_rows(
                    mly,
                    query_lon,
                    query_lat,
                    radius_m=radius_m,
                    max_images=max_images_per_segment,
                    include_thumbnails=include_thumbnails,
                )
            except Exception as exc:
                image_rows = [{"mapillary_has_coverage": 0, "mapillary_image_count": 0, "mapillary_error": str(exc)}]
        else:
            image_rows = [
                {
                    "mapillary_has_coverage": 0,
                    "mapillary_image_count": 0,
                    "mapillary_error": "No StreetImageLink coordinates",
                }
            ]

        feature_context = (
            mapillary_feature_context(mly, query_lon, query_lat, radius_m=radius_m)
            if use_mapillary_api_filters and pd.notna(query_lon) and pd.notna(query_lat)
            else {}
        )
        segment_rows = []
        for image_row in image_rows:
            image_rank = image_row.get("mapillary_image_rank")
            within_detection_limit = (
                max_detection_images_per_segment is None
                or pd.isna(image_rank)
                or int(image_rank) <= max_detection_images_per_segment
            )
            detection_context = (
                mapillary_detection_context_for_image(mly, image_row.get("mapillary_image_id"))
                if use_image_detections
                and image_row.get("mapillary_has_coverage")
                and within_detection_limit
                else {}
            )
            context = merge_mapillary_contexts(feature_context, detection_context)
            evidence_row = {**base, **image_row, **context}
            for col in VRU_REVIEW_COLUMNS:
                evidence_row.setdefault(col, pd.NA)
            segment_rows.append(evidence_row)
        return segment_rows

    rows = []

    def flush_rows() -> None:
        nonlocal existing_cache, rows
        if not rows:
            return
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        partial = pd.DataFrame(rows)
        cache = pd.concat([existing_cache, partial], ignore_index=True) if not existing_cache.empty else partial
        cache.to_csv(evidence_path, index=False)
        existing_cache = cache
        rows = []

    max_workers = max(1, int(max_workers or 1))
    if max_workers == 1:
        for processed_count, (_, row) in enumerate(ranked.iterrows(), start=1):
            rows.extend(process_segment(row))

            if progress_every and (processed_count == 1 or processed_count % progress_every == 0 or processed_count == total_segments):
                print(f"Mapillary evidence progress: {processed_count}/{total_segments} new segment(s)")

            if flush_every and rows and (processed_count % flush_every == 0 or processed_count == total_segments):
                flush_rows()
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_segment, row): row for _, row in ranked.iterrows()}
            for processed_count, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                try:
                    rows.extend(future.result())
                except Exception as exc:
                    rows.append(
                        {
                            "OBJECTID": row.get("OBJECTID"),
                            "english_ro": row.get("english_ro"),
                            "RoadClass": row.get("RoadClass"),
                            "LandUse": row.get("LandUse"),
                            "priority_score": row.get("priority_score"),
                            "query_lon": row.get("query_lon"),
                            "query_lat": row.get("query_lat"),
                            "mapillary_has_coverage": 0,
                            "mapillary_image_count": 0,
                            "mapillary_error": str(exc),
                        }
                    )

                if progress_every and (
                    processed_count == 1 or processed_count % progress_every == 0 or processed_count == total_segments
                ):
                    print(f"Mapillary evidence progress: {processed_count}/{total_segments} new segment(s)")

                if flush_every and rows and (processed_count % flush_every == 0 or processed_count == total_segments):
                    flush_rows()

    cache = existing_cache if not rows else pd.concat([existing_cache, pd.DataFrame(rows)], ignore_index=True)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_csv(evidence_path, index=False)
    return cache


def vru_label_to_number(value):
    """Normalize Mapillary/manual review labels into 1, 0, or missing."""
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


def zero_series(index) -> pd.Series:
    return pd.Series(0.0, index=index)


def evidence_factor(out: pd.DataFrame, column: str, fallback_column: str | None = None) -> pd.Series:
    """Return a numeric evidence factor, optionally falling back to a combined column."""
    primary = out[column] if column in out.columns else pd.Series(pd.NA, index=out.index)
    if fallback_column and fallback_column in out.columns:
        primary = primary.fillna(out[fallback_column])
    return pd.to_numeric(primary, errors="coerce").fillna(0.0)


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
    return aggregate_vru_evidence_by_segment(evidence)


def aggregate_vru_evidence_by_segment(evidence: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-image VRU evidence rows to one row per OBJECTID."""
    if evidence.empty:
        return pd.DataFrame()
    if "OBJECTID" not in evidence.columns:
        raise ValueError("Evidence must contain an OBJECTID column")

    out = evidence.copy()
    out["OBJECTID"] = pd.to_numeric(out["OBJECTID"], errors="coerce").astype("Int64")
    rows = []
    for object_id, group in out.groupby("OBJECTID", dropna=False):
        row = {"OBJECTID": object_id}

        for col in ["english_ro", "RoadClass", "LandUse", "priority_score", "query_lon", "query_lat"]:
            if col in group.columns:
                values = group[col].dropna()
                row[col] = values.iloc[0] if not values.empty else pd.NA

        if "mapillary_has_coverage" in group.columns:
            row["mapillary_has_coverage"] = pd.to_numeric(group["mapillary_has_coverage"], errors="coerce").fillna(0).max()
        if "mapillary_image_count" in group.columns:
            row["mapillary_image_count"] = pd.to_numeric(group["mapillary_image_count"], errors="coerce").fillna(0).max()
        row["mapillary_images_cached"] = int(group["mapillary_image_id"].notna().sum()) if "mapillary_image_id" in group.columns else len(group)

        for col in VRU_BOOLEAN_COLUMNS:
            if col in group.columns:
                row[col] = pd.to_numeric(group[col], errors="coerce").max(skipna=True)

        for col in ["mapillary_thumbnail_url", "notes", "mapillary_api_errors"]:
            if col in group.columns:
                values = group[col].dropna()
                row[col] = values.iloc[0] if not values.empty else pd.NA

        if "mapillary_image_id" in group.columns:
            ids = [str(value) for value in group["mapillary_image_id"].dropna().tolist()]
            row["mapillary_image_ids"] = ";".join(ids)
            row["mapillary_image_id"] = ids[0] if ids else pd.NA

        if "mapillary_api_values" in group.columns:
            labels = set()
            for value in group["mapillary_api_values"].dropna():
                labels.update(label.strip() for label in str(value).split(";") if label.strip())
            row["mapillary_api_values"] = ";".join(sorted(labels))
        if "mapillary_api_value_count" in group.columns:
            row["mapillary_api_value_count"] = pd.to_numeric(group["mapillary_api_value_count"], errors="coerce").fillna(0).sum()

        rows.append(row)

    return pd.DataFrame(rows)


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
    evidence = aggregate_vru_evidence_by_segment(evidence)

    evidence_cols = [
        c
        for c in [
            "OBJECTID",
            "mapillary_has_coverage",
            "mapillary_image_count",
            "mapillary_images_cached",
            "mapillary_image_id",
            "mapillary_image_ids",
            "mapillary_image_lon",
            "mapillary_image_lat",
            "mapillary_thumbnail_url",
            *signal_columns,
            "speed_sign_visible",
            "mapillary_api_values",
            "mapillary_api_value_count",
            "mapillary_api_errors",
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
    check.loc[coverage & ~check["vru_reviewed"], "vru_presence_status"] = "needs_mapillary_or_manual_review"
    check.loc[coverage & check["vru_reviewed"] & ~check["vru_detected"], "vru_presence_status"] = (
        "reviewed_no_vru_visible"
    )
    check.loc[coverage & check["vru_detected"], "vru_presence_status"] = "vru_visible"

    check["vru_check_action"] = check["vru_presence_status"].map(
        {
            "needs_mapillary_or_manual_review": "Review Mapillary API values or label VRU evidence columns manually.",
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
        "traffic_component",
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

    traffic_component = percentile_rank(out["WeightedSample"]) if "WeightedSample" in out.columns else zero_series(out.index)
    out["traffic_component"] = traffic_component

    factor_values = {
        "is_urban_land_use": out["is_urban_land_use"].fillna(0.0),
        "road_class_vru_weight": out["road_class_vru_weight"].fillna(0.0),
        "traffic_component": traffic_component.fillna(0.0),
        "sidewalk_present": evidence_factor(out, "sidewalk_present"),
        "crossing_present": evidence_factor(out, "crossing_present"),
        "school_context": evidence_factor(out, "school_context", fallback_column="school_or_market_context"),
        "market_context": evidence_factor(out, "market_context", fallback_column="school_or_market_context"),
        "school_or_market_context": evidence_factor(out, "school_or_market_context"),
        "transit_stop_context": evidence_factor(out, "transit_stop_context"),
        "pedestrian_activity_visible": evidence_factor(out, "pedestrian_activity_visible"),
        "has_street_image_coords": out["has_street_image_coords"].fillna(0.0),
        "has_mapillary_evidence": out.get("has_mapillary_evidence", zero_series(out.index)).fillna(0.0),
    }

    for factor_name, values in factor_values.items():
        out[f"{factor_name}_factor"] = values

    out["vru_exposure_score"] = zero_series(out.index)
    for factor_name, weight in vru_weights.items():
        if weight and factor_name in factor_values:
            out["vru_exposure_score"] = out["vru_exposure_score"] + (weight * factor_values[factor_name].fillna(0.0))

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
