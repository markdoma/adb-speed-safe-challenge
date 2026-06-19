from pathlib import Path
import os
import re
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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

CV_MODEL_LABEL_MAP = {
    "sidewalk_present": {"sidewalk", "pavement"},
    "crossing_present": {"crosswalk", "zebra crossing", "pedestrian crossing"},
    "school_context": {"school", "school zone", "student", "children"},
    "market_context": {"market", "shop", "store", "vendor", "stall"},
    "transit_stop_context": {"bus stop", "bus", "train station", "station"},
    "pedestrian_activity_visible": {"person", "pedestrian", "bicycle", "bicyclist", "motorcycle", "motorcyclist"},
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


def load_cv_model(model_name: str = "yolov8n.pt"):
    """Load an optional Ultralytics YOLO model for image-based VRU context detection."""
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install `ultralytics` to run CV detection, for example `%pip install ultralytics`."
        ) from exc
    return YOLO(model_name)


def detect_vru_context_with_cv(
    image_path: str | Path,
    model=None,
    model_name: str = "yolov8n.pt",
    confidence_threshold: float = 0.25,
    label_map: dict[str, set[str]] | None = None,
) -> dict:
    """Detect VRU context labels from a local street image using an optional CV model.

    The default YOLO model is useful for visible road users such as people,
    bicycles, motorcycles, buses, and traffic-adjacent objects. Specialized
    models are recommended for sidewalks, crossings, schools, and markets.
    """
    label_map = label_map or CV_MODEL_LABEL_MAP
    model = model or load_cv_model(model_name)
    results = model(str(image_path))

    detections = []
    detected_labels = set()
    for result in results:
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            confidence = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
            if confidence < confidence_threshold:
                continue
            class_id = int(box.cls[0]) if getattr(box, "cls", None) is not None else None
            label = str(names.get(class_id, class_id)).lower()
            detected_labels.add(label)
            detections.append({"label": label, "confidence": confidence})

    context = {field: 0 for field in label_map}
    for field, labels in label_map.items():
        context[field] = int(bool(detected_labels.intersection({label.lower() for label in labels})))
    context["cv_detected_labels"] = ";".join(sorted(detected_labels))
    context["cv_detection_count"] = len(detections)
    context["cv_detections"] = detections
    return context


def apply_cv_context_to_evidence(
    evidence: pd.DataFrame,
    image_path_col: str = "local_image_path",
    model=None,
    model_name: str = "yolov8n.pt",
    confidence_threshold: float = 0.25,
) -> pd.DataFrame:
    """Populate VRU evidence columns from local images when optional CV packages are installed."""
    if image_path_col not in evidence.columns:
        raise ValueError(f"Evidence must include `{image_path_col}` with local image paths.")

    out = evidence.copy()
    model = model or load_cv_model(model_name)
    for idx, row in out.iterrows():
        image_path = row.get(image_path_col)
        if pd.isna(image_path) or not Path(image_path).exists():
            continue
        context = detect_vru_context_with_cv(
            image_path,
            model=model,
            confidence_threshold=confidence_threshold,
        )
        for col, value in context.items():
            if col == "cv_detections":
                continue
            if col in CV_MODEL_LABEL_MAP and col in out.columns and pd.notna(out.loc[idx, col]):
                if value:
                    out.loc[idx, col] = value
                continue
            out.loc[idx, col] = value
    return out


def validate_download_url(url: str) -> tuple[bool, str]:
    """Return whether a URL is usable for thumbnail download plus a reason."""
    parsed = urlparse(str(url).strip())
    if parsed.scheme not in {"http", "https"}:
        return False, "invalid_url_scheme"
    if not parsed.netloc:
        return False, "missing_url_host"
    return True, parsed.netloc


def host_resolves(host: str) -> bool:
    try:
        socket.getaddrinfo(host, None)
    except OSError:
        return False
    return True


def download_file(url: str, output_path: Path, timeout_seconds: int = 30, retries: int = 1) -> Path:
    """Download one URL to a local file using the Python standard library."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "safe-speed-vru-evidence/1.0"})
    last_error = None
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                output_path.write_bytes(response.read())
            return output_path
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1)
    raise last_error


def download_evidence_thumbnails(
    evidence: pd.DataFrame,
    image_dir: Path,
    url_col: str = "mapillary_thumbnail_url",
    id_col: str = "mapillary_image_id",
    output_col: str = "local_image_path",
    limit: int | None = None,
    retry_failed_downloads: bool = False,
    timeout_seconds: int = 30,
    retries: int = 1,
) -> pd.DataFrame:
    """Download Mapillary thumbnail URLs in an evidence table and add local image paths."""
    out = evidence.copy()
    if url_col not in out.columns:
        raise ValueError(f"Evidence must include `{url_col}`.")

    image_dir.mkdir(parents=True, exist_ok=True)
    rows = out.head(limit).iterrows() if limit else out.iterrows()
    unavailable_hosts = set()
    attempted = downloaded = skipped = failed = 0
    for idx, row in rows:
        existing_path = row.get(output_col) if output_col in out.columns else None
        if pd.notna(existing_path) and str(existing_path).strip() and Path(str(existing_path)).exists():
            skipped += 1
            continue

        existing_error = row.get("thumbnail_download_error") if "thumbnail_download_error" in out.columns else None
        if not retry_failed_downloads and pd.notna(existing_error) and str(existing_error).strip():
            skipped += 1
            continue

        url = row.get(url_col)
        if pd.isna(url) or not str(url).strip():
            skipped += 1
            continue
        attempted += 1

        valid_url, host_or_reason = validate_download_url(str(url))
        if not valid_url:
            out.loc[idx, "thumbnail_download_error"] = host_or_reason
            failed += 1
            continue

        host = host_or_reason
        if host in unavailable_hosts:
            out.loc[idx, "thumbnail_download_error"] = f"host_unavailable:{host}"
            skipped += 1
            continue
        if not host_resolves(host):
            unavailable_hosts.add(host)
            out.loc[idx, "thumbnail_download_error"] = f"dns_unavailable:{host}"
            failed += 1
            continue

        image_id = row.get(id_col) if id_col in out.columns else idx
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(image_id or idx)).strip("_")
        image_path = image_dir / f"{safe_id or idx}.jpg"

        if not image_path.exists():
            try:
                download_file(str(url), image_path, timeout_seconds=timeout_seconds, retries=retries)
            except Exception as exc:
                if isinstance(exc, URLError) and "nodename nor servname" in str(exc):
                    unavailable_hosts.add(host)
                out.loc[idx, "thumbnail_download_error"] = str(exc)
                failed += 1
                continue

        out.loc[idx, output_col] = str(image_path)
        out.loc[idx, "thumbnail_download_error"] = pd.NA
        downloaded += 1

    out.attrs["thumbnail_download_summary"] = {
        "attempted": attempted,
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "unavailable_hosts": sorted(unavailable_hosts),
    }
    return out


def run_yolo_on_evidence_cache(
    evidence_path: Path,
    image_dir: Path,
    model_name: str = "yolov8n.pt",
    confidence_threshold: float = 0.25,
    limit: int | None = None,
    retry_failed_downloads: bool = False,
) -> pd.DataFrame:
    """Download Mapillary thumbnails, run YOLO context detection, and update the cache CSV."""
    if not evidence_path.exists():
        raise FileNotFoundError(f"Evidence cache does not exist: {evidence_path}")

    evidence = pd.read_csv(evidence_path)
    evidence = download_evidence_thumbnails(
        evidence,
        image_dir=image_dir,
        limit=limit,
        retry_failed_downloads=retry_failed_downloads,
    )
    download_summary = evidence.attrs.get("thumbnail_download_summary", {})
    evidence = apply_cv_context_to_evidence(
        evidence,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
    )
    evidence.attrs["thumbnail_download_summary"] = download_summary
    evidence.to_csv(evidence_path, index=False)
    return evidence


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
        if image_id:
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
) -> pd.DataFrame:
    """Create a per-image cache CSV with Mapillary metadata plus blank review-label columns."""
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

        image_rows = []
        if pd.notna(query_lon) and pd.notna(query_lat):
            try:
                image_rows = mapillary_image_metadata_rows(
                    mly,
                    query_lon,
                    query_lat,
                    radius_m=radius_m,
                    max_images=max_images_per_segment,
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

        for image_row in image_rows:
            evidence_row = {**base, **image_row}
            for col in VRU_REVIEW_COLUMNS:
                evidence_row.setdefault(col, pd.NA)
            rows.append(evidence_row)

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

        for col in ["mapillary_thumbnail_url", "local_image_path", "notes"]:
            if col in group.columns:
                values = group[col].dropna()
                row[col] = values.iloc[0] if not values.empty else pd.NA

        if "mapillary_image_id" in group.columns:
            ids = [str(value) for value in group["mapillary_image_id"].dropna().tolist()]
            row["mapillary_image_ids"] = ";".join(ids)
            row["mapillary_image_id"] = ids[0] if ids else pd.NA

        if "cv_detected_labels" in group.columns:
            labels = set()
            for value in group["cv_detected_labels"].dropna():
                labels.update(label.strip() for label in str(value).split(";") if label.strip())
            row["cv_detected_labels"] = ";".join(sorted(labels))
        if "cv_detection_count" in group.columns:
            row["cv_detection_count"] = pd.to_numeric(group["cv_detection_count"], errors="coerce").fillna(0).sum()

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
            "local_image_path",
            "cv_detected_labels",
            "cv_detection_count",
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
