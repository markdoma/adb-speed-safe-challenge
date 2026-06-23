from pathlib import Path
import webbrowser

import folium
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString


DEFAULT_RAG_THRESHOLDS = {
    "green": 0.75,
    "amber": 0.40,
}

VRU_EVIDENCE_LABELS = [
    ("School", ["school_present", "school_context", "school_context_factor"]),
    ("Market", ["market_present", "market_context", "market_context_factor"]),
    ("School/Market", ["school_or_market_present", "school_or_market_context", "school_or_market_context_factor"]),
    ("Crossing", ["crossing_present", "crosswalk_present", "crossing_present_factor"]),
    ("Sidewalk", ["sidewalk_present", "sidewalk_present_factor"]),
    ("Transit stop", ["transit_stop_present", "transit_stop_context", "transit_stop_context_factor"]),
    (
        "Pedestrian activity",
        ["pedestrian_activity_present", "pedestrian_activity_visible", "pedestrian_activity_visible_factor"],
    ),
    ("Speed sign", ["speed_sign_present", "speed_sign_visible"]),
]

VRU_FACTOR_TOOLTIP_FIELDS = [
    ("is_urban_land_use", "Urban Land Use"),
    ("road_class_vru_weight", "Road Class VRU Weight"),
    ("traffic_component", "Traffic Component"),
    ("sidewalk_present", "Sidewalk Present"),
    ("crossing_present", "Crossing Present"),
    ("school_context", "School Context"),
    ("market_context", "Market Context"),
    ("transit_stop_context", "Transit Stop Context"),
]


def parse_coords_to_line(coord_string):
    """Convert lon/lat endpoint text into a straight LineString."""
    nums = [float(x) for x in str(coord_string).split(",")]
    if len(nums) < 4:
        return None

    start_point = (nums[0], nums[1])
    end_point = (nums[2], nums[3])
    return LineString([start_point, end_point])


def assign_rag_color(score, thresholds: dict | None = None):
    """Assign RAG labels and colors from a priority score."""
    thresholds = thresholds or DEFAULT_RAG_THRESHOLDS
    if pd.isna(score):
        return pd.Series(["Unknown", "#95a5a6"])
    if score >= thresholds["green"]:
        return pd.Series(["Green", "#2ecc71"])
    if score >= thresholds["amber"]:
        return pd.Series(["Amber", "#f39c12"])
    return pd.Series(["Red", "#e74c3c"])


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first available column from a candidate list."""
    return next((column for column in candidates if column in df.columns), None)


def truthy_evidence(value) -> bool:
    """Return whether an evidence value should be treated as present."""
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "present", "visible"}
    return float(value) > 0


def summarize_vru_evidence(row: pd.Series) -> str:
    """Summarize available VRU evidence flags for a tooltip."""
    labels = []
    for label, columns in VRU_EVIDENCE_LABELS:
        if any(column in row.index and truthy_evidence(row[column]) for column in columns):
            labels.append(label)
    return ", ".join(labels)


def format_tooltip_value(value) -> str:
    """Format compact numeric tooltip values."""
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return str(value)
    if float(numeric).is_integer():
        return str(int(numeric))
    return f"{numeric:.3f}".rstrip("0").rstrip(".")


def summarize_vru_factors(row: pd.Series) -> str:
    """Summarize nonzero VRU factor fields for a tooltip."""
    labels = []
    for column, label in VRU_FACTOR_TOOLTIP_FIELDS:
        if column in row.index and truthy_evidence(row[column]):
            labels.append(f"{label}: {format_tooltip_value(row[column])}")
    return "; ".join(labels)


def add_tooltip_fields(gdf: pd.DataFrame) -> pd.DataFrame:
    """Add formatted fields used by the map tooltip."""
    out = gdf.copy()
    segment_col = first_existing_column(out, ["english_ro", "segment_name", "OBJECTID"])
    vru_col = first_existing_column(out, ["vru_exposure_score", "manual_vru_indicator_score"])
    risk_col = first_existing_column(out, ["safety_priority_score", "priority_score"])
    speed_gap_col = first_existing_column(out, ["speed_gap_85th", "speed_gap_median"])

    out["tooltip_segment_name"] = out[segment_col].fillna("Unknown") if segment_col else "Unknown"
    out["tooltip_vru_score"] = pd.to_numeric(out[vru_col], errors="coerce").round(3) if vru_col else pd.NA
    out["tooltip_risk_score"] = pd.to_numeric(out[risk_col], errors="coerce").round(3) if risk_col else pd.NA
    out["tooltip_speed_gap_score"] = pd.to_numeric(out[speed_gap_col], errors="coerce").round(2) if speed_gap_col else pd.NA
    out["tooltip_vru_evidence"] = out.apply(summarize_vru_evidence, axis=1)
    out["has_specific_vru_evidence"] = out["tooltip_vru_evidence"].astype(str).str.strip().ne("")
    out["tooltip_vru_factors"] = out.apply(summarize_vru_factors, axis=1)
    out["has_vru_factor_details"] = out["tooltip_vru_factors"].astype(str).str.strip().ne("")
    return out


def add_priority_segments_layer(
    segment_map: folium.Map,
    gdf: gpd.GeoDataFrame,
    include_vru_evidence: bool,
    include_vru_factors: bool,
) -> None:
    """Add one priority segment layer with a tooltip shape appropriate to the rows."""
    if gdf.empty:
        return

    fields = [
        "tooltip_segment_name",
        "tooltip_vru_score",
        "tooltip_risk_score",
        "tooltip_speed_gap_score",
    ]
    aliases = [
        "Segment Name:",
        "VRU Score:",
        "Risk Score:",
        "Speed Gap Score:",
    ]
    if include_vru_evidence:
        fields.append("tooltip_vru_evidence")
        aliases.append("VRUs Evident:")
    if include_vru_factors:
        fields.append("tooltip_vru_factors")
        aliases.append("VRU Factors:")

    folium.GeoJson(
        gdf,
        style_function=lambda feature: {
            "color": feature["properties"]["color_hex"],
            "weight": 6,
            "opacity": 0.85,
        },
        highlight_function=lambda _: {
            "weight": 9,
            "opacity": 1.0,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=fields,
            aliases=aliases,
            localize=True,
            style=(
                "font-family: Arial, sans-serif; "
                "font-size: 11pt; "
                "padding: 12px; "
                "width: 220px; "
                "max-width: 520px; "
                "white-space: normal; "
                "background-color: white; "
                "border-radius: 4px; "
                "box-shadow: 2px 2px 8px rgba(0,0,0,0.15);"
            ),
        ),
    ).add_to(segment_map)


def priority_segments_to_geodataframe(
    output_df: pd.DataFrame,
    coord_col: str = "StreetImageLink",
    score_col: str | None = None,
    thresholds: dict | None = None,
) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame for mapped priority segments."""
    gdf = output_df.copy()
    gdf["geometry"] = gdf[coord_col].apply(parse_coords_to_line)
    gdf = gdf[gdf["geometry"].notna()].copy()
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    score_col = score_col or first_existing_column(gdf, ["safety_priority_score", "priority_score"])
    if score_col is None:
        raise ValueError("Map data must include safety_priority_score or priority_score.")
    gdf[["status", "color_hex"]] = gdf[score_col].apply(lambda score: assign_rag_color(score, thresholds))
    return add_tooltip_fields(gdf)


def create_priority_segment_map(
    gdf: gpd.GeoDataFrame,
    location: list[float] | None = None,
    zoom_start: int = 11,
    height: int | str = 650,
) -> folium.Map:
    """Create an interactive Folium priority-segment map."""
    location = location or [14.0, 100.55]

    segment_map = folium.Map(
        location=location,
        zoom_start=zoom_start,
        tiles="cartodbpositron",
        width="100%",
        height=height,
    )

    evidence_mask = gdf["has_specific_vru_evidence"].fillna(False).astype(bool)
    factor_mask = gdf["has_vru_factor_details"].fillna(False).astype(bool)
    for include_evidence in [True, False]:
        for include_factors in [True, False]:
            mask = evidence_mask.eq(include_evidence) & factor_mask.eq(include_factors)
            add_priority_segments_layer(
                segment_map,
                gdf[mask],
                include_vru_evidence=include_evidence,
                include_vru_factors=include_factors,
            )

    return segment_map


def save_priority_segment_map(segment_map: folium.Map, output_path: Path) -> Path:
    """Save a Folium priority-segment map to HTML."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segment_map.save(output_path)
    return output_path


def build_priority_segment_map(
    data_path: Path,
    output_path: Path,
    location: list[float] | None = None,
    zoom_start: int = 11,
    height: int | str = 650,
    thresholds: dict | None = None,
) -> Path:
    """Load exported priority segments and save an interactive map."""
    output_df = pd.read_csv(data_path)
    gdf = priority_segments_to_geodataframe(output_df, thresholds=thresholds)
    segment_map = create_priority_segment_map(gdf, location=location, zoom_start=zoom_start, height=height)
    return save_priority_segment_map(segment_map, output_path)


def build_priority_segment_notebook_map(
    data_path: Path,
    output_path: Path | None = None,
    location: list[float] | None = None,
    zoom_start: int = 11,
    height: int | str = 650,
    thresholds: dict | None = None,
) -> folium.Map:
    """Load exported priority segments and return a Folium map object for notebook display."""
    output_df = pd.read_csv(data_path)
    gdf = priority_segments_to_geodataframe(output_df, thresholds=thresholds)
    segment_map = create_priority_segment_map(gdf, location=location, zoom_start=zoom_start, height=height)
    if output_path is not None:
        save_priority_segment_map(segment_map, output_path)
    return segment_map


def open_html_map(output_path: Path) -> bool:
    """Open a generated HTML map in the default browser."""
    return webbrowser.open("file://" + str(Path(output_path).resolve()))
