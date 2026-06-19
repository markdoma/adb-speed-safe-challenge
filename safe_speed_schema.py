SCHEMA = {
    "OBJECTID": "Unique ID created by GIS software",
    "english_ro": "English street name",
    "OvertureID": "Unique ID created by GIS software",
    "SampleSize_avg": "Average sample size if multiple TomTom datasets were collected",
    "RoadLength": "Ignore; use Shape_Length",
    "WeightedSample": "Average annual traffic weighted value by road length, from TomTom sample size",
    "SampleSize Percent_": "Ignore",
    "Percentile": "Travel percentile for this road",
    "SpeedLimit": "Speed limit obtained by TomTom, not validated",
    "RoadClass": "Overture road class",
    "NumberOverLimit": "Estimated annual vehicles over the speed limit based on the actual sample",
    "MedianSpeed": "Median speed",
    "F85thPercentileSpeed": "85th percentile speed",
    "ForAnalysis": "Ignore",
    "ProvinceID": "Province ID in Thailand only; not useful for this analysis",
    "SpeedLimitFloor": "Ignore",
    "PercentOverLimit": "Percentage estimate based on probe counts and speeds",
    "InvPercentile": "Inverse traffic percentile for dashboard filtering",
    "AnalysisStatus": "Internal validation flag",
    "RankedPercentile": "Presentation ranking of roads by percentage traffic",
    "StreetImageLink": "Lat/lon values for street-view query generation",
    "LandUse": "Urban/rural category based on GRUMP data",
    "NO_OF_Result_Segments": "Ignore",
    "PercentileBand": "Percentile band for dashboard",
    "SampleSizeTotal": "Total sample size if multiple TomTom datasets were collected",
    "Shape_Length": "Geometric length of the shape",
}

KEY_FIELDS = [
    "OBJECTID",
    "english_ro",
    "OvertureID",
    "WeightedSample",
    "Percentile",
    "PercentileBand",
    "SpeedLimit",
    "RoadClass",
    "NumberOverLimit",
    "MedianSpeed",
    "F85thPercentileSpeed",
    "PercentOverLimit",
    "LandUse",
    "SampleSizeTotal",
    "Shape_Length",
    "StreetImageLink",
]

NUMERIC_FIELDS = [
    "SampleSize_avg",
    "RoadLength",
    "WeightedSample",
    "Percentile",
    "SpeedLimit",
    "NumberOverLimit",
    "MedianSpeed",
    "F85thPercentileSpeed",
    "SpeedLimitFloor",
    "PercentOverLimit",
    "InvPercentile",
    "RankedPercentile",
    "SampleSizeTotal",
    "Shape_Length",
]


def schema_rows():
    return [
        {
            "field": field,
            "description": description,
        }
        for field, description in SCHEMA.items()
    ]
