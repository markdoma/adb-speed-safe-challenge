# Safe Speed ADB Challenge

Starter analysis notebook for the Safe Speed ADB Challenge. The notebook loads the challenge GeoJSON, runs basic EDA, filters valid road segments, scores speed-risk indicators, optionally caches Mapillary image metadata, and exports a ranked safety-priority table.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv notebooks/.venv
source notebooks/.venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name safe-speed-adb --display-name "Safe Speed ADB"
```

## Data

Place challenge GeoJSON files in `data/`.

Current notebook inputs:

```text
data/ADB_Innovation_Thailand.geojson
data/ADB_Innovation_Maharashtra.geojson
```

The notebook currently points to the Thailand file. Update `GEOJSON_PATH` in the notebook to switch files.

## Mapillary Token

For Mapillary API access, create a local token file:

```bash
echo "your_mapillary_token_here" > MAPILLARY_TOKEN
```

`MAPILLARY_TOKEN` is ignored by Git.

## Run

Start Jupyter:

```bash
jupyter lab
```

Open:

```text
notebooks/safe_speed_adb_challenge_starter.ipynb
```

Select the `Safe Speed ADB` kernel.

## Outputs

The notebook writes:

```text
outputs/safe_speed_priority_segments.csv
```

If enabled, the Mapillary cache step writes:

```text
data/mapillary_vru_evidence.csv
```
