import os
import overpy
import pandas as pd
import anyio
from pathlib import Path


def get_osm_data_tool(location: str, features: list[str]) -> str:
    """
    Fetch OSM data for a given location and list of features (tags),
    and save the results as a CSV file.

    Args:
        location (str): The geographic area name (e.g. 'Marigot', 'Saint-Martin').
        features (list[str]): List of OSM features/tags (e.g. ['highway', 'amenity', 'building']).

    Returns:
        str: Path to the saved CSV file or an error message.
    """
    api = overpy.Overpass()
    results = []

    output_dir = Path("./data/osm_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    for feature in features:
        print(f"📡 Fetching {feature} in {location} ...")
        query = f"""
        [out:json];
        area["name"="{location}"]->.searchArea;
        (
          node["{feature}"](area.searchArea);
          way["{feature}"](area.searchArea);
          relation["{feature}"](area.searchArea);
        );
        out center;
        """
        try:
            r = api.query(query)
        except Exception as e:
            print(f"⚠️ Error fetching {feature}: {e}")
            continue

        for node in r.nodes:
            results.append({
                "type": "node",
                "feature": feature,
                "name": node.tags.get("name", ""),
                "lat": node.lat,
                "lon": node.lon
            })
        for way in r.ways:
            results.append({
                "type": "way",
                "feature": feature,
                "name": way.tags.get("name", ""),
                "lat": getattr(way, "center_lat", None),
                "lon": getattr(way, "center_lon", None)
            })

    if not results:
        return "❌ No OSM data found or query returned empty results."

    df = pd.DataFrame(results)
    output_path = output_dir / f"osm_{location.replace(' ', '_')}.csv"
    df.to_csv(output_path, index=False)

    return f"✅ OSM data saved at {output_path}"


# ✅ Async-safe wrapper
async def run_osm_data_tool(location: str, features: list[str]) -> str:
    """
    Run blocking Overpass API calls in a background thread to avoid blocking FastMCP.
    """
    return await anyio.to_thread.run_sync(get_osm_data_tool, location, features)
