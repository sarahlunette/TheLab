import os
import yaml
import cdsapi
import xarray as xr
import pandas as pd
import anyio
from pathlib import Path


def get_climate_forecast_tool(
    date_str: str,
    area: list[float] = [18.2, -63.2, 18.0, -62.9]
) -> str:
    """
    Fetch seasonal climate forecast from Copernicus CDS API and convert to CSV.

    Args:
        date_str (str): Date in 'YYYY-MM' format.
        area (list[float]): Bounding box [N, W, S, E].

    Returns:
        str: Path to the generated CSV file or error message.
    """
    output_dir = "./data/climate_data"
    os.makedirs(output_dir, exist_ok=True)

    config_file = Path("config/.cdsapirc")
    if not config_file.exists():
        return "❌ Missing Copernicus config file at config/.cdsapirc"

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    os.environ["CDSAPI_URL"] = config["url"]
    os.environ["CDSAPI_KEY"] = config["key"]

    client = cdsapi.Client()

    filename = f"{output_dir}/forecast_{date_str}.nc"
    leadtimes = [str(i) for i in range(0, 168 + 1, 6)]

    try:
        client.retrieve(
            "seasonal-original-single-levels",
            {
                "originating_centre": "ecmwf",
                "system": "51",
                "variable": ["2m_temperature", "total_precipitation"],
                "year": date_str[:4],
                "month": date_str[5:7],
                "day": "01",
                "leadtime_hour": leadtimes,
                "area": area,
                "format": "netcdf",
            },
            filename,
        )
    except Exception as e:
        return f"❌ Error fetching forecast: {e}"

    ds = xr.open_dataset(filename)
    df = ds.to_dataframe().reset_index()

    if "t2m" in df.columns:
        df["t2m"] -= 273.15  # Convert Kelvin → Celsius

    csv_file = filename.replace(".nc", ".csv")
    df.to_csv(csv_file, index=False)
    return f"✅ Climate forecast CSV saved at {csv_file}"


# ✅ Async-safe wrapper
async def run_climate_forecast_tool(date_str: str, area: list[float] = [18.2, -63.2, 18.0, -62.9]) -> str:
    """
    Run the blocking Copernicus CDS API call in a background thread to prevent blocking.
    """
    return await anyio.to_thread.run_sync(get_climate_forecast_tool, date_str, area)
