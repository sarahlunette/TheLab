from fastmcp import FastMCP
from tools.osm_tool import run_osm_data_tool
from tools.climate_tool import run_climate_forecast_tool

mcp = FastMCP(tools=[run_osm_data_tool, run_climate_forecast_tool])


if __name__ == "__main__":
    mcp.run()
