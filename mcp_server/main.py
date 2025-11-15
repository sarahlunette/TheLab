from fastmcp import FastMCP

# Import the async wrappers
from tools.osm_tool import run_osm_data_tool
from tools.climate_tool import run_climate_forecast_tool

# Create FastMCP server with our tools
mcp = FastMCP("Resilience Crisis Tools", version="1.0.0")

# Add tools to the server
mcp.add_tool(run_osm_data_tool)
mcp.add_tool(run_climate_forecast_tool)

if __name__ == "__main__":
    # Run with SSE transport for HTTP/WebSocket access
    mcp.run(transport="sse")
