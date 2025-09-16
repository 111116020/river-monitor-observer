import io
import logging
import aiohttp

from river_source import ImageData


API_BASE_URL = "https://project-0331.eastasia.cloudapp.azure.com/api"

_api_logger = logging.getLogger("API")

async def upload(depth: float, image_data: ImageData):
    image_bytes = io.BytesIO()
    image_data.realtime_image.save(image_bytes, format="png")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url=f"{API_BASE_URL}/upload",
            data={
                "river_name": image_data.river_name,
                "country_name": image_data.country_name,
                "basin_name": image_data.basin_name,
                "depth": str(depth),
                "image": image_bytes.getvalue()
            }
        ) as resp:
            resp.raise_for_status()
            _api_logger.debug("Server response: %s", await resp.text())
    _api_logger.info("Successfully sent the data.")
