import io
import json
import logging
import ssl
from typing import Tuple

import aiohttp
from PIL import Image

from . import ImageSource, ImageData

from .util import is_ir


class WRAImageSource(ImageSource):
    def __init__(self, camera_id: Tuple[int, int, int]):
        self._logger = logging.getLogger(self.__class__.__name__)

        # self._logger.debug("Creating SSL context for malformed X.509 certificates...")
        self._ssl_ctx = ssl.create_default_context()
        # self._ssl_ctx &= ~ssl.VERIFY_X509_STRICT

        self.camera_id = camera_id
        self._logger.info("Initialized with camera ID (%d, %d, %d).", *camera_id)


    async def get_image_data(self):
        async with aiohttp.ClientSession() as session:
            self._logger.info("Fetching camera data (camera=(%d, %d, %d))...", *self.camera_id)
            async with session.get(
                url="https://fhyv.wra.gov.tw/FhyWeb/v1/Api/CCTV/WRA/Cameras/%d/%d" % self.camera_id[:2],
                ssl=self._ssl_ctx,
                headers={
                    "Content-Type": "application/json"
                }
            ) as camera_resp:
                camera_resp.raise_for_status()
                
                camera_json = await camera_resp.json()
                self._logger.debug(f"Fetched camera data: {json.dumps(camera_json)}")
                img_url = camera_json[0]["cameras"][self.camera_id[2]]["images"][-1]

            self._logger.info("Downloading real-time image (camera=(%d, %d, %d))...", *self.camera_id)
            async with session.get(img_url, ssl=self._ssl_ctx) as img_response:
                img_response.raise_for_status()
                image = Image.open(io.BytesIO(await img_response.read()))
                image_is_ir = is_ir(image)

                inference_data = {
                    "model_name": "WRA-%d_%d_%d-%s" % (*self.camera_id, "ir" if image_is_ir else "non_ir"),
                    "gauge_info": {
                        "max": 8,
                        "meter_in_pixel": 1324
                    } if self.camera_id == (1, 96, 0) else {} ,
                    "postprocess": []
                }
                if not image_is_ir:
                    inference_data["postprocess"].append({
                        "action": "ocr", 
                        "ocr_module": ".wra",
                        "ocr_class": "WRAOCRPreprocess", 
                        "ocr_kwargs": { 
                            "camera_id": self.camera_id,
                            "threshold": (100, 300)
                        }
                    })

                return ImageData(
                    realtime_image=image,
                    river_name=camera_json[0]["Name"],
                    basin_name=camera_json[0]["Basin_name"],
                    country_name=camera_json[0]["Counname"]
                ), inference_data