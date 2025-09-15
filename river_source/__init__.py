from typing import Tuple

from PIL import Image


class ImageData:
    """
    Used to save the Real-time images and metadata.
    """

    realtime_image: Image.Image
    
    river_name: str
    basin_name: str
    country_name: str

    def __init__(
            self,
            realtime_image: Image.Image,
            river_name: str,
            basin_name: str = "",
            country_name: str = ""
    ):
        self.realtime_image = realtime_image
        self.river_name = river_name
        self.basin_name = basin_name
        self.country_name = country_name

class ImageSource:
    async def get_image_data(self) -> Tuple[ImageData, dict]:
        """
        Retrieve the real-time image of the river to predict the depth of water.

        Return
        --------
        An ImageData including the real-time image, metadata, and the inferencing details should be used.
        """

        raise NotImplementedError()

from .wra import WRAImageSource
