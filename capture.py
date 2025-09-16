import asyncio
import pathlib
import time
from river_observer.river_source import WRAImageSource


OUTPUT_PATH = pathlib.Path().joinpath("output")

async def main():
    source = WRAImageSource(camera_id=(1, 96, 0,))
    while True:
        image_data, _ = await source.get_image_data()
        image_data.realtime_image.save(OUTPUT_PATH.joinpath("%d.jpg" % int(time.time())))
        try:
            asyncio.sleep(60)
        except KeyboardInterrupt:
            break
        
if __name__ == "__main__":
    asyncio.run(main=main())
