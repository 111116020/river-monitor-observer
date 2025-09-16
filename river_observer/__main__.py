import argparse
import asyncio
import logging

from river_observer import api
from river_observer.util import CustomFormatter
from river_observer.river_source.wra import WRAImageSource


async def main(args):
    fetching_sources = [
        WRAImageSource(camera_id=(1, 96, 0,))   # 安順寮排水出口
    ]

    while True:
        for source in fetching_sources:
            try:
                image_data, inference_data = await source.get_image_data()
                predicted_depth = model.inference(image_data.realtime_image.copy(), inference_data)
                await api.upload(depth=predicted_depth, image_data=image_data)
            except Exception as e:
                logging.exception("An unexpected error occurred.", exc_info=e)
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            break

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()

    logging_handler = logging.StreamHandler()
    logging_handler.setFormatter(CustomFormatter())
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging_handler]
    )

    from river_observer import config

    config_thread = config.start_watcher()

    from river_observer import model

    asyncio.run(main=main(args=args_parser.parse_args()))
    logging.info("Stopped the main event loop.")
    config_thread.stop()
    config_thread.join()
    logging.info("Stopped the configuration watcher.")
