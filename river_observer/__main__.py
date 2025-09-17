import argparse
import asyncio
import logging
import os
import pathlib
import platform
import signal
import sys

from river_observer import api
from river_observer.util import CustomFormatter
from river_observer.river_source.wra import WRAImageSource


async def main():
    # Catch SIGTERM signal
    if platform.system().lower() != "windows":
        def sigterm_handler():
            logging.info("Received SIGTERM! Cancelling all tasks...")
            loop = asyncio.get_event_loop()
            tasks = asyncio.all_tasks(loop=loop)
            for task in tasks:
                task.cancel()

        asyncio.get_event_loop().add_signal_handler(
            sig=signal.SIGTERM,
            callback=lambda: sigterm_handler()
        )

    fetching_sources = [
        WRAImageSource(camera_id=(1, 96, 0,))   # 安順寮排水出口
    ]

    while True:
        try:
            for source in fetching_sources:
                image_data, inference_data = await source.get_image_data()

                from river_observer import inference
                inference_processor = getattr(inference, inference_data["inference_processor"])(**inference_data["init_kwargs"])
                predicted_depth = inference_processor.inference(image_data.realtime_image.copy())

                await api.upload(depth=predicted_depth, image_data=image_data)
        except Exception as e:
            logging.exception("An unexpected error occurred.", exc_info=e)
        except asyncio.CancelledError as e:
            logging.warning("A task was cancelled while it was still running!", exc_info=e)
            break

        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            break

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--config-file", 
        type=pathlib.Path,
        default=pathlib.Path(".").joinpath("config.yaml")
    )
    args = args_parser.parse_args()

    from river_observer import config

    config.load_config(
        config_file=pathlib.Path(args.config_file)
    )

    logging_handler = logging.StreamHandler()
    logging_handler.setFormatter(CustomFormatter())
    logging.basicConfig(
        level=logging.getLevelNamesMapping()[config.get_config().get("logging", {"level": "info"})["level"].upper()],
        handlers=[logging_handler]
    )
    
    asyncio.run(main=main())
    logging.info("Stopped the main event loop.")
