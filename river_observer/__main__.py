import argparse
import asyncio
import logging
import platform
import signal

from river_observer import api
from river_observer.util import CustomFormatter
from river_observer.river_source.wra import WRAImageSource


async def main(args):
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
                predicted_depth = model.inference(image_data.realtime_image.copy(), inference_data)
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

    from river_observer import config

    config_thread = config.start_watcher()

    logging_handler = logging.StreamHandler()
    logging_handler.setFormatter(CustomFormatter())
    logging.basicConfig(
        level=logging.getLevelNamesMapping()[config.get_config().get("logging", {"level": "info"})["level"].upper()],
        handlers=[logging_handler]
    )

    from river_observer import model

    asyncio.run(main=main(args=args_parser.parse_args()))
    logging.info("Stopped the main event loop.")
    config_thread.stop()
    config_thread.join()
    logging.info("Stopped the configuration watcher.")
