import argparse
import asyncio
import logging

import api
from river_source.wra import WRAImageSource
from util import CustomFormatter


async def main(args):
    fetching_sources = [
        WRAImageSource(camera_id=(1, 96, 0,))   # 安順寮排水出口
    ]

    while True:
        for source in fetching_sources:
            image_data, inference_data = await source.get_image_data()
            predicted_depth = model.inference(image_data.realtime_image.copy(), inference_data)
            await api.upload(depth=predicted_depth, image_data=image_data)
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            break

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    
    logging_handler = logging.StreamHandler()
    logging_handler.setFormatter(CustomFormatter())
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[logging_handler]
    )

    import model
    asyncio.run(main=main(args=args_parser.parse_args()))

    # event_loop = asyncio.new_event_loop()
    # main_task = event_loop.create_task(main(args_parser.parse_args()))

    # Windows does not have signal, so skip if the OS is Windows.
    # if platform.system().lower() != "windows":
    #     for signum in [signal.SIGINT, signal.SIGTERM]:
    #         event_loop.add_signal_handler(
    #             sig=signum, 
    #             callback=main_task.cancel
    #         )
    # try:
    #     event_loop.run_until_complete(main_task)
    # except asyncio.exceptions.CancelledError:
    #     pass
    # except KeyboardInterrupt:
    #     logging.info("Received keyboard interrupt. Exiting...")
    #     if platform.system().lower() == "windows":
    #         print(main_task.cancel())
    #     print(main_task.cancelling())
    # finally:
    #     event_loop.close()
    #     logging.info("Event loop closed.")
