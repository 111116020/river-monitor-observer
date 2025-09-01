import io
import json
import pathlib
import argparse
import asyncio
import logging
import platform
import signal
import ssl

from ultralytics import YOLO
import aiohttp
from PIL import Image
from shapely.geometry import Polygon


CAMERA_ID = (1, 96, 0, 0,)  # 安順寮排水出口
CAPTURE_TIMEOUT = 60
MODEL_FOR_NON_IR: YOLO
MODEL_FOR_IR: YOLO


class CustomFormatter(logging.Formatter):

    _grey = "\033[1;30m"
    _purple = "\033[35m"
    _green = "\033[1;32m"
    _blue = "\033[1;34m"
    _yellow = "\033[1;33m"
    _red = "\033[1;31m"
    _reset = "\033[0m"

    _format = f"{_grey}%(asctime)s{_reset} [{{level_color}}%(levelname)s{_reset}]\t{_purple}%(name)s{_reset} %(message)s"

    FORMATS = {
        logging.DEBUG:      _format.format(level_color=_green),
        logging.INFO:       _format.format(level_color=_blue),
        logging.WARNING:    _format.format(level_color=_yellow),
        logging.ERROR:      _format.format(level_color=_red),
        logging.CRITICAL:   _format.format(level_color=_red),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    

async def exec_periodically(interval_sec: int, coro_name, *args, **kwargs):
    while True:
        await coro_name(*args, **kwargs)
        await asyncio.sleep(interval_sec)

async def fetch_image():

    # Temporarily disable strict SSL verification due to TWCA's malformed certificate
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    
    async with aiohttp.ClientSession() as session:
        async with session.get(
            url=f"https://fhyv.wra.gov.tw/FhyWeb/v1/Api/CCTV/WRA/Cameras/{CAMERA_ID[0]}/{CAMERA_ID[1]}",
            ssl=ssl_ctx
        ) as camera_resp:
            camera_resp.raise_for_status()
            
            camera_json = await camera_resp.json()
            logging.debug(f"Camera data: {json.dumps(camera_json)}")
            img_url = camera_json[0]["cameras"][CAMERA_ID[2]]["images"][-1]

            logging.info(f"Downloading \"{img_url}\"...")
            async with session.get(img_url, ssl=ssl_ctx) as img_response:
                return camera_json[0], Image.open(io.BytesIO(await img_response.read()))

async def predict_image(image: Image.Image):

    def is_ir(image: Image.Image):
        image = image.crop((0, 32, image.width, 64,))
        img_r, img_g, img_b = image.split()
        return img_r == img_g == img_b

    logging.debug(f"Source image size is {image.size}")
    model = MODEL_FOR_IR if is_ir(image=image) else MODEL_FOR_NON_IR
    result = model(image)[0]

    predicted_points = result.masks.xyn[0][1:]
    bounding_box = Polygon([[580, 210], [740, 210], [740, 650], [580, 650]])
    polygon = Polygon(predicted_points * image.size).intersection(other=bounding_box)
    predicted_depth = polygon.area / bounding_box.area * 8
    
    return (predicted_points, predicted_depth, )

def main(args):
    
    async def main_loop():
        try:
            if args.image_path:
                camera_json = {
                    "Name": "Test Camera",
                    "Counname": "Test Country",
                    "Basin_name": "Test Basin",
                }
                image = Image.open(args.image_path)
            else:
                camera_json, image = await fetch_image()
            points, depth = await predict_image(image)
        except Exception as e:
            logging.exception("Unable to calculate the depth of water!", exc_info=e)
            return
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="png")
        try:
            # Load upload server's self-signed certificate
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.load_verify_locations(pathlib.Path(__file__).parent.joinpath("server.crt"))

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=args.UPLOAD_URL,
                    data={
                        "river_name": camera_json["Name"],
                        "country_name": camera_json["Counname"],
                        "basin_name": camera_json["Basin_name"],
                        "points": json.dumps(points.tolist()),
                        "depth": str(depth),
                        "image": image_bytes.getvalue()
                    },
                    ssl=ssl_ctx
                ) as resp:
                    resp.raise_for_status()
                    logging.debug("Server response:")
                    logging.debug(await resp.text())
            logging.info("Successfully sent the data.")
        except Exception as e:
            logging.exception("Unable to upload the data!", exc_info=e)
        finally:
            if args.image_path:
                asyncio.get_event_loop().stop()

    event_loop = asyncio.new_event_loop()
    main_task = event_loop.create_task(
        exec_periodically(
            interval_sec=CAPTURE_TIMEOUT,
            coro_name=main_loop
        )
    )

    if platform.system() != "Windows":
        for signum in [signal.SIGINT, signal.SIGTERM]:
            event_loop.add_signal_handler(
                sig=signum, 
                callback=main_task.cancel
            )
    try:
        event_loop.run_until_complete(main_task)
    except asyncio.exceptions.CancelledError:
        pass
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt. Exiting...")
    finally:
        event_loop.close()
        logging.info("Event loop closed.")
    

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("UPLOAD_URL")
    args_parser.add_argument("--image-path", type=pathlib.Path, default=None)
    
    logging_handler = logging.StreamHandler()
    logging_handler.setFormatter(CustomFormatter())
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[logging_handler]
    )

    logging.info("Initializing models...")
    MODEL_FOR_NON_IR = YOLO(
        model=pathlib.Path(__file__).parent.joinpath("models", "model-non_ir.pt"), 
        task="segment"
    )
    MODEL_FOR_IR: YOLO = YOLO(
        model=pathlib.Path(__file__).parent.joinpath("models", "model-ir.pt"), 
        task="segment"
    )
    main(args_parser.parse_args())
