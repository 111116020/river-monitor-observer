import logging
import pathlib

from PIL import Image
import numpy
import pytesseract
from ultralytics import YOLO


TESSDATA_DIR = pathlib.Path(__file__).parent.joinpath("tessdata")
_models_logger = logging.getLogger("models")
_models_logger.info("Tessdata directory is set to \"%s\".", TESSDATA_DIR)


def inference(image: Image.Image, inference_data: dict) -> float:
    model_name = inference_data["model_name"]
    if model_name not in _models.keys():
        raise ValueError("Requested model \"%s\" is not loaded." % model_name)
    result = _models[model_name].predict(image)[0]
    conf, bound = sorted(
        zip(result.boxes.conf, result.boxes.xyxy),
        key=lambda x: -x[0]
    )[0]
    gauge_image = image.crop(box=bound.tolist())

    predicted_value = None
    for postprocess in inference_data["postprocess"]:
        if postprocess["action"] == "ocr":
            import ocr
            ocr_class = getattr(ocr, postprocess["ocr_class"])
            filtered_image = ocr_class(**postprocess["ocr_kwargs"]).preprocess(gauge_image.copy())
            ocr_result = pytesseract.image_to_data(
                image=filtered_image, 
                config="--tessdata-dir \"%s\" --psm 6 -c tessedit_char_whitelist=\"0123456789\" digits" % TESSDATA_DIR,
                output_type=pytesseract.Output.DICT,
                lang=model_name
            )
            ocr_result = list(zip(
                ocr_result["text"],
                ocr_result["left"],
                ocr_result["top"],
                ocr_result["width"],
                ocr_result["height"]
            ))
            ocr_result.sort(key=lambda x: -x[2])

            for x in range(len(ocr_result)):
                if not ocr_result[x][0].isdigit():
                    continue
                x_num = int(ocr_result[x][0])
                x_mid = ocr_result[x][2] + ocr_result[x][4] / 2
                for y in range(x + 1, len(ocr_result)):
                    if not ocr_result[y][0].isdigit():
                        continue
                    y_num = int(ocr_result[y][0])
                    if y_num <= x_num:
                        break
                    y_mid = ocr_result[y][2] + ocr_result[y][4] / 2
                    predicted_value = x_num + 0.25 - (filtered_image.height - x_mid) / (x_mid - y_mid) * (y_num - x_num)
                    break
                if predicted_value is not None:
                    break

    if predicted_value is None:
        # Calculate the area of non-transparent pixels
        gauge_image_r, gauge_image_g, gauge_image_b, gauge_image_a = gauge_image.split()
        gauge_image_r_arr = numpy.asarray(gauge_image_r)
        gauge_image_g_arr = numpy.asarray(gauge_image_g)
        gauge_image_b_arr = numpy.asarray(gauge_image_b)
        gauge_image_a_arr = numpy.asarray(gauge_image_a)
        gauge_area = numpy.count_nonzero(gauge_image_a_arr > 0)
        # for x in range(gauge_image.width):
        #     for y in range(gauge_image.height):
        #         _, _, _, a = gauge_image.getpixel((x, y))
        #         # if a <= 0 or round(r / 10) == round(g / 10) == round(b / 10) <= 6:
        #         if a <= 0:
        #             continue
        #         gauge_area += 1

        predicted_value = inference_data["gauge_info"]["max"] - gauge_area / inference_data["gauge_info"]["meter_in_pixel"]
    
    return predicted_value

# Load models
_models: dict[pathlib.Path, YOLO] = {}
for model_file in pathlib.Path(__file__).parent.joinpath("models").iterdir():
    if not model_file.is_file():
        _models_logger.warning(
            "Skipping \"%s\" since \"%s\" is not a file.",
            str(model_file), model_file.stem
        )
        continue
    if model_file.stem in _models:
        _models_logger.warning(
            "Skipping \"%s\" since \"%s\" is loaded already.",
            str(model_file), model_file.stem
        )
        continue
    try:
        _models[model_file.stem] = YOLO(model=model_file, task="detect")
        _models_logger.info(
            "Loaded model with name \"%s\" from \"%s\".",
            model_file.stem, str(model_file)
        )
    except Exception as e:
        _models_logger.warning(
            "Failed to load model from \"%s\".",
            str(model_file),
            exc_info=e
        )
