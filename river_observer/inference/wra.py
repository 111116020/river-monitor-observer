import logging
import pathlib
import time
from typing import Tuple

from PIL import Image, ImageEnhance, ImageFilter
import numpy
import pytesseract
import scipy
from ultralytics import YOLO

from river_observer import config
from river_observer.inference import InferenceProcessor


class WRAInferenceProcessor(InferenceProcessor):
    def __init__(self, camera_id: Tuple[int, int, int], gauge_info: dict):
        super().__init__()
        self.camera_id = camera_id
        self.gauge_info = gauge_info
        self._logger = logging.getLogger(self.__class__.__name__)

    def _is_ir(self, image: Image.Image):
        image = image.crop((0, 32, image.width, 64,))
        img_r, img_g, img_b = image.split()
        return img_r == img_g == img_b
    
    def _ocr_1_96_0_nonir_preprocess(self, image: Image.Image):
        image = image.filter(ImageFilter.SHARPEN)
        image = ImageEnhance.Brightness(image).enhance(0.8)
        image = ImageEnhance.Contrast(image).enhance(5)
        
        expanded_image = Image.new("RGB", (image.width + 16, image.height + 16), "red")
        expanded_image.paste(image, (8, 8))

        img_r, img_g, img_b = expanded_image.split()
        img_r_arr = numpy.asarray(img_r)
        img_g_arr = numpy.asarray(img_g)
        img_b_arr = numpy.asarray(img_b)
        # image_arr = (img_r_arr > 128) & (img_g_arr < 128) & (img_b_arr < 128)
        img_g_arr = img_g_arr + (img_g_arr == 0)    # Avoid division by zero
        img_b_arr = img_b_arr + (img_b_arr == 0)    # Avoid division by zero
        image_arr = (img_r_arr > 96) & (img_r_arr / img_g_arr > 1) & (img_r_arr / img_b_arr > 1)
        
        labels, num_groups = scipy.ndimage.label(image_arr)
        for i in range(1, num_groups + 1):
            m = (labels == i)
            rows, cols = m.nonzero()
            width = cols.max() + 1 - cols.min()
            height = rows.max() + 1 - rows.min()
            size_of_component = m.sum()
            if not (self.threshold[0] < size_of_component < self.threshold[1]):
                labels[labels == i] = 0
            if not (10 < width < 30 and 10 < height < 30):
                labels[labels == i] = 0
            if width / height > 1:
                labels[labels == i] = 0
        image_arr = scipy.ndimage.binary_dilation(labels != 0, [
            [True, True],
            [True, True],
        ])
        result_image = Image.fromarray(~image_arr)
        return result_image.crop(
            box=(8, 8, result_image.width - 8, result_image.height - 8)
        )

    def _ocr_1_96_0_nonir(self, gauge_image: Image.Image, tesseract_config: dict = {}):
        if "tesseract_cmd" in tesseract_config.keys():
            pytesseract.pytesseract.tesseract_cmd = tesseract_config["tesseract_cmd"]
        ocr_config = "--psm 6 -c tessedit_char_whitelist=\"0123456789\""
        if "tessdata_path" in tesseract_config.keys():
            ocr_config += " --tessdata-dir \"%s\"" % tesseract_config["tessdata_path"]
        ocr_result = pytesseract.image_to_data(
            image=gauge_image, 
            config=ocr_config,
            output_type=pytesseract.Output.DICT,
            lang="WRA_%d-%d-%d_non-ir" % self.camera_id
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
                return x_num + 0.25 - (gauge_image.height - x_mid) / (x_mid - y_mid) * (y_num - x_num)
        return -1.0

    def _area_1_96_0_nonir(self, gauge_image: Image.Image):
        # Calculate the area of red pixels
        gauge_area = numpy.count_nonzero(numpy.asarray(gauge_image.split()[0]) & 0b10000000)
        return self.gauge_info["max"] - gauge_area / self.gauge_info["meter_in_pixel"]

    def inference(self, image: Image.Image):
        image_is_ir = self._is_ir(image)

        # Water gauge detection
        model_path = pathlib.Path(config.get_config()["models"]["path"]).joinpath(
            "WRA_%d-%d-%d_%s.pt" % (*self.camera_id, "ir" if image_is_ir else "non-ir")
        )
        self._logger.info("Loading gauge detection model from \"%s\"...", model_path)
        detect_model = YOLO(model_path, task="detect")
        detect_result = detect_model.predict(image)[0]
        _, bound = sorted(
            zip(detect_result.boxes.conf, detect_result.boxes.xyxy),
            key=lambda x: -x[0]
        )[0]
        gauge_image = image.crop(box=bound.tolist())
        del detect_model

        # OCR
        inferenced_depth = -1.0
        tesseract_config = config.get_config().get("tesseract", {})
        if self.camera_id == (1, 96, 0,):
            if image_is_ir:
                pass
            else:
                inferenced_depth = self._ocr_1_96_0_nonir(
                    gauge_image=self._ocr_1_96_0_nonir_preprocess(gauge_image), 
                    tesseract_config=tesseract_config
                )
        
        # Estimate with area
        if inferenced_depth < 0:
            self._logger.info("OCR failed! Falling back to area calculation.")
            image.save("%d.jpg" % time.time())
            if self.camera_id == (1, 96, 0,):
                if image_is_ir:
                    pass
                else:
                    inferenced_depth = self._area_1_96_0_nonir(gauge_image)

        return inferenced_depth
