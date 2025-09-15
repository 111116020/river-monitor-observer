from typing import Tuple

import numpy
from PIL import Image, ImageFilter, ImageEnhance
import scipy

from . import OCRPreprocess


class WRAOCRPreprocess(OCRPreprocess):
    def __init__(self, camera_id: Tuple[int, int, int], threshold: Tuple[int, int]):
        self.camera_id = camera_id
        self.threshold = threshold

    def preprocess(self, image: Image.Image):
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