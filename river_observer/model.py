import logging

from PIL import Image


def inference(image: Image.Image, inference_data: dict) -> float:
    from river_observer import inference

    inference_processor: inference.InferenceProcessor = getattr(inference, inference_data["inference_processor"])(**inference_data["init_kwargs"])
    return inference_processor.inference(image)
