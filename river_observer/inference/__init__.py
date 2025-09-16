from PIL import Image


class InferenceProcessor:
    def inference(self, image: Image.Image) -> float:
        raise NotImplementedError()

class InferenceError(Exception):
    pass

from .wra import WRAInferenceProcessor
