from PIL import Image


def is_ir(image: Image.Image):
    image = image.crop((0, 32, image.width, 64,))
    img_r, img_g, img_b = image.split()
    return img_r == img_g == img_b
