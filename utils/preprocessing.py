import cv2
import numpy as np
from PIL import Image

def preprocess_image(image):
    # Melakukan prapemrosesan gambar mentah (rezise, grayscale, normalisasi 0-1, 
    # dan manipulasi dimensi/batch) agar sesuai dengan dimensi *input* DenseNet121.
    if isinstance(image, Image.Image):
        img = np.array(image)

    else:
        img = image

    img = cv2.resize(img, (224,224))

    # pastikan grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=-1)  # channel
    img = np.expand_dims(img, axis=0)   # batch

    return img