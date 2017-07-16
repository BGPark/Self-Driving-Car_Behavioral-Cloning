import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def load_image(image_uri):
    return mpimg.imread(image_uri)


def preprocess(image):
    image = rgb2yuv(image)
    return image


def correct_ground_true(image, horizontal_bias=0):
    rows, cols, = image.shape[:2]
    M = np.float32([[1, 0, horizontal_bias], [0, 1, 0]])
    image = cv2.warpAffine(image, M, (cols, rows))
    return image


if __name__ == '__main__':
    image = load_image('./images/ground_true.jpg')
    image = correct_ground_true(image, -20)
    plt.imshow(image)
    plt.show()

