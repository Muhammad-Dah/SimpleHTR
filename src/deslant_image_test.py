import cv2
import matplotlib.pyplot as plt
import numpy as np

# read and convert image to black and white
from src import deslant_image


def main(filename):
    img = cv2.imread(filename)
    p = deslant_image.RotateAndDeslantImage()
    gray_image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    _, im = cv2.threshold(gray_image, 160, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # call deslant method
    im = p.deslant_image(im)

    # show images

    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

    # write output
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main(filename="../data/deslant_test_img1.png")
    main(filename="../data/deslant_test_img2.png")
