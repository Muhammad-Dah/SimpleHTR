import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

from word_segmentation import extract_words_from_image


def word_image_preprocess(img, imgSize=(128, 32), dataAugmentation=False):
    """put img into target img of size imgSize, transpose for TF and normalize gray-values"""

    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = np.zeros(imgSize[::-1])

    # data augmentation
    img = img.astype(np.float)
    if dataAugmentation:
        # photometric data augmentation
        if random.random() < 0.25:
            rand_odd = lambda: random.randint(1, 3) * 2 + 1
            img = cv2.GaussianBlur(img, (rand_odd(), rand_odd()), 0)
        if random.random() < 0.25:
            img = cv2.dilate(img, np.ones((3, 3)))
        if random.random() < 0.25:
            img = cv2.erode(img, np.ones((3, 3)))
        if random.random() < 0.5:
            img = img * (0.25 + random.random() * 0.75)
        if random.random() < 0.25:
            img = np.clip(img + (np.random.random(img.shape) - 0.5) * random.randint(1, 50), 0, 255)
        if random.random() < 0.1:
            img = 255 - img

        # geometric data augmentation
        wt, ht = imgSize
        h, w = img.shape
        f = min(wt / w, ht / h)
        fx = f * np.random.uniform(0.75, 1.25)
        fy = f * np.random.uniform(0.75, 1.25)

        # random position around center
        txc = (wt - w * fx) / 2
        tyc = (ht - h * fy) / 2
        freedom_x = max((wt - fx * w) / 2, 0) + wt / 10
        freedom_y = max((ht - fy * h) / 2, 0) + ht / 10
        tx = txc + np.random.uniform(-freedom_x, freedom_x)
        ty = tyc + np.random.uniform(-freedom_y, freedom_y)

        # map image into target image
        M = np.float32([[fx, 0, tx], [0, fy, ty]])
        target = np.ones(imgSize[::-1]) * 255 / 2
        img = cv2.warpAffine(img, M, dsize=imgSize, dst=target, borderMode=cv2.BORDER_TRANSPARENT)

    # no data augmentation
    else:
        # center image
        wt, ht = imgSize
        h, w = img.shape
        f = min(wt / w, ht / h)
        tx = (wt - w * f) / 2
        ty = (ht - h * f) / 2

        # map image into target image
        M = np.float32([[f, 0, tx], [0, f, ty]])
        target = np.ones(imgSize[::-1]) * 255 / 2
        img = cv2.warpAffine(img, M, dsize=imgSize, dst=target, borderMode=cv2.BORDER_TRANSPARENT)

    # transpose for TF
    img = cv2.transpose(img)

    # convert to range [-1, 1]
    img = img / 255 - 0.5
    return img


def image_preprocess(filename, is_lines=False):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) / 255

    images = [img]
    if is_lines:
        images = extract_words_from_image(img)

    def _word_im(_img):
        # center image
        imgSize = (128, 48) if is_lines else (128, 32)
        wt, ht = imgSize
        h, w = _img.shape
        f = min(wt / w, ht / h)
        tx = (wt - w * f) / 2
        ty = (ht - h * f) / 2

        # map image into target image
        M = np.float32([[f, 0, tx], [0, f, ty]])
        target = np.ones(imgSize[::-1]) * 255 / 2
        _img = cv2.warpAffine(_img, M, dsize=imgSize, dst=target, borderMode=cv2.BORDER_TRANSPARENT)
        return _img

    images = [_word_im(img * 255) for img in images]
    return images


if __name__ == '__main__':

    res = image_preprocess('../data/lines/3.png', is_lines=True)
    for (j, w) in enumerate(res):
        plt.imshow(w, cmap='gray')
        plt.show()
