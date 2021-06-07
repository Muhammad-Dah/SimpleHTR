import numpy as np
from skimage.filters import gaussian
from skimage.filters import threshold_otsu


def extract_words_from_image(img):
    col_of_zeros = np.ones(shape=(img.shape[0], 5))
    img = np.column_stack((col_of_zeros, img, col_of_zeros))

    # plt.imshow(img, cmap='gray')
    # plt.show()
    # binarize the image, guassian blur will remove any noise in the image
    thresh = threshold_otsu(gaussian(img))
    binary = img > thresh

    # find the vertical projection by adding up the values of all pixels along rows
    vertical_projection = np.sum(binary, axis=0)

    height = img.shape[0]

    # we will go through the vertical projections and
    # find the sequence of consecutive white spaces in the image
    whitespace_lengths = []
    whitespace = 0
    for vp in vertical_projection:
        if vp == height:
            whitespace = whitespace + 1
        elif vp != height:
            if whitespace != 0:
                whitespace_lengths.append(whitespace)
            whitespace = 0  # reset whitepsace counter.

    avg_white_space_length = np.mean(whitespace_lengths)

    # find index of whitespaces which are actually long spaces using the avg_white_space_length
    whitespace_length = 0
    divider_indexes = []
    for index, vp in enumerate(vertical_projection):
        if vp == height:
            whitespace_length = whitespace_length + 1
        elif vp != height:
            if whitespace_length != 0 and whitespace_length > avg_white_space_length:
                divider_indexes.append(index - int(whitespace_length / 2))
                whitespace_length = 0  # reset it

    # lets create the block of words from divider_indexes
    divider_indexes = np.array(divider_indexes)
    divider_indexes = np.insert(divider_indexes, 0, 0)
    divider_indexes = np.append(divider_indexes, img.shape[1])
    dividers = np.column_stack((divider_indexes[:-1], divider_indexes[1:]))

    images = []
    if len(dividers) <= 2:
        images = [img]
    else:
        images = [img[:, window[0]:window[1]] for window in dividers]

    return images
