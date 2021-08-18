import logging
import os

import editdistance
import matplotlib.pyplot as plt

from DataLoaderIAM import Batch
from Model import Model
from SamplePreprocessor import word_image_preprocess, image_preprocess
from spelling_correction import SpellingCorrector
from translator import Translator, google_translate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def infer_line(model, images_lst):
    """recognize text in image provided by file path"""

    recognized_words = []
    for img in images_lst:
        img = word_image_preprocess(img, Model.lineImgSize)
        batch = Batch(None, [img])
        (recognized, _) = model.inferBatch(batch)
        recognized_words.append(recognized[0])
    recognized_line = ' '.join(w for w in recognized_words)
    return SpellingCorrector().get_correct(recognized_line)


def infer_batch(model, batch):
    """recognize text in image provided by file path"""
    recognized_lines = [infer_line(model, lst) for lst in batch.imgs]
    return recognized_lines


def validate_lines(model, loader):
    """validate NN"""
    print('Validate NN')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numLinesOK = 0
    numLinesTotal = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print(f'Batch: {iterInfo[0]} / {iterInfo[1]}')
        batch = loader.getNext()
        recognized = infer_batch(model, batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numLinesOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numLinesTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gtTexts[i] + '"', '->',
                  '"' + recognized[i] + '"')
    # print validation result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numLinesOK / numLinesTotal
    print(f'Character error rate: {charErrorRate * 100.0}%. Word accuracy: {wordAccuracy * 100.0}%.')
    return charErrorRate, wordAccuracy


def main():
    """main function"""

    model = Model(mustRestore=True, is_line=True)
    res = image_preprocess('../data/lines/3.png', is_lines=True)
    for (j, w) in enumerate(res):
        plt.imshow(w, cmap='gray')
        plt.show()
    pred_line = infer_line(model, res)
    translator = Translator()
    print(f'\n\n\nPredicted line: \t {pred_line}')
    print('Our translator: \t', translator.translate(pred_line))
    print('Google translator: \t', google_translate(pred_line))


if __name__ == '__main__':
    main()
