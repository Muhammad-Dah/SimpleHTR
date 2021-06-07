from __future__ import division
from __future__ import print_function

import cv2
import editdistance
import matplotlib.pyplot as plt
import numpy as np

import BKTree
import BeamSearch
import BestPath
import LanguageModel
import LexiconSearch
import Loss
import PrefixSearch
import TokenPassing


def softmax(mat):
    "calc softmax such that labels per time-step form probability distribution"
    maxT, _ = mat.shape  # dim0=t, dim1=c
    res = np.zeros(mat.shape)
    for t in range(maxT):
        y = mat[t, :]
        e = np.exp(y)
        s = np.sum(e)
        res[t, :] = e / s
    return res


def loadRNNOutput(fn):
    "load RNN output from csv file. Last entry in row terminated by semicolon."
    return np.genfromtxt(fn, delimiter=';')[:, : -1]


def testMiniExample():
    "example which shows difference between taking most probable path and most probable labeling. No language model used."

    # chars and input matrix
    classes = 'ab'
    mat = np.array([[0.4, 0, 0.6], [0.4, 0, 0.6]])

    # decode
    gt = 'a'
    print('TARGET       :', '"' + gt + '"')
    print('BEST PATH    :', '"' + BestPath.ctcBestPath(mat, classes) + '"')
    print('PREFIX SEARCH:', '"' + PrefixSearch.ctcPrefixSearch(mat, classes) + '"')
    print('BEAM SEARCH  :', '"' + BeamSearch.ctcBeamSearch(mat, classes, None) + '"')
    print('TOKEN        :', '"' + TokenPassing.ctcTokenPassing(mat, classes, ['a', 'b', 'ab', 'ba']) + '"')
    print('PROB(TARGET) :', Loss.ctcLabelingProb(mat, gt, classes))
    print('LOSS(TARGET) :', Loss.ctcLoss(mat, gt, classes))


def testWordExample():
    "example which decodes a RNN output of a single word. Taken from IAM dataset. RNN output produced by TensorFlow model (see github.com/githubharald/SimpleHTR)."

    # chars of IAM dataset
    classes = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

    # matrix containing TxC RNN output. C=len(classes)+1 because of blank label.
    mat = softmax(loadRNNOutput('../data/word/rnnOutput.csv'))

    # BK tree to find similar words
    with open('../data/word/corpus.txt') as f:
        words = f.read().split()
    tolerance = 4
    bkTree = BKTree.BKTree(words)

    # decode RNN output with different decoding algorithms
    gt = 'aircraft'
    print('TARGET        :', '"' + gt + '"')
    print('BEST PATH     :', '"' + BestPath.ctcBestPath(mat, classes) + '"')
    print('LEXICON SEARCH:', '"' + LexiconSearch.ctcLexiconSearch(mat, classes, bkTree, tolerance) + '"')


def plot_training_results(classes, mat):
    plt.imshow(mat.transpose())
    plt.colorbar()
    plt.show()

    plt.imshow(cv2.imread('../data/line/img.png'))
    plt.grid(False)
    plt.show()

    o_index = classes.index('o')
    f_index = classes.index('f')
    blank_index = classes.index(' ')

    o_freq = mat[:, o_index]
    f_freq = mat[:, f_index]
    blank_freq = mat[:, blank_index]

    p1, _, _ = plt.stem(o_freq, markerfmt='o', label="o 68", use_line_collection=True)
    p1.set_markerfacecolor('none')
    p2, _, _ = plt.stem(f_freq, markerfmt='o', label="f 59", use_line_collection=True)
    p2.set_markerfacecolor('none')
    p3, _, _ = plt.stem(blank_freq, markerfmt='o', label="blank", use_line_collection=True)
    p3.set_markerfacecolor('none')

    plt.legend(loc='upper right')
    plt.show()


def testLineExample():
    "example which decodes a RNN output of a text line. Taken from IAM dataset. RNN output produced by TensorFlow model."

    # chars of IAM dataset
    classes = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

    # matrix containing TxC RNN output. C=len(classes)+1 because of blank label.
    mat = softmax(loadRNNOutput('../data/line/rnnOutput.csv'))

    # language model: used for token passing (word list) and beam search (char bigrams)
    lm = LanguageModel.LanguageModel('../data/line/corpus.txt', classes)

    # decode RNN output with different decoding algorithms
    gt = 'the fake friend of the family, like the'

    results = {'TARGET': gt,
               'BEST PATH ': BestPath.ctcBestPath(mat, classes),
               'PREFIX SEARCH': PrefixSearch.ctcPrefixSearchHeuristicSplit(mat, classes),
               'BEAM SEARCH': BeamSearch.ctcBeamSearch(mat, classes, None),
               'BEAM SEARCH LM': BeamSearch.ctcBeamSearch(mat, classes, lm),
               'TOKEN': TokenPassing.ctcTokenPassing(mat, classes, lm.getWordList()),
               }

    for key, value in results.items():
        print(f'{key:15s}: \"{value:35}\" {editdistance.eval(value, gt)}')

    print('PROB(TARGET)  :', Loss.ctcLabelingProb(mat, gt, classes))
    print('LOSS(TARGET)  :', Loss.ctcLoss(mat, gt, classes))

    return classes, mat


if __name__ == '__main__':
    # example decoding matrix containing 2 time-steps and 2 chars
    print('=====Mini example=====')
    testMiniExample()

    # example decoding a word
    print('=====Word example=====')
    testWordExample()

    # example decoding a text-line
    print('=====Line example=====')
    classes, mat = testLineExample()

    plot_training_results(classes, mat)
