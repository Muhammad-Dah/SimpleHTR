import argparse
import json
import warnings

import cv2
import editdistance
import matplotlib.pyplot as plt
from path import Path

from DataLoaderIAM import DataLoaderIAM, Batch
from Model import Model, DecoderType
from SamplePreprocessor import word_image_preprocess
from spelling_correction import SpellingCorrector

warnings.simplefilter(action='ignore', category=FutureWarning)


class FilePaths:
    """filenames and paths to data"""
    fnCharList = '../model/charList.txt'
    fnSummary = '../model/summary.json'
    fnInfer = '../data/test.png'
    fnLineInfer = '../data/lines/4.png'
    fnCorpus = '../data/corpus.txt'


def write_summary(charErrorRates, wordAccuracies):
    with open(FilePaths.fnSummary, 'w') as f:
        json.dump({'charErrorRates': charErrorRates, 'wordAccuracies': wordAccuracies}, f)


def train(model, loader):
    """train NN"""
    epoch = 0  # number of training epochs since start
    summaryCharErrorRates = []
    summaryWordAccuracies = []
    bestCharErrorRate = float('inf')  # best valdiation character error rate
    noImprovementSince = 0  # number of epochs no improvement of character error rate occured
    earlyStopping = 25  # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.trainSet()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print(f'Epoch: {epoch} Batch: {iterInfo[0]}/{iterInfo[1]} Loss: {loss}')

        # validate
        charErrorRate, wordAccuracy = validate_words(model, loader)

        # write summary
        summaryCharErrorRates.append(charErrorRate)
        summaryWordAccuracies.append(wordAccuracy)
        write_summary(summaryCharErrorRates, summaryWordAccuracies)

        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
        else:
            print(f'Character error rate not improved, best so far: {charErrorRate * 100.0}%')
            noImprovementSince += 1

        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print(f'No more improvement since {earlyStopping} epochs. Training stopped.')
            break


def validate_words(model, loader):
    """validate NN"""
    print('Validate NN')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    correcter = SpellingCorrector()
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print(f'Batch: {iterInfo[0]} / {iterInfo[1]}')
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            corrected_pred = correcter.get_correct(recognized[i])
            numWordOK += 1 if batch.gtTexts[i] == corrected_pred else 0
            numWordTotal += 1
            dist = editdistance.eval(corrected_pred, batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gtTexts[i] + '"', '->',
                  '"' + corrected_pred + '"')

    # print validation result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print(f'Character error rate: {charErrorRate * 100.0}%. Word accuracy: {wordAccuracy * 100.0}%.')
    return charErrorRate, wordAccuracy


def infer_word(model, fnImg: str):
    """recognize text in image provided by file path"""
    img = word_image_preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
    batch = Batch(None, [img])
    (recognized, prob) = model.inferBatch(batch, True)
    return SpellingCorrector().get_correct(recognized), prob[0]


def plot_experiment_results():
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    checkpoint_path = '../model/summary.json'
    with open(checkpoint_path, 'r') as f:
        results = json.load(f)

    train_loss = results['charErrorRates']
    train_accuracy = results['wordAccuracies']

    axs[0].set_title('Training Model Char Error Rates')
    axs[0].plot(train_loss)
    axs[0].set(xlabel="epoch", ylabel="Loss")
    axs[0].grid(True)

    axs[1].set_title('Training Model Word Accuracies')
    axs[1].plot(train_accuracy)
    axs[1].set(xlabel="epoch", ylabel=" Accuracy")
    axs[1].grid(True)

    plt.show()


def main():
    """main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train the NN', action='store_true')
    parser.add_argument('--validate', help='validate the NN', action='store_true')
    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch'], default='bestpath', help='CTC decoder')
    parser.add_argument('--batch_size', help='batch size', type=int, default=100)
    parser.add_argument('--data_dir', help='directory containing IAM dataset', type=Path, required=False,
                        default='../data/words')
    parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')
    args = parser.parse_args()

    # set chosen CTC decoder
    decoderType = DecoderType.BestPath  # if args.decoder == 'bestpath':
    if args.decoder == 'beamsearch':
        decoderType = DecoderType.BeamSearch

    # train or validate on IAM dataset
    if args.train or args.validate:
        # load training data, create TF model
        loader = DataLoaderIAM(args.data_dir, args.batch_size, Model.imgSize, Model.maxTextLen)

        # save characters of model for inference mode
        open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))

        # save img contained in dataset into file
        open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

        # execute training or validation
        if args.train:
            model = Model(loader.charList, decoderType)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, decoderType, mustRestore=True)
            validate_words(model, loader)

    # infer text on test image
    else:
        model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)
        recognized, probability = infer_word(model, FilePaths.fnInfer)
        print(f'Recognized: "{recognized}"')
        print(f'Probability: {probability}')


if __name__ == '__main__':
    main()
    # plot_experiment_results()
