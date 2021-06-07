import collections
import json
import os
import pickle

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import GRU, Dense, TimeDistributed, RepeatVector, Bidirectional, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from translator_model import translator_tests


def load_data(path):
    """
    Load dataset
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data.split('\n')


def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    return tokenizer.texts_to_sequences(x), tokenizer


def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    return pad_sequences(x, maxlen=length, padding='post')


def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk


def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


def model_final(input_shape=(137861, 15), output_sequence_length=21, english_vocab_size=200, french_vocab_size=345):
    """
    Build and train a model that incorporates embedding, encoder-decoder, and bidirectional RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """

    # Hyperparameters
    learning_rate = 0.003

    # Build the layers
    model = Sequential()
    # Embedding
    model.add(Embedding(english_vocab_size, 128, input_length=input_shape[1],
                        input_shape=input_shape[1:]))
    # Encoder
    model.add(Bidirectional(GRU(128)))
    model.add(RepeatVector(output_sequence_length))
    # Decoder
    model.add(Bidirectional(GRU(128, return_sequences=True)))
    model.add(TimeDistributed(Dense(512, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model


"""## Prediction (IMPLEMENTATION)"""


def train(x, y, x_tk, y_tk, use_check_point=False):
    """
    train the translation model and save checkpoint
    :param x: Preprocessed English data
    :param y: Preprocessed French data
    :param x_tk: English tokenizer
    :param y_tk: French tokenizer
    :param use_check_point: whether save model weights to file or not
    """

    model = model_final(x.shape, y.shape[1],
                        len(x_tk.word_index) + 1,
                        len(y_tk.word_index) + 1)
    model.summary()

    checkpoint_path = "translator_checkpoint_dir/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if use_check_point and os.listdir(checkpoint_dir).__len__() > 0:
        latest_cp = tf.train.latest_checkpoint(checkpoint_dir)
        print(f'loading last model from {latest_cp}')
        model.load_weights(latest_cp)

        with open(checkpoint_dir + '/' + 'summary', 'r') as f:
            summary = json.load(f)

        return model, summary


    else:
        # Create a callback that saves the model's weights
        cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      save_weights_only=True,
                                                      verbose=1)
        # Train the model with the new callback
        summary = model.fit(x, y,
                            batch_size=1024, epochs=25, validation_split=0.2,
                            callbacks=[cp_callback])  # Pass callback to training

        with open(checkpoint_dir + '/' + 'summary', 'w') as f:
            json.dump(summary.history, f)

        return model, summary


def test_model(model, x_tk, y_tk, x_max_len=15):
    """
    Gets predictions using the final model
    :param x: Preprocessed English data
    :param y: Preprocessed French data
    :param x_tk: English tokenizer
    :param y_tk: French tokenizer
    :param model: translation model instance
    """

    y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
    y_id_to_word[0] = '<PAD>'

    raw_sentence = 'he saw a old yellow truck'
    sentence = [x_tk.word_index[word] for word in raw_sentence.split()]
    sentence = pad_sequences([sentence], maxlen=x_max_len, padding='post')
    sentences = np.array([sentence[0]])
    predictions = model.predict(sentences, len(sentences))

    print('Sample 1: ', raw_sentence)
    sentence_pred = ' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]).replace('<PAD>', '')
    print('Model prediction: ', sentence_pred)


def main(use_check_point):
    # Load English data
    english_sentences = load_data('data/small_vocab_en')
    # Load French data
    french_sentences = load_data('data/small_vocab_fr')
    preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = \
        preprocess(english_sentences, french_sentences)

    model, summary = train(preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer,
                           use_check_point)
    return model, summary


def plot_training_results(results):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    train_loss = results['loss']
    train_accuracy = results['accuracy']
    validation_loss = results['val_loss']
    validation_accuracy = results['val_accuracy']

    axs[0].set_title('Training translation model progress Loss')
    axs[0].plot(train_loss, label="train loss")
    axs[0].plot(validation_loss, label="validation loss")
    axs[0].set(xlabel="epoch", ylabel="Loss")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].set_title('Training translation model progress accuracy')
    axs[1].plot(train_accuracy, label="train accuracy")
    axs[1].plot(validation_accuracy, label="validation accuracy")
    axs[1].set(xlabel="epoch", ylabel="accuracy")
    axs[1].grid(True)
    axs[1].legend()

    plt.show()


def get_en_fr_tokenizers(checkpoint_path='translator_checkpoint_dir/tokenizers'):
    if not os.path.isfile(checkpoint_path):
        english_sentences = load_data('data/small_vocab_en')
        french_sentences = load_data('data/small_vocab_fr')
        english_tokenizer = Tokenizer()
        english_tokenizer.fit_on_texts(english_sentences)
        french_tokenizer = Tokenizer()
        french_tokenizer.fit_on_texts(french_sentences)
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({'english_tokenizer': english_tokenizer, 'french_tokenizer': french_tokenizer}, f)
            return english_tokenizer, french_tokenizer

    else:
        with open(checkpoint_path, 'rb') as f:
            tokenizers_dict = pickle.load(f)
            english_tokenizer = tokenizers_dict['english_tokenizer']
            french_tokenizer = tokenizers_dict['french_tokenizer']
            return english_tokenizer, french_tokenizer


def preprocessing_experiments():
    # Load English data
    english_sentences = load_data('data/small_vocab_en')
    # Load French data
    french_sentences = load_data('data/small_vocab_fr')

    for sample_i in range(5):
        print('English sample {}:  {}'.format(sample_i + 1, english_sentences[sample_i]))
        print('French sample {}:  {}\n'.format(sample_i + 1, french_sentences[sample_i]))

    english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
    french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])

    print('{} English words.'.format(len([word for sentence in english_sentences for word in sentence.split()])))
    print('{} unique English words.'.format(len(english_words_counter)))
    print('10 Most common words in the English dataset:')
    print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
    print()
    print('{} French words.'.format(len([word for sentence in french_sentences for word in sentence.split()])))
    print('{} unique French words.'.format(len(french_words_counter)))
    print('10 Most common words in the French dataset:')
    print('"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')

    translator_tests.test_tokenize(tokenize)

    # Tokenize Example output
    text_sentences = [
        'The quick brown fox jumps over the lazy dog .',
        'By Jove , my quick study of lexicography won a prize .',
        'This is a short sentence .']
    text_tokenized, text_tokenizer = tokenize(text_sentences)
    print(text_tokenizer.word_index)
    print()
    for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
        print('Sequence {} in x'.format(sample_i + 1))
        print('  Input:  {}'.format(sent))
        print('  Output: {}'.format(token_sent))

    translator_tests.test_pad(pad)

    # Pad Tokenized output
    test_pad = pad(text_tokenized)
    for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
        print('Sequence {} in x'.format(sample_i + 1))
        print('  Input:  {}'.format(np.array(token_sent)))
        print('  Output: {}'.format(pad_sent))

    preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = \
        preprocess(english_sentences, french_sentences)

    max_english_sequence_length = preproc_english_sentences.shape[1]
    max_french_sequence_length = preproc_french_sentences.shape[1]
    english_vocab_size = len(english_tokenizer.word_index)
    french_vocab_size = len(french_tokenizer.word_index)

    print('Data Preprocessed')
    print("Max English sentence length:", max_english_sequence_length)
    print("Max French sentence length:", max_french_sequence_length)
    print("English vocabulary size:", english_vocab_size)
    print("French vocabulary size:", french_vocab_size)

    translator_tests.test_model_final(model_final)


if __name__ == '__main__':
    # preprocessing_experiments()

    model, summary = main(use_check_point=True)
    # plot_training_results(summary)

    english_tokenizer, french_tokenizer = get_en_fr_tokenizers()
    test_model(model=model, x_tk=english_tokenizer, y_tk=french_tokenizer)
