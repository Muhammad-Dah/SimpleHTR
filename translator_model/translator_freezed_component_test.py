import json
import os

import keras
import tensorflow as tf
from keras.layers import GRU, Dense, TimeDistributed, RepeatVector, Bidirectional, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.optimizers import Adam

from translator_model.translator_main import preprocess, load_data, plot_training_results, get_en_fr_tokenizers, \
    test_model


def freezed_encoder_model(input_shape=(137861, 15), output_sequence_length=21, english_vocab_size=200,
                          french_vocab_size=345):
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
    # Encoder
    encoder_layers = [Embedding(english_vocab_size, 128, input_length=input_shape[1], input_shape=input_shape[1:]),
                      Bidirectional(GRU(128)),
                      RepeatVector(output_sequence_length)]

    for layer in encoder_layers:
        layer.trainable = False
        model.add(layer)

    # Decoder
    model.add(Bidirectional(GRU(128, return_sequences=True)))
    model.add(TimeDistributed(Dense(512, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model


def freezed_decoder_model(input_shape=(137861, 15), output_sequence_length=21, english_vocab_size=200,
                          french_vocab_size=345):
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
    # Embedding + Encoder
    encoder_layers = [Embedding(english_vocab_size, 128, input_length=input_shape[1], input_shape=input_shape[1:]),
                      Bidirectional(GRU(128)),
                      RepeatVector(output_sequence_length)]

    for layer in encoder_layers:
        model.add(layer)

    # Decoder
    decoder_layers = [Bidirectional(GRU(128, return_sequences=True, trainable=False)),
                      TimeDistributed(Dense(512, activation='relu', trainable=False)),
                      Dropout(0.5, trainable=False),
                      TimeDistributed(Dense(french_vocab_size, activation='softmax', trainable=False))]

    for layer in decoder_layers:
        model.add(layer)

    # Decoder
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model


def train_freezed_model(x, y, x_tk, y_tk, freezed_comp='encoder', use_check_point=False):
    """
    train the translation model and save checkpoint
    :param x: Preprocessed English data
    :param y: Preprocessed French data
    :param x_tk: English tokenizer
    :param y_tk: French tokenizer
    :param freezed_comp: which component in the model is freezed
    :param use_check_point: whether save model weights to file or not
    """
    mode_constructor = freezed_encoder_model if freezed_comp == 'encoder' else freezed_decoder_model
    model = mode_constructor(x.shape, y.shape[1],
                             len(x_tk.word_index) + 1,
                             len(y_tk.word_index) + 1)
    model.summary()

    checkpoint_path = f"freezed_translator_checkpoint_dir/freezed_{freezed_comp}/cp.ckpt"
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

        return model, summary.history


def main_model(freezed_comp, use_check_point):
    # Load English data
    english_sentences = load_data('data/small_vocab_en')
    # Load French data
    french_sentences = load_data('data/small_vocab_fr')
    preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = \
        preprocess(english_sentences, french_sentences)

    model, summary = train_freezed_model(preproc_english_sentences, preproc_french_sentences, english_tokenizer,
                                         french_tokenizer, freezed_comp, use_check_point)
    return model, summary


if __name__ == '__main__':
    model, summary = main_model(freezed_comp='encoder', use_check_point=True)
    plot_training_results(summary)

    english_tokenizer, french_tokenizer = get_en_fr_tokenizers()
    test_model(model=model, x_tk=english_tokenizer, y_tk=french_tokenizer)
