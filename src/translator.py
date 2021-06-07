import numpy as np
import tensorflow as tf
from deep_translator import GoogleTranslator
from keras.preprocessing.sequence import pad_sequences

from translator_model.translator_main import model_final, get_en_fr_tokenizers


def google_translate(text: str, source='auto', target='fr'):
    return GoogleTranslator(source=source, target=target).translate(text)


class Translator:
    def __init__(self):
        # Load English data

        translator_dir = '../translator_model'
        checkpoint_dir = translator_dir + '/translator_checkpoint_dir'

        self.english_tokenizer, self.french_tokenizer = get_en_fr_tokenizers(checkpoint_dir + '/tokenizers')
        self.model = model_final()

        # if model must be restored (for inference), there must be a snapshot
        latest_cp = tf.train.latest_checkpoint(checkpoint_dir)
        if not latest_cp:
            raise Exception('No saved translator model found in: ' + checkpoint_dir)

        self.model.load_weights(latest_cp)

    def translate(self, sentences: str or list, x_max_len=15):
        if isinstance(sentences, str):
            sentences = [sentences]
        y_id_to_word = {value: key for key, value in self.french_tokenizer.word_index.items()}
        y_id_to_word[0] = '<PAD>'

        preprocess_sentences = [''] * len(sentences)
        predictions = [''] * len(sentences)
        num_tokens = len(self.english_tokenizer.word_index) + 1
        for i, sentence in enumerate(sentences):
            tokenized_sentence = [self.english_tokenizer.word_index.get(w, num_tokens) for w in sentence.split()]
            preprocess_sentence = pad_sequences([tokenized_sentence], maxlen=x_max_len, padding='post')
            preprocess_sentences[i] = (i, sentence, preprocess_sentence)

        for (i, sentence, preprocess_sentence) in preprocess_sentences:
            if num_tokens in preprocess_sentence:
                # just in case our model have no idea what the sentence is
                predictions[i] = google_translate(sentence)
            else:
                sentence_pred = self.model.predict(preprocess_sentence, 1)
                sentence_pred = ' '.join([y_id_to_word[np.argmax(x)] for x in sentence_pred[0]]).replace('<PAD>', '')
                predictions[i] = sentence_pred

        return predictions[0] if len(predictions) == 1 else predictions


if __name__ == '__main__':
    # input languages
    translator = Translator()
    print('Our translator: \t', translator.translate('i like peaches'))
    print('Google translator: \t', google_translate('i like peaches'))
