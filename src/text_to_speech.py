from gtts import gTTS

from Model import Model
from SamplePreprocessor import image_preprocess
from lines_main import infer_line
from translator import Translator

if __name__ == '__main__':
    model = Model(mustRestore=True, is_line=True)
    res = image_preprocess('../data/lines/4.png', is_lines=True)

    english_sentence = infer_line(model, res)

    translator = Translator()
    french_sentence = translator.translate(english_sentence)

    print('\n\n\n')
    print(f'Predicted english sentence: \t {english_sentence}')
    print('En-Fr Translator sentence: \t', french_sentence)

    tts_dir = '../text_to_speech_out_files/'
    target_lang = 'fr'

    tts = gTTS(english_sentence, lang='en')
    tts.save(tts_dir + 'english_sentence.mp3')

    tts = gTTS(french_sentence, lang='fr')
    tts.save(tts_dir + 'french_sentence.mp3')
