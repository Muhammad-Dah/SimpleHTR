from autocorrect import Speller


class SpellingCorrector:
    def __init__(self):
        self.spell = Speller()

    def get_correct(self, to_correct):
        sentence_list = to_correct
        if isinstance(to_correct, str):
            sentence_list = [to_correct]
        if isinstance(to_correct, tuple):
            sentence_list = list(to_correct)

        concatenate_sting = ' '.join(w for w in sentence_list)
        corrected = self.spell(concatenate_sting)
        return corrected


if __name__ == '__main__':
    w = ("machne", "nw")
    print(SpellingCorrector().get_correct(w))

    w = "haveplease"
    print(SpellingCorrector().get_correct(w))

    w = ["machne", "learnng"]
    print(SpellingCorrector().get_correct(w))
