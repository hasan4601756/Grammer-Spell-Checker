import re
import unicodedata
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import contractions

nltk.download('punkt')


class Preprocessor:
    def __init__(self, text: str):
        self.original_text = text
        self.text = text.strip().lower()

    @staticmethod
    def normalize_unicode(text: str) -> str:
        return unicodedata.normalize('NFKD', text)

    @staticmethod
    def remove_extra_spaces(text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def handle_contractions(text: str) -> str:
        return contractions.fix(text)

    @staticmethod
    def remove_punctuations(text: str) -> str:
        return re.sub(r"[^\w\s.]", ' ', text)
    
    @staticmethod
    def remove_non_keyboard_chars(text: str) -> str:
        return re.sub(r'[^\x00-\x7F]+', '', text)


    def preprocess(self,
                   normalize=True,
                   remove_punct=True,
                   remove_spaces=True,
                   remove_emojis=True,
                   expand_contractions=True) -> str:

        text = self.text

        if normalize:
            text = self.normalize_unicode(text)
        
        if remove_emojis:
            text = self.remove_non_keyboard_chars(text)

        if expand_contractions:
            text = self.handle_contractions(text)

        if remove_punct:
            text = self.remove_punctuations(text)

        if remove_spaces:
            text = self.remove_extra_spaces(text)

        return text

    def tokenize(self, **kwargs):
        processed_text = self.preprocess(**kwargs)
        return word_tokenize(processed_text)

    def sentence_tokenize(self, **kwargs):
        processed_text = self.preprocess(remove_punct=False, **kwargs)
        return sent_tokenize(processed_text)


if __name__ == '__main__':
    input_string = input("Enter the String: ")

    preprocessor = Preprocessor(input_string)

    clean_text = preprocessor.preprocess(expand_contractions=True)
    word_tokens = preprocessor.tokenize()
    sentence_tokens = preprocessor.sentence_tokenize()

    print(f"\nPreprocessed Text:\n{clean_text}")
    print(f"\nWord Tokens:\n{word_tokens}")
    print(f"\nSentence Tokens:\n{sentence_tokens}")