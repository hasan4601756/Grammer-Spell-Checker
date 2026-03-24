import re
import unicodedata
import nltk
from nltk.tokenize import word_tokenize

class Preprocessor:
    def __init__(self, text:str):
        self.text = text.strip().lower()

    def normalize_unicode(self):
        self.text = unicodedata.normalize('NFKD', self.text)

    def remove_extra_spaces(self):
        # self.text = re.sub(r'^\s+|\s+$', '', self.text)
        self.text = re.sub(r'\s+', ' ', self.text)
    
    def remove_punctuations(self):
        self.text = re.sub(r'[^\w\s.\']', ' ', self.text)
    
    def preprocess(self) -> str:
        self.normalize_unicode()
        self.remove_punctuations()
        self.remove_extra_spaces()
        return self.text
    
    def tokenize(self):
        self.normalize_unicode()
        self.remove_punctuations()
        self.remove_extra_spaces()
        return word_tokenize(self.text)
    
    def sent_tokenize(self):
        self.normalize_unicode()
        self.remove_punctuations()
        self.remove_extra_spaces()
        return nltk.sent_tokenize(self.text)

if __name__ == '__main__':
    input_string = input("Enter the String: ")
    preprocessor = Preprocessor(input_string)
    preprocessed_string = preprocessor.preprocess()
    tokenized = preprocessor.tokenize()
    sent_tokenized = preprocessor.sent_tokenize()
    print(f"Preprocessed Text: \n{preprocessed_string}")
    print(f"Word Tokens: \n{tokenized}")
    print(f"Sentence Tokens: \n{sent_tokenized}")