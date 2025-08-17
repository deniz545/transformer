import spacy
from nltk.stem.snowball import SnowballStemmer

class Lemmatization:
    def __init__(self, language="en"):
        if language == "english":
            self.nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
        elif language == "french":
            self.nlp = spacy.load("fr_core_news_sm", disable=["parser","ner"])
        else:
            raise ValueError("Language not supported")

    def transform(self, text: str) -> str:
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc])


class Stemming:
    def __init__(self, language):
        self.stemmer = SnowballStemmer(language)

    def transform(self, text: str) -> str:
        return " ".join([self.stemmer.stem(tok) for tok in text.split()])


class DataPreparation:
    def __init__(self, option):
        if option == "stemming":
            self.preparation_english = Stemming('english')
            self.preparation_french = Stemming('french')
        elif option == "lemmatization":
            self.preparation_english = Lemmatization('english')
            self.preparation_french = Lemmatization('french')
        else:
            raise ValueError("Option must be 'stemming' or 'lemmatization'")

    def process(self, text: str) -> str:
        return self.preparation.transform(text)
