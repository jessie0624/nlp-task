'''
@description: 对英文单词进行词干化
'''
import nltk

from src.base import Unit

class Stemming(Unit):
    """Process unit for token stemming"""
    def __init__(self, stemmer='porter'):
        """stemmer可选`porter`或者`lancaster`"""
        self.stemmer = stemmer

    def transform(self, input_: list) -> list:
        if self.stemmer == 'porter':
            porter_stemmer = nltk.stem.PorterStemmer()
            return [porter_stemmer.stem(token) for token in input_]
        elif self.stemmer == 'lancaster' or self.stemmer == 'krovetz':
            lancaster_stemmer = nltk.stem.lancaster.LancasterStemmer()
            return [lancaster_stemmer.stem(token) for token in input_]
        else:
            raise ValueError(
                'Not supported supported stemmer type: {}'.format(
                    self.stemmer))


