'''
@description: 英文去除停止词，输入是list
'''

import nltk
from src.base import Unit

class StopRemoval(Unit):
    def __init__(self, lang: str='english'):
        self._lang = lang
        self._stop = nltk.corpus.stopwords.words(self._lang)

    def transform(self, input_: list) -> list:
        return [token for token in input_ if token not in self._stop]

    @property
    def stopwords(self) -> list:
        return self._stop

    @stopwords.setter
    def stopwords(self, item: list):
        self._stop = item
