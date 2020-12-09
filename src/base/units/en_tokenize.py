'''
@description: 用于英文分词，输入是str
'''
import nltk
from src.base import Unit

class Tokenize(Unit):
    """Process unit for english text tokenization"""
    def transform(self, input_: str) -> list:
        return nltk.word_tokenize(input_)