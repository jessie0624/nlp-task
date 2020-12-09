'''
@description: 中文的分词或者分字，注意这里的输入是str型
'''
import jieba
from src.base import Unit
from src.tools.common import is_chinese_char

class CNTokenize(Unit):
    """Process unit for text tokenization."""

    def transform(self, input_: str) -> list:
        """
        Process input data from raw terms to list of tokens.

        :param input_: raw textual input.

        :return tokens: tokenized tokens as a list.
        """

        return jieba.lcut(input_)

class CNCharTokenize(Unit):
    """基于字符进行分隔"""
    def transform(self, input_: str) -> list:
        result = ""
        for ch in input_:
            if is_chinese_char(ch):
                result += ' ' + ch + ' '
            else:
                result += ch

        return result.split()