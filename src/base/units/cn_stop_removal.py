'''
@description: 移除中文停止词，输入是list型
'''
from src.base import Unit

class CNStopRemoval(Unit):
    def __init__(self, stopwords: list = []):
        self._stop = stopwords

    def transform(self, input_: list) -> list:
        return [token for token in input_ if token not in self._stop]

    @property
    def stopwords(self) -> list:
        """
        getter of stopwords
        :return:
        """
        return self._stop