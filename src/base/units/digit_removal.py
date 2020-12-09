'''
@description: 移除数字，输入是list型
'''

from src.base import Unit
from src.tools.common import is_number

class DigitRemoval(Unit):
    """
    Remove digits from list of tokens
    """
    def transform(self, input_: list) -> list:
        return [token for token in input_ if not is_number(token)]
