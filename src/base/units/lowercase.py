'''
@description: 将字母小写，输入是list
'''
from src.base import Unit


class Lowercase(Unit):
    """Process unit for text lower case."""

    def transform(self, input_: list) -> list:
        return [token.lower() for token in input_]