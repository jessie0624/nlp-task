from src.base import BaseCallback

class LambdaCallback(BaseCallback):
    def __init__(self,
                 on_batch=None):
        self._on_batch = on_batch

    def on_batch(self, x, y):
        if self._on_batch:
            self.on_batch(x, y)