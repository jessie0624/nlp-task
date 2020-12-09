import abc

class BaseCallback(abc.ABC):
    """
    The base callback class
    """
    def on_batch(self, x, y):
        """
        callback during iter batch.
        """