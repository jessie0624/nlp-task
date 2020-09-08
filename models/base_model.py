__all__ = []

import torch

class BaseModel(torch.nn.Module):
    r"""
    Base Pytorch model for all models
    """
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def fit(self, train_data, dev_data=None, **train_args):
        pass
    
    def predict(self, *args, **kwargs):
        raise NotImplementedError
    