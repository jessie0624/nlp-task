'''
@description:DSSM 模型，适用于文本匹配，Adhoc检索
'''
import torch.nn.functional as F 
from src.params.param_table import ParamTable
from src.params.param import Param 
from src.base.base_model import BaseModel 

class DSSM(BaseModel):
    def __init__(self):
        super().__init__() 
    
    @classmethod
    def get_default_params(cls) -> ParamTable:
        params = super().get_default_params(with_multi_layer_perceptron=True)
        params.add(Param(name='vocab_size', vaule=419, desc='Size of Vocabulary.'))
        return params 
    
    @classmethod
    def get_default_padding_callback(cls):
        return None 
    
    def build(self):
        '''DSSM use Siamese architecture'''
        self.mlp_left = self._make_multi_perceptron_layer(
            self._params['vocab_size']
        )
        self.mlp_right = self._make_multi_perceptron_layer(
            self._params['vocab_size']
        )
        self.out = self._make_output_layer()
    
    def forward(self, inputs):
        input_left, input_right = inputs['input_left'], inputs['input_right']
        input_left = self.mlp_left(input_left)
        input_right = self.mlp_right(input_right)
        x = F.cosine_similarity(input_left, input_right)
        out = self.out(x.unsqueeze(dim=1))
        return out 
