import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class Depth_Model(nn.Module):
    '''
    Trainable model,
    CUDA is advisable other wise it will take 15hr per epoch,
    '''
    def __init__(self):
        super(Depth_Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder( self.encoder(x) )