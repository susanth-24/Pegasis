import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class Encoder(nn.Module):
    '''
    This class return all the features extracted from the images, these features are extracted
    through a combination of convolutional layers and pooling operations in the DenseNet model,
    forward-> input image, example size (batch_size,1,480,640).
    '''
    def __init__(self):
        super(Encoder, self).__init__()       
        self.original_model = models.densenet169( pretrained=False )

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items(): features.append( v(features[-1]) )
        return features
    
