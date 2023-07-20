import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()        
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB( self.convB( self.convA( torch.cat([up_x, concat_with], dim=1)  ) )  )

class Decoder(nn.Module):
    '''
    This class is for decoding the previously encoded features from densenet
    Residual Block is also being used for fine details and features in the output
    forward-> takes input from features at index: 3,4,6,8,12.
    '''
    def __init__(self, num_features=1664, decoder_width = 1.0):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)

        self.up1 = UpSample(skip_input=features//1 + 256, output_features=features//2)
        self.up2 = UpSample(skip_input=features//2 + 128,  output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 + 64,  output_features=features//8)
        self.up4 = UpSample(skip_input=features//8 + 64,  output_features=features//16)

        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)
        self.residual_block1 = self._make_residual_block(features // 1, features // 1)
        self.residual_block2 = self._make_residual_block(features // 2, features // 2)
        self.residual_block3 = self._make_residual_block(features // 4, features // 4)
        self.residual_block4 = self._make_residual_block(features // 8, features // 8)

    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, features):
        x_0, x_1, x_2, x_3, x_4 = features[3], features[4], features[6], features[8], features[12]
        d0 = self.conv2(F.relu(x_4))
        d0 = self.residual_block1(d0)  
        d1 = self.up1(d0, x_3) 
        d1 = self.residual_block2(d1)
        d2 = self.up2(d1, x_2)
        d2 = self.residual_block3(d2)
        d3 = self.up3(d2, x_1)
        d3 = self.residual_block4(d3)
        d4 = self.up4(d3, x_0)
        return self.conv3(d4)
    
    