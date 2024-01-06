# Description: Unofficial implementation of STAN architecture: https://arxiv.org/ftp/arxiv/papers/2002/2002.01034.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Class to build Encoder block.
    :param in_channels: int, the number of channels of the input tensor.
    :param out_channels: int, the number of channels of the output tensor.
    :param stem: bool, if True, the block is the first block of the encoder.
    :param middle: bool, if True, the block is the bottleneck block of the model.
    """
    def __init__(self, in_channels, out_channels, stem=False, middle=False):
        super().__init__()

        if stem:
            self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
            self.conv3x3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')
        else:
            self.conv3x3_1 = nn.Conv2d(in_channels//2, out_channels, kernel_size=3, padding='same')
            self.conv3x3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')

        self.conv5x5_1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding='same')
        self.conv5x5_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')

        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')
        self.conv1x1_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')

        self.relu = nn.ReLU(replace=True)

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.middle = middle

    def forward(self, x_branche1, x_branche2):

        x1_3 = self.conv3x3_1(x_branche2) # to concat with x5_1
        x1_3 = self.relu(x1_3)
        x1_3 = self.conv3x3_2(x1_3)
        x1_3 = self.relu(x1_3)
        skip2 = x1_3
        x1_3_down = self.down(skip2)

        x1_5 = self.conv5x5_1(x_branche1)
        x1_5 = self.relu(x1_5)
        x1_5 = self.conv5x5_2(x1_5)
        x1_5 = self.relu(x1_5)

        x1_1 = self.conv1x1_1(x_branche1)
        x1_1 = self.relu(x1_1)
        x1_1 = self.conv1x1_2(x1_1)
        x1_1 = self.relu(x1_1)

        if self.middle:
            return torch.cat((x1_5, x1_1, x1_3), 1)

        x5_1 = torch.cat((x1_5, x1_1), 1) # for skip connection, to concat with x1_3
        
        x5_1_down = self.down(x5_1) # for moving to the next encoder

        skip1 = torch.cat((x1_3, x5_1), 1) # for skip connection, to concat with x3_3

        return skip1, skip2, x1_3_down, x5_1_down

class Decoder(nn.Module):
    """
    Class to build Decoder block.
    :param filters: list[int], the number of filters for each layer.
    """
    def __init__(self, filters=[32, 64, 128, 256, 512]):
        super().__init__()
        self.filters = filters
        # Decoder 1
        self.up1 = nn.ConvTranspose2d(in_channels=self.filters[4]*3, out_channels=self.filters[4], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3x3_1_dec1 = nn.Conv2d(self.filters[3]*3, self.filters[3]*3, kernel_size=3, padding='same')
        self.conv3x3_2_dec1 = nn.Conv2d(self.filters[4]*3, self.filters[3], kernel_size=3, padding='same')

        # Decoder 2
        self.up2 = nn.ConvTranspose2d(in_channels=self.filters[3], out_channels=self.filters[2], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3x3_1_dec2 = nn.Conv2d(self.filters[3], self.filters[3], kernel_size=3, padding='same')
        self.conv3x3_2_dec2 = nn.Conv2d(self.filters[2]*5, self.filters[2], kernel_size=3, padding='same')

        # Decoder 3
        self.up3 = nn.ConvTranspose2d(in_channels=self.filters[2], out_channels=self.filters[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3x3_1_dec3 = nn.Conv2d(self.filters[1]*2, self.filters[1], kernel_size=3, padding='same')
        self.conv3x3_2_dec3 = nn.Conv2d(self.filters[1]*4, self.filters[1], kernel_size=3, padding='same')

        # Decoder 4
        self.up4 = nn.ConvTranspose2d(in_channels=self.filters[1], out_channels=self.filters[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3x3_1_dec4 = nn.Conv2d(self.filters[0]*2, self.filters[0], kernel_size=3, padding='same')
        self.conv3x3_2_dec4 = nn.Conv2d(self.filters[0]*4, 32, kernel_size=3, padding='same')

        self.relu = nn.ReLU(replace=True)

    def forward(self, x, skip_connections):
        
        self.skip2_enc4 = skip_connections[0]
        self.skip1_enc4 = skip_connections[1]

        self.skip2_enc3 = skip_connections[2]
        self.skip1_enc3 = skip_connections[3]

        self.skip2_enc2 = skip_connections[4]
        self.skip1_enc2 = skip_connections[5]

        self.skip2_enc1 = skip_connections[6]
        self.skip1_enc1 = skip_connections[7]

       # Decoder 1
        x = self.up1(x)
        x = torch.cat((x, self.skip2_enc4), 1)
        x = self.conv3x3_1_dec1(x)
        x = self.relu(x)
        x = torch.cat((x, self.skip1_enc4), 1)
        x = self.conv3x3_2_dec1(x)
        x = self.relu(x)

        # Decoder 2
        x = self.up2(x)
        x = torch.cat((x, self.skip2_enc3), 1)
        x = self.conv3x3_1_dec2(x)
        x = self.relu(x)
        x = torch.cat((x, self.skip1_enc3), 1)
        x = self.conv3x3_2_dec2(x)
        x = self.relu(x)

        # Decoder 3
        x = self.up3(x)
        x = torch.cat((x, self.skip2_enc2), 1)
        x = self.conv3x3_1_dec3(x)
        x = self.relu(x)
        x = torch.cat((x, self.skip1_enc2), 1)
        x = self.conv3x3_2_dec3(x)
        x = self.relu(x)

        # Decoder 4
        x = self.up4(x)
        x = torch.cat((x, self.skip2_enc1), 1)
        x = self.conv3x3_1_dec4(x)
        x = self.relu(x)
        x = torch.cat((x, self.skip1_enc1), 1)
        x = self.conv3x3_2_dec4(x)
        x = self.relu(x)

        return x

class STAN(nn.Module):
    """
    Class to build STAN architecture: https://arxiv.org/ftp/arxiv/papers/2002/2002.01034.pdf.
    :param in_channels: int, the number of channels of the input image, default is 3
    :param n_classes: int, the number of classes of the segmentation task, default is 1
    :param filters: list[int], the number of filters for each layer, default is [32, 64, 128, 256, 512]
    """

    def __init__(self, in_channels=3, n_classes=1):
        super(STAN, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.filters = [32, 64, 128, 256, 512]


        # Encoder
        self.enc1 = Encoder(in_channels, self.filters[0], stem=True)
        self.enc2 = Encoder(self.filters[1], self.filters[1])
        self.enc3 = Encoder(self.filters[2], self.filters[2])
        self.enc4 = Encoder(self.filters[3], self.filters[3])

        # Bottleneck (middle)
        self.middle = Encoder(self.filters[4], self.filters[4], middle=True)

        # Decoder
        self.decoder = Decoder(filters=self.filters)

        self.out_conv = nn.Conv2d(32, self.n_classes, kernel_size=1, padding='same')

        

    def forward(self, x):
        skip1_enc1, skip2_enc1, x3_1_down, x5_1_down = self.enc1(x, x)
        skip1_enc2, skip2_enc2, x3_1_down, x5_1_down = self.enc2(x5_1_down, x3_1_down)
        skip1_enc3, skip2_enc3, x3_1_down, x5_1_down = self.enc3(x5_1_down, x3_1_down)
        skip1_enc4, skip2_enc4, x3_1_down, x5_1_down = self.enc4(x5_1_down, x3_1_down)

        x = self.middle(x5_1_down, x3_1_down) # torch.Size([1, 1536, 16, 16])
        
        # Decoder
        x = self.decoder(x, [skip2_enc4, skip1_enc4, skip2_enc3, skip1_enc3, skip2_enc2, skip1_enc2, skip2_enc1, skip1_enc1])

        x = self.out_conv(x)

        return  x
       
if __name__ == "__main__":
    model = STAN()
    # print(model)
    x = torch.randn((1, 3, 256, 256))
    print(model(x).shape)