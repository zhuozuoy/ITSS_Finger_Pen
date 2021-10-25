import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvUnit, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, img):
        return self.conv_layer(img)


class SimpleCNN(nn.Module):
    def __init__(self, NUM_CLASS):
        super(SimpleCNN, self).__init__()
        self.cnn_layers = nn.Sequential(
            # Input Tensor Shape: [batch_size, 1, 28, 28]
            # Output Tensor Shape: [batch_size, 16, 14, 14]
            ConvUnit(1, 16, 5, 1, 2),
            # Input Tensor Shape: [batch_size, 16, 14, 14]
            # Output Tensor Shape: [batch_size, 32, 7, 7]
            ConvUnit(16, 32, 5, 1, 2),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(7 * 7 * 32, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, NUM_CLASS)
        )

    def forward(self, x):
        output = self.cnn_layers(x)
        output = output.view(output.size(0), -1)
        output = self.fc_layer(output)
        return output

    # def init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=self.init_std)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.BatchNorm2d):
    #         module.weight.data.fill_(1.0)
    #         module.bias.data.zero_()
