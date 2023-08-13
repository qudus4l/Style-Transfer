import torch 
import torch.nn as nn
import torchvision.models as models

class style_transfer_VGG(nn.Module):
    def __init__(self):
        super(style_transfer_VGG, self).__init__()

        # VGG19 layers
        self.chosen_features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        # Store relevant features
        features = []

        # Go through layers
        for i, layer in enumerate(self.model):
            x = layer(x)

            # Store relevant features
            if str(i) in self.chosen_features:
                features.append(x)

        return features
