import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.utils import save_image
from data import load_image, device



class style_transfer_VGG(nn.Module):
    def __init__(self):
        super(style_transfer_VGG, self).__init__()
        # The first number x in convx_y gets added by 1 after it has gone
        # through a maxpool, and the second y if we have several conv layers
        # in between a max pool. These strings (0, 5, 10, ..) then correspond
        # to conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 mentioned in NST paper
        self.chosen_features = ["0", "5", "10", "19", "28"]

        # We don't need to run anything further than conv5_1 (the 28th module in vgg)
        # Since remember, we dont actually care about the output of VGG: the only thing
        # that is modified is the generated image (i.e, the input).
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        # Store relevant features
        features = []

        # Go through each layer in model, if the layer is in the chosen_features,
        # store it in features. At the end we'll just return all the activations
        # for the specific layers we have in chosen_features
        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features

model = style_transfer_VGG().to(device).eval()

def train_image(original_img, style_img):
    original_img = load_image(original_img)
    style_img = load_image(style_img)
    total_steps = 6000
    learning_rate = 0.001
    alpha = 1
    beta = 0.01
    generated = original_img.clone().requires_grad_(True)  # Initialize generated here
    optimizer = torch.optim.Adam([generated], lr=learning_rate)

    for e in range(total_steps):
        generated_features = model(generated)
        original_img_features = model(original_img)
        style_features = model(style_img)

        style_loss = original_loss = 0

        for gen_feature, orig_feature, style_feature in zip(
                generated_features, original_img_features, style_features):

            batch_size, channel, height, width = gen_feature.shape
            original_loss += torch.mean((gen_feature - orig_feature) ** 2)

            # Compute Gram Matrix of generated
            G = gen_feature.view(channel, height * width).mm(
                gen_feature.view(channel, height * width).t())
            # Compute Gram Matrix of style
            A = style_feature.view(channel, height * width).mm(
                style_feature.view(channel, height * width).t())
            style_loss += torch.mean((G - A) ** 2)

        total_loss = alpha * original_loss + beta * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    return generated

#        if e % 100 == 0:
 #           print(f"Epoch {e} Total loss: {total_loss.item()}")
  #          save_image(generated, f"generated_{e}.png")
