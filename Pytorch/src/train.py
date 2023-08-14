import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.utils import save_image
from data import load_image, device
from model import style_transfer_VGG
from optimize import calculate_loss

model = style_transfer_VGG().to(device).eval()

def train_image(original_img, style_img):
    original_img = load_image(original_img)
    style_img = load_image(style_img)
    epoch = 1
    learning_rate = 0.001
    alpha = 1
    beta = 0.01
    generated = original_img.clone().requires_grad_(True)  # Initialize generated here
    optimizer = torch.optim.Adam([generated], lr=learning_rate)

    for e in range (epoch):
        #extracting the features of generated, content and the original required for calculating the loss
        gen_features=model(generated)
        orig_feautes=model(original_img)
        style_featues=model(style_img)

        #iterating over the activation of each layer and calculate the loss and add it to the content and the style loss
        total_loss=calculate_loss(gen_features, orig_feautes, style_featues, alpha, beta)
        #optimize the pixel values of the generated image and backpropagate the loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        #return the generated image
    return generated
