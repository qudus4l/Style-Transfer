from PIL import Image
import torch
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(img):
    '''
    Load image from path
    :param img: path to image
    :return: image tensor
    '''
    image = Image.open(img)
    transform = transforms.Compose([
        transforms.Resize(356,356),
        transforms.ToTensor()
        #transforms.Normalize(
        #    mean=[],  # RGB
        #    std=[])
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)


