from PIL import Image
import torch
from torchvision import transforms

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps:0")

def load_image(img):
    '''
    Load image from path
    :param img: path to image
    :return: image tensor
    '''
    image = Image.open(img)
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
        #transforms.Normalize(
        #    mean=[],  # RGB
        #    std=[])
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float)
