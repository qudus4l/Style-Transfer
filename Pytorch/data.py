from PIL import Image
from torchvision import transforms

def load_image(img):
    image = Image.open(img)

