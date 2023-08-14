import torch 
import torch.nn as nn
import torchvision.models as models

model=models.vgg19(pretrained=True).features

class style_transfer_VGG(nn.Module):
    def __init__(self):
        super(style_transfer_VGG,self).__init__()
        #Here we will use the following layers and make an array of their indices
        # 0: block1_conv1
        # 5: block2_conv1
        # 10: block3_conv1
        # 19: block4_conv1
        # 28: block5_conv1
        self.req_features= ['0','5','10','19','28'] 
        self.model=models.vgg19(pretrained=True).features[:29] #model will contain the first 29 layers
    
    def forward(self,x):
        # Store relevant features
        features=[]
        # Go through layers
        for layer_num,layer in enumerate(self.model):
            x=layer(x)
            # Store relevant features
            if (str(layer_num) in self.req_features):
                features.append(x)
                
        return features
