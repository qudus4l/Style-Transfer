import torch


def calc_content_loss(gen_feat,orig_feat):
    #calculating the content loss 
    content_l=torch.mean((gen_feat-orig_feat)**2)#*0.5
    return content_l

def calc_style_loss(gen,style):
    #gram matrix for style and generated image
    batch_size,channel,height,width=gen.shape

    G=torch.mm(gen.view(channel,height*width),gen.view(channel,height*width).t())
    A=torch.mm(style.view(channel,height*width),style.view(channel,height*width).t())
        
    #Calculating the style loss
    style_l=torch.mean((G-A)**2)#/(4*channel*(height*width)**2)
    return style_l

def calculate_loss(gen_features, orig_feautes, style_featues, alpha, beta):
    style_loss=content_loss=0
    for gen,cont,style in zip(gen_features,orig_feautes,style_featues):
        content_loss+=calc_content_loss(gen,cont)
        style_loss+=calc_style_loss(gen,style)
    
    #calculating the total loss \\
    total_loss=alpha*content_loss + beta*style_loss 
    return total_loss