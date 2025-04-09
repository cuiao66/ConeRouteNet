from timm import create_model
from gc_vit import GCViT
from layers import GC_ViTBackBone
import torch

if __name__ == "__main__":


    model = GC_ViTBackBone() # create_model('gcvit_xtiny', pretrained=False)
    
    # model.load_state_dict(torch.load("/root/GCVit/weight/gcvit_xtiny_224_nvidia-274b92b7.pth"))
    model.train()
    fake_image = torch.ones(1, 3, 224, 224)
    pred = model(fake_image)
    pass