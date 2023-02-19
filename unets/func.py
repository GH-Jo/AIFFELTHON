import segmentation_models_pytorch as smp
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import copy

_device = None
UNet_sc, UNet_de, UNet_sp = None, None, None
img_transformer = None


def model_init():
    global _device, img_transformer
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(_device)
    
    global UNet_sp, UNet_de, UNet_sc
    UNet_sp = smp.Unet(
    encoder_name="efficientnet-b7",
    encoder_weights="imagenet",
    in_channels=3,                  
    classes=2,                      
    )
    UNet_de = copy.deepcopy(UNet_sp)
    UNet_sc = copy.deepcopy(UNet_sp)
    
    UNet_sp.to(_device).eval()
    UNet_de.to(_device).eval()
    UNet_sc.to(_device).eval()
    
    UNet_sp.load_state_dict(torch.load('/home/lkj004124/unets/best_sp.pt')) #,map_location=_device)
    UNet_de.load_state_dict(torch.load('/home/lkj004124/unets/best_de.pt')) #,map_location=_device)
    UNet_sc.load_state_dict(torch.load('/home/lkj004124/unets/best_sc.pt'))#,map_location=_device)
    

    img_transformer = transforms.Compose(
        [transforms.Resize((512, 512)),
        transforms.ToTensor()]
    )
    return True


def predict(img): 
    # 1. output = model(input)
    global _device, UNet_sp, UNet_de, UNet_scm, img_transformer
    input = img_transformer(img).to(_device)
    input = torch.unsqueeze(input, 0)
    
    with torch.no_grad():
        UNet_sc.load_state_dict(torch.load('/home/lkj004124/unets/best_sc.pt'))
        mask_scratch = UNet_sc(input)[0]
        UNet_de.load_state_dict(torch.load('/home/lkj004124/unets//best_de.pt'))
        mask_dent = UNet_de(input)[0]
        UNet_sp.load_state_dict(torch.load('/home/lkj004124/unets/best_sp.pt'))
        mask_spacing = UNet_sp(input)[0]
    
    # 2. make img
    mask_spacing = torch.argmax(mask_spacing, dim=0).unsqueeze(-1)
    mask_dent = torch.argmax(mask_dent, dim=0).unsqueeze(-1)
    mask_scratch = torch.argmax(mask_scratch, dim=0).unsqueeze(-1)
    mask = torch.cat([mask_spacing, mask_dent, mask_scratch], axis=2) * 255
    input = (input[0].permute(1,2,0).cpu() + mask.clip(min=0, max=255).cpu()).numpy()
    input = input.astype(np.uint8)
    
    return input, {'scratch': bool(mask_spacing.sum()), 
            'dent': bool(mask_spacing.sum()), 
            'spacing': bool(mask_spacing.sum()) }


if __name__ == '__main__':
    model_init()
    predict(Image.open('/home/lkj004124/data2/spacing/test/images/20190207_314361549530754003.jpeg'))
    print('Good.')
