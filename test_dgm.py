import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.models as models
from guidance.resnet_gm import ResNet34
from matplotlib import pyplot as plt
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


model = ResNet34().cuda()
state_dict = torch.load('res34_model_best.pth.tar')['state_dict']
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

model.eval()


with torch.no_grad():
    image = Image.open('line_render.png').convert('RGB')
    image = transform(image).unsqueeze(0)
    print(image.shape)
    image = image.cuda()

    _, gms = model(image)
    
    # gms = gms.repeat(1,3,1,1)
    # gms = gms * 255.
    # gms = gms.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    
    # gms represent dgm based attention maps
    # save gms[0], gms[1] ..etc to visualize attention maps
    
    plt.imshow(gms.permute(0, 2, 3, 1).cpu().numpy()[0])
    plt.show()
    plt.savefig('lion_whire_attention_map.png')

