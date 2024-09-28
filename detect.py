import os
import torch
from unet.unet_model import UNet
from torchvision.transforms import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np

device = 0
model = UNet(n_channels=3, n_classes=2)
state_dict = torch.load('use/model/run.pt')
model.load_state_dict(state_dict)
model.to(device)
transformer = transforms.Compose([transforms.Resize([224, 224]),
                                  transforms.ToTensor()])


def detect():
    images = os.listdir("use/input")
    t1 = Image.open(os.path.join('use/input', images[0]))
    t2 = Image.open(os.path.join('use/input', images[1]))
    h, w = t1.size[0], t1.size[1]
    t1 = transformer(t1).to(device)
    t2 = transformer(t2).to(device)
    t1_features = model(t1.unsqueeze(0)).view(1, 2, 224 * 224)
    t2_features = model(t2.unsqueeze(0)).view(1, 2, 224 * 224)
    predict = t1_features * t2_features
    result = F.softmax(predict, dim=1).max(dim=1).indices
    result = result.view(224, 224)
    im = Image.fromarray(np.uint8(result.cpu()))
    im = im.resize((h, w))
    im.save('use/output/result.jpg')
    os.remove(os.path.join('use/input', images[0]))
    os.remove(os.path.join('use/input', images[1]))


detect()
