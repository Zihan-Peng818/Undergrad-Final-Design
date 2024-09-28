from flask import Flask, request
from flask_cors import CORS
import os
import torch
from unet.unet_model_change import UNet
from torchvision.transforms import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
import time
count = 0
app = Flask(__name__,
            static_folder='./use',
            static_url_path='',
            )
CORS(app)


@app.route('/', methods=["post"])
def getdata():
    while len(os.listdir('use/output')) != 0:
        os.remove(os.path.join('use/output/', os.listdir('use/output')[0]))
    print(request.files)
    request.files.get('file').save('use/input/' + request.files.get('file').filename)
    img_num = len(os.listdir("use/input"))
    if img_num == 2:
        t = detect()
        path = 'http://localhost:5000/output/' + str(t) + '.jpg'
        return path
    else:
        return ""


def detect():
    image_size = 224*224
    t = time.time()
    images = os.listdir("use/input")
    t1 = Image.open(os.path.join('use/input', images[0]))
    t2 = Image.open(os.path.join('use/input', images[1]))
    h, w = t1.size[0], t1.size[1]
    t1 = transformer(t1).to(device).unsqueeze(0)
    t2 = transformer(t2).to(device).unsqueeze(0)
    predict, t1_features, t2_features = model(t1, t2)
    predict = predict.view(1, 2, image_size)
    result = F.softmax(predict, dim=1).max(dim=1).indices
    result = result.view(224, 224) * 255
    im = Image.fromarray(np.uint8(result.cpu()))
    im = im.resize((h, w))
    im.save('use/output/' + str(t) + '.jpg')
    os.remove(os.path.join('use/input', images[0]))
    os.remove(os.path.join('use/input', images[1]))
    return t


if __name__ == '__main__':
    device = 0
    model = UNet(n_channels=3, n_classes=2)
    state_dict = torch.load('use/model/run_v3.pt')
    model.load_state_dict(state_dict)
    model.to(device)
    transformer = transforms.Compose([transforms.Resize([224, 224]),
                                      transforms.ToTensor()])
    app.run()
