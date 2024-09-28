import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.transforms import transforms


class mydata(Dataset):

    def __init__(self, data_dir, label_dir, data_index):
        self.data_path = os.path.join(data_dir, data_index, "RGB")
        self.label_path = os.path.join(label_dir, data_index)
        self.pair_num = int(len(os.listdir(self.data_path)) / 2)
        self.transform = transforms.Compose([transforms.Resize([224, 224]),
                                             transforms.ToTensor()])

    def __getitem__(self, item):
        images = os.listdir(self.data_path)
        labels = os.listdir(self.label_path)
        pair_num = self.pair_num

        t1 = self.transform(Image.open(os.path.join(self.data_path, images[item])))
        t2 = self.transform(Image.open(os.path.join(self.data_path, images[item + pair_num])))
        label = np.load(os.path.join(self.label_path, labels[item]))
        label[label <= 1] = 0
        label[label > 1] = 1
        pair = [t1, t2]
        return pair, label

    def __len__(self):
        return self.pair_num


def get_VL_CMU_CD():
    images_path = "VL-CMU-CD/images"
    labels_path = "VL-CMU-CD/224labels"
    total = 0
    for i, index in zip(range(len(os.listdir(images_path))), os.listdir(images_path)):
        if i == 0:
            train_data = mydata(images_path, labels_path, index)
        if i != 0 and i < 122:
            train_data = train_data + mydata(images_path, labels_path, index)
        if i == 122:
            test_data = mydata(images_path, labels_path, index)
        if i > 122:
            test_data = test_data + mydata(images_path, labels_path, index)
    return train_data, test_data
