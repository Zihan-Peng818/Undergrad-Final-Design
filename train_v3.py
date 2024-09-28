from dataset import get_VL_CMU_CD
from unet.unet_model_change import UNet
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import os
from metrics import get_pre_recall_F1
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
model = UNet(n_channels=3, n_classes=2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.to(device)
print("using device:", device)

train_data, test_data = get_VL_CMU_CD()
batch_size = 8
EPOCH = 1500

image_size = 224 * 224

train_loader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0,
                          pin_memory=True,
                          drop_last=True)
test_loader = DataLoader(test_data,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0,
                         pin_memory=True,
                         drop_last=True)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.0003,
                             weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=len(train_loader),
                                                       eta_min=0,
                                                       last_epoch=-1)

# CE_loss = torch.nn.CrossEntropyLoss()


def train():
    train_loss = 0.0
    for counter, (instance, label) in enumerate(tqdm(train_loader)):
        t1 = instance[0].to(device)
        t2 = instance[1].to(device)
        mask = label.view(-1, image_size).to(device)
        prop = mask.sum()/(mask.shape[0]*mask.shape[1])
        weight = torch.tensor([prop, 1-prop]).to(device)
        CE_loss = torch.nn.CrossEntropyLoss(weight=weight.to(torch.float32))
        predict, t1_features, t2_features = model(t1, t2)
        predict = predict.view(batch_size, 2, image_size)
        t1_features = t1_features.view(batch_size, 256, image_size)
        t2_features = t2_features.view(batch_size, 256, image_size)
        logit = 0.5 + 0.5 * F.cosine_similarity(t1_features, t2_features, dim=1).unsqueeze(1)
        logit = torch.cat([1 - logit, logit], dim=1)
        loss1 = CE_loss(predict, mask.long())
        loss2 = CE_loss(logit, mask.long())
        loss = loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= (counter + 1)
    print("Train Epoch:{}\t Loss:{}\t".format(epoch + 1, train_loss))
    return train_loss


def test():
    with torch.no_grad():
        p = 0.0
        r = 0.0
        f = 0.0
        for counter, (instance, label) in enumerate(tqdm(test_loader)):
            t1 = instance[0].to(device)
            t2 = instance[1].to(device)
            mask = label.view(-1, image_size).to(device)
            predict, t1_features, t2_features = model(t1, t2)
            predict = predict.view(batch_size, 2, image_size)
            result = F.softmax(predict, dim=1).max(dim=1).indices
            precision, recall, F1 = get_pre_recall_F1(result.cpu(), mask.cpu())
            p += precision
            r += recall
            f += F1
        p = p / (counter + 1)
        r = r / (counter + 1)
        f = f / (counter + 1)
        print("precision:{}\t recall:{}\t F1:{}".format(p, r, f))
    return p, r, f


EPO = []
LOS = []
PRE = []
REC = []
F1 = []
for epoch in range(EPOCH):
    trl = train()
    LOS = np.append(LOS, trl)

    tep, ter, tef = test()
    PRE = np.append(PRE, tep)
    REC = np.append(REC, ter)
    F1 = np.append(F1, tef)

    EPO = np.append(EPO, epoch + 1)

torch.save(model.state_dict(), "use/model/run_v4.pt")

from matplotlib import pyplot as plt

fig1 = plt.figure(1)
plt.plot(EPO, LOS)
with open("exp/v4/loss_v4.txt", 'w') as train_los:
    train_los.write(str(LOS))
plt.xlabel('EPOCH')
plt.ylabel('LOSS')
plt.savefig('exp/v4/loss_v4.png')

fig2 = plt.figure(2)
plt.plot(EPO, PRE)
with open("exp/v4/pre_v4.txt", 'w') as test_pre:
    test_pre.write(str(PRE))
plt.xlabel('EPOCH')
plt.ylabel('PRE')
plt.savefig('exp/v4/pre_v4.png')

fig3 = plt.figure(3)
plt.plot(EPO, REC)
with open("exp/v4/rec_v4.txt", 'w') as test_rec:
    test_rec.write(str(REC))
plt.xlabel('EPOCH')
plt.ylabel('REC')
plt.savefig('exp/v4/rec_v4.png')

fig4 = plt.figure(4)
plt.plot(EPO, F1)
with open("exp/v4/f1_v4.txt", 'w') as test_f1:
    test_f1.write(str(F1))
plt.xlabel('EPOCH')
plt.ylabel('F1')
plt.savefig('exp/v4/f1_v4.png')

fig5 = plt.figure(5)
plt.plot(REC, PRE)
plt.xlabel('REC')
plt.ylabel('PRE')
plt.savefig('exp/v4/rp_v4.png')
