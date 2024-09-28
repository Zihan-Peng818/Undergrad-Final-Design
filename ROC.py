import numpy

from unet.unet_model_change import UNet
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from dataset import get_VL_CMU_CD

from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt

device = 0
image_size = 224 * 224
batch_size = 1
model = UNet(n_channels=3, n_classes=2)
state_dict = torch.load('use/model/run_v3.pt')
model.load_state_dict(state_dict)
model.to(device)

train_data, test_data = get_VL_CMU_CD()
test_loader = DataLoader(test_data,
                         batch_size=1,
                         shuffle=True,
                         num_workers=0,
                         pin_memory=True,
                         drop_last=True)
a = numpy.array([])
b = numpy.array([])


def plot_ROC(y_label, y_pre):
    """
    Args:
        labels : ground truth
        preds : model prediction
        savepath : save path
    """

    fpr, tpr, thersholds1 = roc_curve(y_label, y_pre, pos_label=1)
    precision, recall, thresholds2 = precision_recall_curve(y_label, y_pre, pos_label=1)

    t = 0
    for i, value in enumerate(thersholds1):
        if 0.01 <= fpr[i] and t == 0:
            print("%f %f" % (fpr[i], tpr[i]))
            t += 1
            r1 = tpr[i]
        if 0.05 <= fpr[i] and t == 1:
            print("%f %f" % (fpr[i], tpr[i]))
            t += 1
            r2 = tpr[i]
        if 0.1 <= fpr[i] and t == 2:
            print("%f %f" % (fpr[i], tpr[i]))
            t += 1
            r3 = tpr[i]

    t = 0
    for i, value in enumerate(thresholds2):
        if r3 >= recall[i] and t == 0:
            print("%f %f %f" % (precision[i], recall[i], (2 * recall[i] * precision[i] / (recall[i] + precision[i]))))
            t += 1
            p1 = precision[i]
        if r2 >= recall[i] and t == 1:
            print("%f %f %f" % (precision[i], recall[i], (2 * recall[i] * precision[i] / (recall[i] + precision[i]))))
            t += 1
            p2 = precision[i]
        if r1 >= recall[i] and t == 2:
            print("%f %f %f" % (precision[i], recall[i], (2 * recall[i] * precision[i] / (recall[i] + precision[i]))))
            t += 1
            p3 = precision[i]

    roc_auc = auc(fpr, tpr)
    fig1 = plt.figure(1)
    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('ROC.png')
    fig2 = plt.figure(2)
    plt.plot(recall, precision, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
    plt.xlim([0.4, 1])
    plt.ylim([0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.legend(loc="lower right")
    plt.savefig('PR.png')


# with open("softmax.csv", "w", newline="") as f:
#     csv_writer = csv.writer(f)
#     name = ['group', 'value']
#     csv_writer.writerow(name)

with torch.no_grad():
    for counter, (instance, label) in enumerate(tqdm(test_loader)):
        t1 = instance[0].to(device)
        t2 = instance[1].to(device)
        mask = label.view(-1, image_size).to(device).cpu().numpy().astype(np.int32).squeeze()
        predict, t1_features, t2_features = model(t1, t2)
        predict = predict.view(batch_size, 2, image_size).squeeze()
        result = F.softmax(predict, dim=1).squeeze()
        result = result[1].unsqueeze(0).cpu().numpy().squeeze()
        a = np.hstack((a, mask))
        b = np.hstack((b, result))
        # if counter >= 239:
    plot_ROC(a, b)
    # break

    # z = torch.cat([mask, result], dim=0).T.cpu().numpy()
    # csv_writer.writerows(z)

    # f.close()
