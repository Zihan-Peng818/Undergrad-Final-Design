import torch
import numpy as np
import shutil


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def get_pre_recall_F1(result, target):
    result = result.numpy()
    target = target.numpy()
    TP = (result * target).sum()
    precision = TP / result.sum()
    precision = np.nan_to_num(precision, nan=0)
    recall = TP / target.sum()
    recall = np.nan_to_num(recall, nan=0)
    F1 = 2 * precision * recall / (precision + recall)
    F1 = np.nan_to_num(F1, nan=0)
    return precision, recall, F1

