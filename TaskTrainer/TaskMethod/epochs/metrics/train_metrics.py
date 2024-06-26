import numpy as np
from sklearn import metrics

from utils.BOX.evaluation_metrics3D import metrics_3d


def AUC_score(SR, GT, threshold=0.5):
    # SR = SR.numpy()
    GT = GT.ravel()  # we want to make them into vectors
    SR = SR.ravel()  # .detach()
    # fpr, tpr, _ = metrics.roc_curve(GT, SR)
    # fpr, tpr, _ = metrics.roc_curve(SR, GT)
    # roc_auc = metrics.auc(fpr, tpr)
    # roc_auc = metrics()
    roc_auc = metrics.roc_auc_score(GT, SR)
    return roc_auc


def threshold(image):
    image[image >= 100] = 255
    image[image < 100] = 0
    return image


def numeric_score(pred, gt):
    FP = np.float(np.sum((pred == 255) & (gt == 0)))
    FN = np.float(np.sum((pred == 0) & (gt == 255)))
    TP = np.float(np.sum((pred == 255) & (gt == 255)))
    TN = np.float(np.sum((pred == 0) & (gt == 0)))
    return FP, FN, TP, TN


def metrics1(pred, label, batch_size):
    # pred = torch.argmax(pred, dim=1) # for CE Loss series
    outputs = (pred.data.cpu().numpy() * 255).astype(np.uint8)
    labels = (label.data.cpu().numpy() * 255).astype(np.uint8)
    outputs = outputs.squeeze(1)  # for MSELoss()
    labels = labels.squeeze(1)  # for MSELoss()
    outputs = threshold(outputs)  # for MSELoss()

    Acc, SEn = 0., 0.
    for i in range(batch_size):
        img = outputs[i, :, :]
        gt = labels[i, :, :]
        acc, sen = get_acc(img, gt)
        Acc += acc
        SEn += sen
    return Acc, SEn


def metrics3dmse(pred, label, batch_size):
    outputs = (pred.data.cpu().numpy() * 255).astype(np.uint8)
    labels = (label.data.cpu().numpy() * 255).astype(np.uint8)
    outputs = outputs.squeeze(1)  # for MSELoss()
    labels = labels.squeeze(1)  # for MSELoss()
    outputs = threshold(outputs)  # for MSELoss()

    tp, fn, fp, IoU = 0, 0, 0, 0
    for i in range(batch_size):
        img = outputs[i, :, :, :]
        gt = labels[i, :, :, :]
        tpr, fnr, fpr, iou = metrics_3d(img, gt)
        # dcr = Dice(img, gt)
        tp += tpr
        fn += fnr
        fp += fpr
        IoU += iou
    return tp, fn, fp, IoU


def metrics3d(pred, label, batch_size):
    # pred = torch.argmax(pred, dim=1)  # for CE loss series
    # print((pred.data.cpu().numpy() * 255).astype(np.uint8))
    # print('1',pred.max(), pred.min())
    # print(pred.shape)

    try:
        pred = (pred.data.cpu().numpy() * 255).astype(np.uint8)
        label = (label.data.cpu().numpy() * 255).astype(np.uint8)
    except RuntimeError as e:
        print(f"CUDA error: {e}")

    # print(pred.max(),pred.min())
    # input("wait2..")

    # label = label * 255#
    # auc = AUC_score(pred, label)
    # print(pred.max(), pred.min())
    # outputs = outputs.squeeze(1)  # for MSELoss()
    # labels = labels.squeeze(1)  # for MSELoss()
    # outputs = threshold(outputs)  # for MSELoss()

    auc, acc, sen, spe, iou, dsc, pre, f1 = 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(batch_size):
        img = pred[i, :, :, :, :]
        gt = label[i, :, :, :, :]
        # AUC = AUC_score(img/255, gt/255)
        # print(img.shape, gt.shape)
        ACC, SEN, SPE, IOU, DSC, PRE = metrics_3d(img[-1], gt[-1])  # TODO: 这块可能有问题
        # DCR = Dice(img, gt)
        # auc += AUC
        acc += ACC
        sen += SEN
        spe += SPE
        iou += IOU
        dsc += DSC
        pre += PRE
        # f1 += F1
    return acc / batch_size, sen / batch_size, spe / batch_size, iou / batch_size, dsc / batch_size, pre / batch_size


def get_acc(image, label):
    image = threshold(image)
    FP, FN, TP, TN = numeric_score(image, label)
    acc = (TP + TN) / (TP + FN + TN + FP + 1e-10)
    sen = (TP) / (TP + FN + 1e-10)
    return acc, sen
