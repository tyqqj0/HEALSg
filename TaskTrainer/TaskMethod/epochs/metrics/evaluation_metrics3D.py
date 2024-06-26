#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║                                                                                                             ║
# ║        __  __                                        ____                __                                 ║
# ║       /\ \/\ \                                      /\  _`\             /\ \  __                            ║
# ║       \ \ \_\ \     __     _____   _____   __  __   \ \ \/\_\    ___    \_\ \/\_\    ___      __            ║
# ║        \ \  _  \  /'__`\  /\ '__`\/\ '__`\/\ \/\ \   \ \ \/_/_  / __`\  /'_` \/\ \ /' _ `\  /'_ `\          ║
# ║         \ \ \ \ \/\ \L\.\_\ \ \L\ \ \ \L\ \ \ \_\ \   \ \ \L\ \/\ \L\ \/\ \L\ \ \ \/\ \/\ \/\ \L\ \         ║
# ║          \ \_\ \_\ \__/.\_\\ \ ,__/\ \ ,__/\/`____ \   \ \____/\ \____/\ \___,_\ \_\ \_\ \_\ \____ \        ║
# ║           \/_/\/_/\/__/\/_/ \ \ \/  \ \ \/  `/___/> \   \/___/  \/___/  \/__,_ /\/_/\/_/\/_/\/___L\ \       ║
# ║                              \ \_\   \ \_\     /\___/                                         /\____/       ║
# ║                               \/_/    \/_/     \/__/                                          \_/__/        ║
# ║                                                                                                             ║
# ║           49  4C 6F 76 65  59 6F 75 2C  42 75 74  59 6F 75  4B 6E 6F 77  4E 6F 74 68 69 6E 67 2E            ║
# ║                                                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# @File   : evaluation_metrics3D.py
import numpy as np
from skimage import filters
# import SimpleITK as sitk
from sklearn import metrics


def AUC_score(SR, GT, threshold=0.5):
    # SR = SR.numpy()
    GT = GT.ravel()  # we want to make them into vectors
    SR = SR.ravel()  # .detach()
    # fpr, tpr, _ = metrics.roc_curve(GT, SR)
    # fpr, tpr, _ = metrics.roc_curve(SR, GT)
    # roc_auc = metrics.auc(fpr, tpr)
    roc_auc = metrics.roc_auc_score(GT, SR)
    return roc_auc


def numeric_score(pred, gt):
    FP = np.float64(np.sum((pred == 255) & (gt == 0)))
    FN = np.float64(np.sum((pred == 0) & (gt == 255)))
    TP = np.float64(np.sum((pred == 255) & (gt == 255)))
    TN = np.float64(np.sum((pred == 0) & (gt == 0)))
    # print('FP,FN,TP,TN',FP,FN,TP,TN)
    return FP, FN, TP, TN


def Dice(pred, gt):
    pred = np.int64(pred / 255)
    gt = np.int64(gt / 255)
    dice = np.sum(pred[gt == 1]) * 2.0 / (np.sum(pred) + np.sum(gt))
    return dice


def IoU(pred, gt):
    pred = np.int64(pred / 255)
    gt = np.int64(gt / 255)
    m1 = np.sum(pred[gt == 1])
    m2 = np.sum(pred == 1) + np.sum(gt == 1) - m1
    iou = m1 / m2
    return iou


def metrics_3d(pred, gt):
    # auc = AUC_score(pred/255, gt)
    # print('pred and gt',pred,gt)
    # pred = (pred.data.numpy() * 255).astype(np.uint8)
    # # # input("wait2..")
    # gt = (gt.data.numpy() * 255).astype(np.uint8)
    pred1 = np.sum(pred)
    # print(pred.shape, gt.shape)

    if pred1 == 0:
        print('pred:', pred)
    else:
        # print('pred1:',pred)
        threshold = filters.threshold_otsu(pred, nbins=255)
    #
    # # print('threshold',threshold)
        pred = np.where(pred > threshold, 255.0, 0)

    # print(pred.shape)
#     ###########add##
# def threshold(image):
#     image[image >= 100] = 255
#     image[image < 100] = 0
#     return image
# def metrics_3d1(pred, gt):
    
#     pred1 = np.sum(pred)
#     if pred1 == 0:
#         print('pred:', pred)
#     else:
#         pred = threshold(pred)
#     # print(pred.shape)
#     ##########
#     print(type(pred))
    FP, FN, TP, TN = numeric_score(pred, gt)
    acc = (TP + TN) / (TP + FP + FN + TN + 1e-10)
    sen = TP / (TP + FN + 1e-10)  # recall sensitivity
    spe = TN / (TN + FP + 1e-10)
    iou = TP / (TP + FN + FP + 1e-10)
    dsc = 2.0 * TP / (TP * 2.0 + FN + FP + 1e-10)
    pre = TP / (TP + FP + 1e-10)
    # f1_score = (2.0*pre*sen)/(pre+sen+ 1e-10)

    return acc, sen, spe, iou, dsc, pre


def over_rate(pred, gt):
    # pred = np.int64(pred / 255)
    # gt = np.int64(gt / 255)
    Rs = np.float(np.sum(gt == 255))
    Os = np.float(np.sum((pred == 255) & (gt == 0)))
    OR = Os / (Rs + Os)
    return OR


def under_rate(pred, gt):
    # pred = np.int64(pred / 255)
    # gt = np.int64(gt / 255)
    Rs = np.float(np.sum(gt == 255))
    Us = np.float(np.sum((pred == 0) & (gt == 255)))
    Os = np.float(np.sum((pred == 255) & (gt == 0)))
    UR = Us / (Rs + Os)
    return UR
