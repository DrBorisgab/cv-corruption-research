import numpy as np
from PIL import Image

import torch
import torch.nn as nn

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.metrics import IoU

from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

def calculate_metric(model, num, augs=None, threshold=0.501):
        img_paths = '/images/' + num + '.tif'
        mask_paths = '/masks/mask' + num + '.tif'

        iou = IoU(threshold=threshold)
        scores = {'IoU': [], 'P': [], 'R': [], 'f1': []}
        
        print(img_paths)
        img = np.array(Image.open(img_paths))
        mask = np.array(Image.open(mask_paths))
        
        if augs:
            trans = augs(image=img, mask=mask)
            img, mask = trans['image'], trans['mask']
        
        mask[mask > 0] = 1
        pred = model(img)

        pred[pred > threshold] = 1
        pred[pred != 1] = 0
        
        scores['IoU'].append(iou(pred, mask).item())

        mask, pred = mask.numpy().flatten(), pred.flatten()
        scores['P'].append(precision_score(mask, pred))
        scores['R'].append(recall_score(mask, pred))
        scores['f1'].append(f1_score(mask, pred))
        return scores
