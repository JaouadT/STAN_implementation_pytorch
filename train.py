import torch
import torch.nn as nn
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import albumentations as A
import torch.nn.functional as F
import math
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import pandas as pd

from utils import get_data, mask_load_test, BCEDiceLoss, train_fn, val_fn
from dataset import Dataset

from model import STAN

from collections import OrderedDict


# parse arugements
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--fold', type=int, default=4)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--model_dir', type=str, default='model')
parser.add_argument('--loss', type=str, default='bcediceloss')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--early_stopping_rounds', type=int, default=50)

# parse arguments
args = parser.parse_args()
data_dir = args.data_dir
data_train = os.path.join(data_dir, 'train')
data_test = os.path.join(data_dir, 'test')
fold = args.fold
img_size = args.img_size
model_dir = args.model_dir
loss = args.loss
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
gpu = args.gpu
early_stopping_rounds = args.early_stopping_rounds


# set device
device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
print(device)

# set seed
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

imagesListTrain, makskListTrain, imagesListValid, maskListValid, imagesListTest, makskListTest = get_data(data_train, data_test, fold)
print(f"Number of training images : {len(imagesListTrain)}, Number of training masks: {len(makskListTrain)}")
print(f"Number of images of validation images : {len(imagesListValid)}, number of validation masks: {len(maskListValid)}")
print(f"Number of testing images : {len(imagesListTest)}, Number of testing masks : {len(makskListTest)}")

num_masks_test = len(makskListTest)

masksTest = np.zeros((num_masks_test, img_size, img_size))

transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                #A.ElasticTransform(p=0.5),
                #A.GridDistortion(p=0.5),
                #A.OpticalDistortion(p=0.5),
                #A.ShiftScaleRotate(p=0.5),
            ])

train_dataset = Dataset(imagesListTrain, makskListTrain, transform=transform)
valid_dataset = Dataset(imagesListValid, maskListValid)
test_dataset = Dataset(imagesListTest, makskListTest)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = STAN().to(device)

# define the scaler for mixed precision training
scaler = torch.cuda.amp.GradScaler()

# define the optimizer and the scheduler
optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-7)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)

BCE_dice_loss = BCEDiceLoss().to(device)

# Run the training and validation for the specified number of epochs
train_loss_list = []
val_loss_list = []

best_dice = 0
trigger = 0

log = OrderedDict([
        ('epoch', []),
        ('loss', []),
        ('iou', []),
        ('dice', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])

for epoch in range(epochs):
    print(f"Epoch: {epoch+1}/{epochs}")
    train_log = train_fn(train_loader, model, optimizer, BCE_dice_loss, scaler, device)
    val_log = val_fn(valid_loader, model,  BCE_dice_loss, device)
    # train_loss_list.append(train_loss)
    # val_loss_list.append(val_loss)

    print('loss %.4f - iou %.4f - dice %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f - val_SE %.4f - val_PC %.4f - val_F1 %.4f - val_SP %.4f - val_ACC %.4f'
            % (train_log['loss'], train_log['iou'], train_log['dice'], val_log['loss'], val_log['iou'], val_log['dice'], val_log['SE'],
               val_log['PC'], val_log['F1'], val_log['SP'], val_log['ACC']))

    scheduler.step(val_log['loss'])

    log['epoch'].append(epoch)
    log['loss'].append(train_log['loss'])
    log['iou'].append(train_log['iou'])
    log['dice'].append(train_log['dice'])

    log['val_loss'].append(val_log['loss'])
    log['val_iou'].append(val_log['iou'])
    log['val_dice'].append(val_log['dice'])

    pd.DataFrame(log).to_csv(os.path.join(model_dir, f'stan_f{fold}.csv'), index=False)

    trigger += 1

    if val_log['dice'] > best_dice:
      torch.save(model.state_dict(), os.path.join(model_dir, f'stan_f{fold}.pth'))
      best_dice = val_log['dice']
      print("=> saved best model. Best dice: {}".format(best_dice))
      trigger = 0

    # early stopping
    if trigger >= early_stopping_rounds:
        print("=> early stopping")
        break
    torch.cuda.empty_cache()

