import os 
from glob import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

    
from collections import OrderedDict

def get_data(data_train, data_test, fold):

    imagesListTrain = []
    makskListTrain = []

    for idx in range(5):
        if idx == fold:
            imagesListValid = glob(f"{data_train}/split{fold}/images"+'/*.png')
            maskListValid = glob(f"{data_train}/split{fold}/masks"+'/*.png')
        else:

            for img in glob(f'{data_train}/split{idx}/images'+'/*.png'):
                imagesListTrain.append(img)
            for msk in glob(f'{data_train}/split{idx}/masks'+'/*.png'):
                makskListTrain.append(msk)

    imagesListTest = glob(os.path.join(data_test, 'images', '*.png'))
    makskListTest = glob(os.path.join(data_test, 'masks', '*.png'))

    return imagesListTrain, makskListTrain, imagesListValid, maskListValid, imagesListTest, makskListTest

def mask_load_test(dir_path, imgs_list, masks_array, img_size):
    for i in range(len(imgs_list)):
        # tmp_img = Image.open(os.path.join(dir_path, imgs_list[i])).resize((256, 256))
        img = cv2.imread(os.path.join(dir_path, imgs_list[i]))
        img = cv2.resize(img, (img_size, img_size))
        img = np.array(img)
        img = img.transpose(2, 0, 1)
        masks_array[i] = img[0,:,:]/255.0

    # Expand the dimensions of the arrays
    masks_array = np.expand_dims(masks_array, axis=3)
    return masks_array

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)
    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
        # TP : True Positive
        # FN : False Negative
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FN = ((SR == 0).byte() + (GT == 1).byte()) == 2
    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SP = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
        # TN : True Negative
        # FP : False Positive
    TN = ((SR == 0).byte() + (GT == 0).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    return SP

def get_precision(SR,GT,threshold=0.5):
    PC = 0
    SR = SR > threshold
    GT = GT== torch.max(GT)
        # TP : True Positive
        # FP : False Positive
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)
    return PC

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)

    output_ = torch.tensor(output_)
    target_=torch.tensor(target_)
    SE = get_sensitivity(output_,target_,threshold=0.5)
    PC = get_precision(output_,target_,threshold=0.5)
    SP= get_specificity(output_,target_,threshold=0.5)
    ACC=get_accuracy(output_,target_,threshold=0.5)
    F1 = 2*SE*PC/(SE+PC + 1e-6)
    return iou, dice , SE, PC, F1,SP,ACC

def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

# Define the training function without the mixed precision
def train_fn(loader, model, optimizer, loss_fn, scaler, DEVICE):
    loop = tqdm(loader)

    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter(),
                   'SE':AverageMeter(),
                   'PC':AverageMeter(),
                   'F1':AverageMeter(),
                   'SP':AverageMeter(),
                   'ACC':AverageMeter()
                   }

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE, dtype=torch.float)
        targets = targets.float().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            iou, dice, SE, PC, F1, SP, ACC = iou_score(predictions, targets)

        # backward
        optimizer.zero_grad()
        # loss.backward()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        avg_meters['loss'].update(loss.item(), data.size(0))
        avg_meters['iou'].update(iou, data.size(0))
        avg_meters['dice'].update(dice, data.size(0))
        avg_meters['SE'].update(SE, data.size(0))
        avg_meters['PC'].update(PC, data.size(0))
        avg_meters['F1'].update(F1, data.size(0))
        avg_meters['SP'].update(SP, data.size(0))
        avg_meters['ACC'].update(ACC, data.size(0))

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('SE', avg_meters['SE'].avg),
                        ('PC', avg_meters['PC'].avg),
                        ('F1', avg_meters['F1'].avg),
                        ('SP', avg_meters['SP'].avg),
                        ('ACC', avg_meters['ACC'].avg)
                        ])

def val_fn(loader, model, loss_fn, DEVICE):

    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter(),
                   'SE':AverageMeter(),
                   'PC':AverageMeter(),
                   'F1':AverageMeter(),
                   'SP':AverageMeter(),
                   'ACC':AverageMeter()
                   }

    loop = tqdm(loader)
    model.eval()
    fin_loss = 0
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE, dtype=torch.float)
            targets = targets.float().to(device=DEVICE)
            predictions = model(data)

            loss = loss_fn(predictions, targets)

            iou, dice, SE, PC, F1, SP, ACC = iou_score(predictions, targets)
            avg_meters['loss'].update(loss.item(), data.size(0))
            avg_meters['iou'].update(iou, data.size(0))
            avg_meters['dice'].update(dice, data.size(0))
            avg_meters['SE'].update(SE, data.size(0))
            avg_meters['PC'].update(PC, data.size(0))
            avg_meters['F1'].update(F1, data.size(0))
            avg_meters['SP'].update(SP, data.size(0))
            avg_meters['ACC'].update(ACC, data.size(0))

            fin_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('SE', avg_meters['SE'].avg),
                        ('PC', avg_meters['PC'].avg),
                        ('F1', avg_meters['F1'].avg),
                        ('SP', avg_meters['SP'].avg),
                        ('ACC', avg_meters['ACC'].avg)
                        ])
    #return fin_loss / len(loader)
