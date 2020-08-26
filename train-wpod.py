import cv2
import time
import copy
import math
import argparse
import numpy as np
import imgaug.augmenters as iaa

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.optim import lr_scheduler
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader

from src import WPOD, LPD_Loss, SquarePad, LPD_Dataset

torch.backends.cudnn.deterministic = True


def train(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
#                 model.apply(set_bn_eval)
#                 print('set batchnorm eval')
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            if phase == 'val' and epoch % 10 != 0:
                continue

            # Iterate over data.
            for batch_idx, data in enumerate(dataloaders[phase]):
                inputs, labels = data
                
                if epoch < 2 and batch_idx < 5:
                        save_image(make_grid(inputs, 8),
                                   'samples/train-sample-{:02d}-{:02d}.jpg'.format(epoch, batch_idx))
                        
                inputs = inputs.to(device)
                labels = labels.to(device)
        
                target_conf = labels[:,0,:,:]
                target_pts  = labels[:,1:,:,:]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    pred_conf, pred_aff = model(inputs)
                    loss = criterion(pred_conf, pred_aff, target_conf, target_pts) / inputs.size(0)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
#                 print('batch {} loss: {:.4f}'.format(batch_idx, loss.item()))        
#                 exit()
                running_loss += loss.item()
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase])
            
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
        
        if epoch % 10 == 0:
            model_name = 'trained_models/{}_epoch_{}_loss_{:.4f}.pth'.format(
                model.__class__.__name__, epoch, epoch_loss)
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, model_name)
            
        print()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    model_name = 'trained_models/{}_epoch_{}_loss_{:.4f}.pth'.format(
        model.__class__.__name__, epoch, epoch_loss)
    torch.save(best_model_wts, model_name)
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model

def variance_scale(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out', 'fan_avg']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    
    fan = (fan_in + fan_out) / 2.0

    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

def set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()
        
def init_weights(m):
    if type(m) == nn.Conv2d:
        variance_scale(m.weight, 1, 'fan_avg', 'linear')
        nn.init.zeros_(m.bias)
    elif type(m) == nn.BatchNorm2d:
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-rd'        ,'--root-dir'       ,type=str   , required=True  ,help='Dataset directory')
    parser.add_argument('-gt'        ,'--gt-file'        ,type=str   , required=True  ,help='Groundtruth file')
    parser.add_argument('-eps'        ,'--epochs'        ,type=int   , default=300    ,help='Number of  epochs (default = 300)')
    parser.add_argument('-bs'        ,'--batch-size'     ,type=int   , default=32  ,help='Mini-batch size (default = 32)')
    parser.add_argument('-od'        ,'--output-dir'     ,type=str   , default='./'        ,help='Output directory (default = ./)')
    parser.add_argument('-op'        ,'--optimizer'      ,type=str   , default='Adam'    ,help='Optmizer (default = Adam)')
    parser.add_argument('-lr'        ,'--learning-rate'  ,type=float , default=.001        ,help='Optmizer (default = 0.01)')
    parser.add_argument('-pt'        ,'--pretrained-model'  ,type=str ,help='Path to pretrain model')
    args = parser.parse_args()

    
    model = WPOD()
    lp_datasets = {'train': LPD_Dataset(args.root_dir, args.gt_file),
                   'val': LPD_Dataset(args.root_dir, args.gt_file)}
    
    dataset_lens = {x: len(lp_datasets[x]) for x in ['train', 'val']}
    dataloaders = {
        x: DataLoader(lp_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4) 
        for x in ['train', 'val']
    }
    
    for x, v in dataset_lens.items():
        print('{} datasize is {}'.format(x, v))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.apply(init_weights)
    
    if args.pretrained_model:
        sd = torch.load(args.pretrained_model)
        model.load_state_dict(sd)
        print('Load pretrain model from:', args.pretrained_model)
        
    criterion = LPD_Loss()

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    model = train(model, dataloaders, criterion, optimizer, exp_lr_scheduler, device, num_epochs=args.epochs)