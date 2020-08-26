import os
import numpy as np
import torch
import cv2
import random
from .sampler import augment_sample, labels2output_map
from .label import readShapes, Shape

LABEL = 'license_plate'

class LPD_Dataset(object):
    def __init__(self, root_dir, gt_file, input_dim=208, model_stride=16):
        self.fnames = []
        self.polys = []
        self.input_dim = input_dim
        self.model_stride = model_stride
        
        with open(gt_file, 'r') as f:
            lines = f.readlines()
            lines = [x.strip().split(',') for x in lines]
            
            for lid, line in enumerate(lines):
                objs = line[2:]
                has_lp = False
                lps = []
                for idx, token in enumerate(objs):
                    if token == LABEL:
                        try:
                            lps.append([float(x) for x in objs[idx+1:idx+9]])
                            has_lp = True
                        except:
                            print('Error in line: ', lid)
                
                fname = os.path.join(root_dir, line[0])
                if has_lp and os.path.isfile(fname):
                    self.polys.append(lps)
                    self.fnames.append(fname)
                                    
    def _process_data_item(self, img, label):    
        XX, llp, pts = augment_sample(img, label.pts, self.input_dim)
        YY = labels2output_map(llp, pts, self.input_dim, self.model_stride)
        return XX, YY

    def __getitem__(self, idx):
        img = cv2.imread(self.fnames[idx])
        lps = self.polys[idx]
        
        # get 1 license plate per image
        pts = random.choice(lps)
        
        # change points format: 4,x1,...,x4,y1,...,y4
        pts = pts[::2] + pts[1::2]
        pts = [4] + pts
        pts = ','.join([str(x) for x in pts])
        label = Shape()
        label.read(pts)
        
        # preprocess data
        img, label = self._process_data_item(img, label)
        img = torch.from_numpy(img.transpose(2,0,1))
        label = torch.from_numpy(label.transpose(2,0,1))
        
        return img, label
    
    def __len__(self):
        return len(self.fnames)