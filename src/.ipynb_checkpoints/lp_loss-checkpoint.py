import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def logloss(Pred,Ptrue,szs,eps=10e-10):
    b,ch,h,w = szs
    Pred = torch.clamp(Pred,min=eps,max=1.)
    Pred = -torch.log(Pred)
    Pred = Pred*Ptrue
    Pred = Pred.reshape(b,h*w*ch)
    Pred = torch.sum(Pred,1)
    return Pred

def l1(pred,true,szs):
    b,ch,h,w = szs
    res = (true-pred).reshape(b,h*w*ch)
    res = torch.abs(res)
    res = torch.sum(res,1)
    return res

class LPD_Loss(nn.Module):
    def __init__(self):
        super(LPD_Loss, self).__init__()
        
    def forward(self, pred_conf, pred_affine, target_conf, target_pts):
        '''
        pred: batchsize, 8, 13, 13 (the grid size is 13x13. Each cell includes obj, non-obj, affine-matrix)
        target: batchsize, 9, 13, 13 (the grid size is 13x13. Each cell includes obj/non-obj, 4-clockwise-points)
        '''
        b, _, h, w = pred_conf.size()
        
        affinex = torch.stack([torch.clamp(pred_affine[:,0,:,:], min=0.),pred_affine[:,1,:,:],pred_affine[:,2,:,:]], dim=1)
        affiney = torch.stack([pred_affine[:,3,:,:],torch.clamp(pred_affine[:,4,:,:], min=0.),pred_affine[:,5,:,:]], dim=1)
    
        v = 0.5
        base_pts = torch.Tensor([-v,-v,1., v,-v,1., v,v,1., -v,v,1.]).to(pred_conf.device)
        base_pts = base_pts.reshape(1,-1,1,1)
        base_pts = base_pts.expand(b,-1,h,w)
        pts = torch.zeros(b, 0, h, w).to(pred_conf.device)
        
        for i in range(0,12,3):
            row = base_pts[:,i:(i+3),:,:]
            ptsx = torch.sum(affinex*row, 1)
            ptsy = torch.sum(affiney*row, 1)

            pts_xy = torch.stack([ptsx,ptsy], 1)
            pts = (torch.cat([pts,pts_xy], 1))
        
        obj_conf = pred_conf[:, 0, :, :]
        non_obj_conf = pred_conf[:, 1, :, :]
        
        # object loss
        loss1 = logloss(obj_conf, target_conf, (b,1,h,w))
        loss2 = logloss(non_obj_conf, 1.0 - target_conf, (b,1,h,w))
        
        # location loss
#         print(pts[0, :, 5, 5])
#         print(pts[0, :, 5, 6])
        
        pts_shape = (b,2*4,h,w)
        flags = target_conf.unsqueeze(1).expand(pts_shape)
#         print(flags[0, :, 5, 5])
        loss3 = l1(pts * flags, target_pts * flags, pts_shape)
        
        loss = loss1 + loss2 + loss3
#         print(loss1.item(), loss2.item(), loss3.item())
        
        return loss.sum()
        
if __name__ == '__main__':
    pred_affine = torch.rand(1,6,1,1)
    
    print(pred_affine)
    
    affinex = torch.stack([torch.clamp(pred_affine[:,0,:,:], min=0.),pred_affine[:,1,:,:],pred_affine[:,2,:,:]], dim=1)
    
    affiney = torch.stack([pred_affine[:,3,:,:],torch.clamp(pred_affine[:,4,:,:], min=0.),pred_affine[:,5,:,:]], dim=1)
    
    print(affinex)