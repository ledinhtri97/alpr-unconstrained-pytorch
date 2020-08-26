import torch
import tensorflow as tf
import numpy as np

b, w, h, c = 1, 13, 13, 9
v = 0.5
base = tf.stack([[[[-v,-v,1., v,-v,1., v,v,1., -v,v,1.]]]])
print(base)
base = tf.tile(base,tf.stack([b,h,w,1]))

print(base[0,0,0])
print(base.shape)

row = base[...,0:(0+3)]
print(row[0,0,0])

pts = tf.zeros((b,h,w,0))
print(pts.shape)

print('-'*30)
affine_pred = torch.rand(1,6,1,1)

affinex = torch.stack([torch.clamp(affine_pred[:,0,:,:], min=0.),affine_pred[:,1,:,:],affine_pred[:,2,:,:]], dim=1)
affiney = torch.stack([affine_pred[:,3,:,:],torch.clamp(affine_pred[:,4,:,:], min=0.),affine_pred[:,5,:,:]], dim=1)
    
bt = torch.Tensor([-v,-v,1., v,-v,1., v,v,1., -v,v,1.])
bt = bt.reshape(1,-1,1,1)
bt = bt.expand(b,-1,h,w)
print(bt.size())
print(bt[0,:,0,0])
pts = torch.zeros(b, 0, h, w)

for i in range(0,12,3):
    row = bt[:, i:(i+3), :, :]
    ptsx = torch.sum(affinex*row, 1)
    ptsy = torch.sum(affiney*row, 1)
    
    pts_xy = torch.stack([ptsx,ptsy], 1)
    pts = (torch.cat([pts,pts_xy], 1))
    
    print('#'*30)
    print(pts.size())

flag = torch.ones(1,1,13,13)
pred = torch.ones(1,2,13,13)
print(pred[:,0,:,:].size())
