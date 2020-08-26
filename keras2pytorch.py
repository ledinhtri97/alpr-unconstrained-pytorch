import src.pytorch_utils as put
import src.keras_utils as kut
from src.wpod_net import WPOD
import torch
from keras import backend as K
from tensorflow.keras.models import Model

keras_model = kut.load_model('data/lp-detector/wpod-net_update1.h5')
pytorch_model = WPOD()
# pytorch_model = put.load_model('trained_models/WPOD_epoch_299_loss_20.4253.pth')
pytorch_model.eval()
keras_names = [weight.name for layer in keras_model.layers for weight in layer.weights]
keras_weights = keras_model.get_weights()
keras_map = {}
for name, weight in zip(keras_names, keras_weights):
    keras_map[name] = weight

pytorch_names = pytorch_model.state_dict().keys()

def copy_weight(m):
    if type(m) == nn.Conv2d:
        m.weight = 0
        m.bias = 0
    elif type(m) == nn.BatchNorm2d:
        m.weight = 0
        m.bias = 0
        m.running_mean = 0
        m.running_var = 0


partial_model = Model(keras_model.inputs, keras_model.get_layer('conv2d_25').output)

# print(len(keras_names))
# print(keras_names)
# print(len(pytorch_names))

print([layer.name for layer in keras_model.layers])

print(pytorch_names)

print(keras_map['conv2d_25/kernel:0'].shape)
print(keras_map['conv2d_26/kernel:0'].shape)

x = torch.load('sample_input.pth')
x = x.cpu().numpy().transpose(0,2,3,1)
print('input')
print(x[0,:3,:3,0])
out = partial_model([x], training=False)
print('output')
print(out.shape)
# print(out[0,:3,:3,0])
print(out[0,0,0,:])



# print(pytorch_model.state_dict()['conf.weight'].size())

pytorch_map = pytorch_model.state_dict()

counter = 0
for pname in pytorch_names:
    if '.0.' in pname:
        name_idx = pname[5:pname.index('.')]
        if 'weight' in pname:
            counter += 1
            weight =  keras_map['conv2d_{}/kernel:0'.format(name_idx)].transpose(3,2,0,1)
            pytorch_map[pname] = torch.tensor(weight, dtype=torch.float)
            
        if 'bias' in pname:
            counter += 1
            bias =  keras_map['conv2d_{}/bias:0'.format(name_idx)]
            pytorch_map[pname] = torch.tensor(bias, dtype=torch.float)
    
    if '.1.' in pname:        
        name_idx = pname[5:pname.index('.')]
        if 'weight' in pname:
            counter += 1
            gamma =  keras_map['batch_normalization_{}/gamma:0'.format(name_idx)]
            pytorch_map[pname] = torch.tensor(gamma, dtype=torch.float)
            
        if 'bias' in pname:
            counter += 1
            beta =  keras_map['batch_normalization_{}/beta:0'.format(name_idx)]
            pytorch_map[pname] = torch.tensor(beta, dtype=torch.float)
        
        if 'running_mean' in pname:
            counter += 1
            mean =  keras_map['batch_normalization_{}/moving_mean:0'.format(name_idx)]
            pytorch_map[pname] = torch.tensor(mean, dtype=torch.float)  
        
        if 'running_var' in pname:
            counter += 1
            var =  keras_map['batch_normalization_{}/moving_variance:0'.format(name_idx)]
            pytorch_map[pname] = torch.tensor(var, dtype=torch.float)
        
        if 'tracking' in pname:
            counter += 1
            
            pytorch_map[pname] = torch.ones_like(pytorch_map[pname], dtype=torch.float)
            
    if 'conf.' in pname:
        print('conf')
        name_idx = 25
        if 'weight' in pname:
            counter += 1
            weight =  keras_map['conv2d_{}/kernel:0'.format(name_idx)].transpose(3,2,0,1)
            pytorch_map[pname] = torch.tensor(weight, dtype=torch.float)
            
        if 'bias' in pname:
            counter += 1
            bias =  keras_map['conv2d_{}/bias:0'.format(name_idx)]
            pytorch_map[pname] = torch.tensor(bias, dtype=torch.float)
            
    if 'loc.' in pname:
        print('loc')
        name_idx = 26
        if 'weight' in pname:
            counter += 1
            weight =  keras_map['conv2d_{}/kernel:0'.format(name_idx)].transpose(3,2,0,1)
            pytorch_map[pname] = torch.tensor(weight, dtype=torch.float)
            
        if 'bias' in pname:
            counter += 1
            bias =  keras_map['conv2d_{}/bias:0'.format(name_idx)]
            pytorch_map[pname] = torch.tensor(bias, dtype=torch.float)

torch.save(pytorch_map, 'wpod_cvt.pth')
print(counter)