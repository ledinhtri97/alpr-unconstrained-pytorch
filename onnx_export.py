# from src.keras_utils import load_model
# import onnx
# import keras2onnx

# onnx_model_name = 'wpod.onnx'

# model = load_model('data/lp-detector/wpod-net_update1.h5')
# onnx_model = keras2onnx.convert_keras(model, model.name)
# onnx.save_model(onnx_model, onnx_model_name)

import torch
import os
from src import WPOD


trained = 'trained_models/WPOD_epoch_299_loss_5.1569.pth'
onnx_file = os.path.basename(trained).split('.')[0]+'.onnx'
model = WPOD()
model.load_state_dict(torch.load(trained))
input = torch.randn(1,3,208,208)
torch.onnx.export(model, input, onnx_file, verbose=True, input_names=['input'], output_names=['conf', 'loc'])
