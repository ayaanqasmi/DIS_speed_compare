from PIL import Image
import io
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from glob import glob
import time
import os

# Device setup
device = 'cpu'

# Load the traced model
model_path = 'model.pt'
model = torch.jit.load(model_path, map_location=device)
model.eval()

# Preprocessing function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Prediction function
def predict(net, inputs_val, shapes_val, device):
    inputs_val = inputs_val.type(torch.FloatTensor).to(device)
    inputs_val_v = Variable(inputs_val, requires_grad=False)
    ds_val = net(inputs_val_v)[0]
    pred_val = ds_val[0][0, :, :, :]
    pred_val = torch.squeeze(F.upsample(torch.unsqueeze(pred_val, 0), (shapes_val[0][0], shapes_val[0][1]), mode='bilinear'))
    pred_val = (pred_val - pred_val.min()) / (pred_val.max() - pred_val.min())
    return pred_val

# Image directory
src_dir = 'images/images'
image_paths = glob(os.path.join(src_dir, '*'))

# Timing inference for each image
times = []
for image_path in image_paths:
    print(f'Processing {image_path} ...')
    input_image = preprocess_image(image_path)
    im = np.array(Image.open(image_path))
    
    start_time = time.time()
    with torch.no_grad():
        _ = predict(model, input_image, [im.shape[0:2]], device)
    end_time = time.time()
    
    inference_time = end_time - start_time
    times.append(inference_time)
    print(f'Time for {image_path}: {inference_time:.4f} seconds')

# Calculate and print average time
if times:
    avg_time = sum(times) / len(times)
    print(f'Average inference time: {avg_time:.4f} seconds')
else:
    print('No images processed.')
