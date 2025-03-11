from PIL import Image
import torch
from torchvision import transforms
import os
from glob import glob
import time

#Load BiRefNet model
from transformers import AutoModelForImageSegmentation
birefnet = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)


# birefnet = BiRefNet(bb_pretrained=False)
# state_dict = torch.load('../BiRefNet-general-epoch_244.pth', map_location='cpu')
# state_dict = check_state_dict(state_dict)
# birefnet.load_state_dict(state_dict)

device = 'cpu'
torch.set_float32_matmul_precision('high')

birefnet.to(device)
birefnet.eval()
print('BiRefNet is ready to use.')

# Input Data transformation
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Image directory
src_dir = '../images/images'
image_paths = glob(os.path.join(src_dir, '*'))

# Timing inference for each image
times = []
for image_path in image_paths:
    print(f'Processing {image_path} ...')
    image = Image.open(image_path).convert('RGB')
    input_images = transform_image(image).unsqueeze(0).to(device)

    # Time inference
    start_time = time.time()
    with torch.no_grad():
        _ = birefnet(input_images)[-1].sigmoid().cpu()
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
