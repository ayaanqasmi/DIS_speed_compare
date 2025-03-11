import os
import time
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# Load model
model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])
model.to('cpu')
model.eval()

# Data settings
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Directory containing images
image_dir = "images/images"
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

# Ensure only 10 images are processed
image_files = image_files[:10]

# Store inference times
inference_times = []

# Iterate through images and run inference
for image_file in image_files:
    image = Image.open(image_file).convert('RGB')
    input_images = transform_image(image).unsqueeze(0).to('cpu')
    
    start_time = time.time()
    
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    
    end_time = time.time()
    
    inference_time = end_time - start_time
    inference_times.append(inference_time)
    print(f"Inference time for {os.path.basename(image_file)}: {inference_time:.4f} seconds")

# Calculate and print average inference time
avg_time = sum(inference_times) / len(inference_times)
print(f"Average inference time: {avg_time:.4f} seconds")
