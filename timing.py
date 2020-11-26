import time
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from efd import s3fd

img = Image.open("./example.jpg")

img = transforms.ToTensor()(img)
if img.shape[0] == 1:
    # Gray => RGB
    img = torch.repeat_interleave(img, 3, 0)
imgs = torch.stack([img] * 16)

model = s3fd(pretrained=True)
model = model.cuda()

start_time = time.time()
bbox_lists, patch_iters = model.detect(imgs, scale_factor=0.5)
print(time.time() - start_time)
