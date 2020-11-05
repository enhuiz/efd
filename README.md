# EasyFD: **Easy** **F**ace **D**etection

This package is based on the [S3FD](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_S3FD_Single_Shot_ICCV_2017_paper.pdf) implementation from [face-alignment](https://github.com/1adrianb/face-alignment).

## Why not face-alignment?

Face alignment is relatively heavy as it incorporate facial landmark detection, thus I made this package for face detection only.

## Installation

```
pip install git+https://github.com/enhuiz/easyfd
```

## Example

```python3
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from easyfd import s3fd

# 1. Open an image.
img = Image.open("./example.jpg")

# 2. Use torchvision to transform it as tensor.
img = transforms.ToTensor()(img)
imgs = torch.stack([img])

# 3. Initialize the s3fd model.
model = s3fd(pretrained=True)
model = model.cuda()

# 4. Detect. The imgs feed to the model will be scaled by scale_factor.
#    Smaller scale_factor make inference faster but less accurate.
#    Notice that the patches are still cropped from the original image.
bbox_lists, patch_iters = model.detect(imgs, scale_factor=0.2)

# 5. Print & plot the results.
print(bbox_lists)
for patch_iter in patch_iters:
    for patch in patch_iter:
        plt.imshow(patch.permute(1, 2, 0).cpu().numpy())
        plt.title(str(patch.shape))
        plt.show()
```

## Credits

1. [face-alignment](https://github.com/1adrianb/face-alignment)
