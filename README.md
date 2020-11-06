# EasyFD: **Easy** **F**ace **D**etection

This package is based on the [S3FD](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_S3FD_Single_Shot_ICCV_2017_paper.pdf) implementation from [face-alignment](https://github.com/1adrianb/face-alignment).

## Why not face-alignment?

Face alignment is relatively heavy as it incorporate facial landmark detection, and I have encountered some performance issue when using the S3FD detector from face-alignment during decoding stage due to the implementation. To make thing faster and easier, I made this package for face detection only and fix some performance problem of the original implementation of decoding.

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
if img.shape[0] == 1:
    # Gray => RGB
    img = torch.repeat_interleave(img, 3, 0)
imgs = torch.stack([img])

# 3. Initialize the s3fd model.
model = s3fd(pretrained=True)
model = model.cuda()

# 4. Detect. The imgs feed to the model will be scaled by scale_factor.
#    Smaller scale_factor make inference faster but less accurate.
#    Notice that the patches are still cropped from the original image.
bbox_lists, patch_iters = model.detect(imgs, scale_factor=0.5)

# 5. Print & plot the results.
print(bbox_lists)
for patch_iter in patch_iters:
    for patch in patch_iter:
        plt.imshow(patch.permute(1, 2, 0).cpu().numpy())
        plt.title(str(patch.shape))
        plt.show()
```

## Comparison

| commit                                                       | Time   |
| ------------------------------------------------------------ | ------ |
| git checkout 04eac0a (from face-alignment, pytorch decoding) | 5.8595 |
| git checkout master (numpy based decoding)                   | 1.0739 |

This implementation is around 5.5x faster.

## Credits

1. [face-alignment](https://github.com/1adrianb/face-alignment)
2. [Example image](https://upload.wikimedia.org/wikipedia/commons/d/df/The_Fabs.JPG)
