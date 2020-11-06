"""
BSD 3-Clause License

Copyright (c) 2017, Adrian Bulat
Copyright (c) 2020, enhuiz
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import nms


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale=1.0):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.weight.data *= 0.0
        self.weight.data += self.scale

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = x / norm * self.weight.view(1, -1, 1, 1)
        return x


class S3FD(nn.Module):
    def __init__(self):
        super(S3FD, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.fc6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=3)
        self.fc7 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)

        self.conv6_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv6_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv7_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.conv7_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv3_3_norm = L2Norm(256, scale=10)
        self.conv4_3_norm = L2Norm(512, scale=8)
        self.conv5_3_norm = L2Norm(512, scale=5)

        self.conv3_3_norm_mbox_conf = nn.Conv2d(
            256, 4, kernel_size=3, stride=1, padding=1
        )
        self.conv3_3_norm_mbox_loc = nn.Conv2d(
            256, 4, kernel_size=3, stride=1, padding=1
        )
        self.conv4_3_norm_mbox_conf = nn.Conv2d(
            512, 2, kernel_size=3, stride=1, padding=1
        )
        self.conv4_3_norm_mbox_loc = nn.Conv2d(
            512, 4, kernel_size=3, stride=1, padding=1
        )
        self.conv5_3_norm_mbox_conf = nn.Conv2d(
            512, 2, kernel_size=3, stride=1, padding=1
        )
        self.conv5_3_norm_mbox_loc = nn.Conv2d(
            512, 4, kernel_size=3, stride=1, padding=1
        )

        self.fc7_mbox_conf = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)
        self.fc7_mbox_loc = nn.Conv2d(1024, 4, kernel_size=3, stride=1, padding=1)
        self.conv6_2_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv6_2_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        self.conv7_2_mbox_conf = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)
        self.conv7_2_mbox_loc = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        f3_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        f4_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        f5_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        ffc7 = h
        h = F.relu(self.conv6_1(h))
        h = F.relu(self.conv6_2(h))
        f6_2 = h
        h = F.relu(self.conv7_1(h))
        h = F.relu(self.conv7_2(h))
        f7_2 = h

        f3_3 = self.conv3_3_norm(f3_3)
        f4_3 = self.conv4_3_norm(f4_3)
        f5_3 = self.conv5_3_norm(f5_3)

        cls1 = self.conv3_3_norm_mbox_conf(f3_3)
        reg1 = self.conv3_3_norm_mbox_loc(f3_3)
        cls2 = self.conv4_3_norm_mbox_conf(f4_3)
        reg2 = self.conv4_3_norm_mbox_loc(f4_3)
        cls3 = self.conv5_3_norm_mbox_conf(f5_3)
        reg3 = self.conv5_3_norm_mbox_loc(f5_3)
        cls4 = self.fc7_mbox_conf(ffc7)
        reg4 = self.fc7_mbox_loc(ffc7)
        cls5 = self.conv6_2_mbox_conf(f6_2)
        reg5 = self.conv6_2_mbox_loc(f6_2)
        cls6 = self.conv7_2_mbox_conf(f7_2)
        reg6 = self.conv7_2_mbox_loc(f7_2)

        # max-out background label
        chunk = torch.chunk(cls1, 4, 1)
        bmax = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
        cls1 = torch.cat([bmax, chunk[3]], dim=1)

        return [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6]

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def decode(locations, priors, variances):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            locations (tensor): location predictions for locations layers,
                Shape: [num_priors, 4]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors, 4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded bounding box predictions
        """
        boxes = np.concatenate(
            [
                priors[:, :2] + locations[:, :2] * variances[0] * priors[:, 2:],
                priors[:, 2:] * np.exp(locations[:, 2:] * variances[1]),
            ],
            axis=1,
        )
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def crop_patches(self, img, bbox_list):
        for bbox in bbox_list:
            x1, y1, x2, y2 = map(int, np.clip(bbox[:4], 0, None))
            yield img[:, y1:y2, x1:x2]

    @torch.no_grad()
    def detect(self, imgs, scale_factor=1, threshold=0.05):
        """
        Args:
            imgs: (b c h w)
            scale_factor: [0, 1], network input scale
            threshold: [0, 1], minimal probability for a detected bbox.
        """
        x = imgs.clone()
        x = x.to(self.device)
        x = F.interpolate(x, scale_factor=scale_factor, recompute_scale_factor=False)

        olist = self.forward(x * 255.0)
        for i in range(0, len(olist), 2):
            olist[i] = F.softmax(olist[i], dim=1)
        olist = [oelem.cpu().numpy() for oelem in olist]

        bbox_lists = []
        patch_iters = []

        for i in range(len(olist[0])):
            scores = []
            priors = []
            locations = []

            for j in range(0, len(olist), 2):
                ocls, oreg = olist[j], olist[j + 1]
                stride = 2 ** (j // 2 + 2)  # 4, 8, 16, 32, 64, 128
                possible = list(zip(*np.where(ocls[i, 1, :, :] > threshold)))
                for h, w in possible:
                    axc = stride / 2 + w * stride
                    ayc = stride / 2 + h * stride
                    scores.append([ocls[i, 1, h, w]])
                    priors.append([axc, ayc, stride * 4, stride * 4])
                    locations.append(oreg[i, :, h, w].flatten())

            if len(scores) > 0:
                scores = np.array(scores)
                priors = np.array(priors)
                locations = np.array(locations)
                variances = [0.1, 0.2]

                bbox_list = self.decode(locations, priors, variances)
                bbox_list /= scale_factor

                bbox_list = np.concatenate((bbox_list, scores), axis=-1)

                bbox_list = bbox_list[nms(bbox_list, 0.3)]
                bbox_list = bbox_list[np.where(bbox_list[:, -1] > 0.5)]
            else:
                bbox_list = []

            bbox_lists.append(bbox_list)
            patch_iters.append(self.crop_patches(imgs[i], bbox_list))

        bbox_lists = [
            pd.DataFrame(bbox_list, columns=["x1", "y1", "x2", "y2", "score"])
            if len(bbox_list) > 0
            else bbox_list
            for bbox_list in bbox_lists
        ]

        return bbox_lists, patch_iters
