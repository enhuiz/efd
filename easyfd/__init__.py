from torch.utils.model_zoo import load_url
from .s3fd import S3FD

models_urls = {
    "s3fd": "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
}


def s3fd(pretrained=True):
    model = S3FD()
    if pretrained:
        model.load_state_dict(load_url(models_urls["s3fd"]))
    return model
