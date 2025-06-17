import torch


class Config:
    downsample_size = (224, 224)
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    slowfast_sample_size = 32
    resnet_sample_size = 1
    T = 10  # seconds batch
    train = True
    load_model = False
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    cache = False