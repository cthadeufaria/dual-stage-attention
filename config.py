

class Config:
    downsample_size = (224, 224)
    mean = [0.45, 0.45, 0.45] # TODO: check if normalization parameters are correct.
    std = [0.225, 0.225, 0.225]
    slowfast_sample_size = 32
    resnet_sample_size = 1
    T = 10  # seconds batch