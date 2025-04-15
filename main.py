import torch

from torch.optim import Adam

from dataset import VideoDataset
from dual_attention import DualAttention
from trainer import Trainer
from loss import Loss


def main():
    """
    Install PyTorch with ROCm Compute Platform using info @ https://pytorch.org/get-started/locally/.
    HIP and ROCm installation instructions for cuda impl. @ https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html.
    Unsupportted GPU Github issue @ https://github.com/ROCm/rocBLAS/issues/1352.
    GPU support and compatibility matrices @
    https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#linux-supported-gpus
    https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility/native_linux/native_linux_compatibility.html
    https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html#architecture-support-compatibility-matrix
    """ 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device set to: {device}")
    
    dual_attention = DualAttention(device)

    dataset = VideoDataset('./dual-stage-attention/datasets/LIVE_NFLX_Plus')

    trainer = Trainer(
        model=dual_attention,
        optimizer=Adam(dual_attention.parameters(), lr=0.001),
        dataset=dataset,
        loss_function=Loss().to(device),
    )

    trainer.train(EPOCHS=600)


if __name__ == "__main__":
    main()