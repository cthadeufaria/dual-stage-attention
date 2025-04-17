import torch

from torch.optim import Adam

from dataset import VideoDataset
from dual_attention import DualAttention
from trainer import Trainer
from loss import Loss


def main():
    """
    Main function to run the dual-stage attention model.
    Install PyTorch with CUDA using info available @ https://pytorch.org/get-started/locally/.
    """ 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device set to: {device}")

    dual_attention = DualAttention(device)

    dataset = VideoDataset('./dual-stage-attention/datasets/LIVE_NFLX_Plus')

    trainer = Trainer(
        model=dual_attention,
        optimizer=Adam(dual_attention.parameters(), lr=0.004),
        dataset=dataset,
        loss_function=Loss().to(device),
    )

    trainer.train(EPOCHS=600)


if __name__ == "__main__":
    main()