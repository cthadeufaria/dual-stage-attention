import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from dataset import PickleDataset
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

    dataset = PickleDataset('./datasets/LIVE_NFLX_Plus')

    dual_attention = DualAttention(device, dataset.dataset.max_duration)

    optimizer=Adam(dual_attention.parameters(), lr=5e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.98)

    trainer = Trainer(
        model=dual_attention,
        optimizer=optimizer,
        scheduler=scheduler,
        dataset=dataset,
        loss_function=Loss().to(device)        
    )

    trainer.train_and_validate(EPOCHS=30)


if __name__ == "__main__":
    main()