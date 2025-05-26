from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from datetime import datetime
from dataset import PickleDataset
from dual_attention import DualAttention
from trainer import Trainer
from loss import Loss
from config import Config as cfg
from torch.utils.tensorboard import SummaryWriter


def main():
    """
    Main function to run the dual-stage attention model.
    Install PyTorch with CUDA using info available @ https://pytorch.org/get-started/locally/.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    writer = SummaryWriter('./runs/summaries/DUAL_ATTENTION_LIVENFLX_II_SUMMARY_{}'.format(timestamp))

    device = cfg.device
    print(f"Device set to: {device}")

    writer = SummaryWriter('./runs/logs/DUAL_ATTENTION_LIVENFLX_II')

    dataset = PickleDataset('./datasets/LIVE_NFLX_Plus')

    dual_attention = DualAttention(device, dataset.dataset.max_duration)

    optimizer=AdamW(dual_attention.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.98)

    trainer = Trainer(
        model=dual_attention,
        optimizer=optimizer,
        scheduler=scheduler,
        dataset=dataset,
        loss_function=Loss().to(device)
    )

    if cfg.load_model:
        trainer.load_model('./runs/models/state_dict/DUAL_ATTENTION_LIVENFLX_II_2025-05-16_17:43:05_EPOCH_23')
    if cfg.train:
        trainer.train_and_validate(EPOCHS=50, writer=writer, timestamp=timestamp)

    trainer.get_performance_metrics()


if __name__ == "__main__":
    main()