import torch
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from torch import autocast
from utils import collate_function
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np


class Trainer:
    """
    Class to train the model.
    Built using instructions available @ https://pytorch.org/tutorials/beginner/introyt/trainingyt.html.
    Tensorboard instructions available @ https://pytorch.org/docs/stable/tensorboard.html.
    """
    def __init__(self, model, optimizer, scheduler, dataset, loss_function):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = model.device
        self.loss_function = loss_function
        self.scaler = GradScaler()

        training_length = int(len(dataset) * 0.8)
        training_dataset, validation_dataset = random_split(dataset, [training_length, len(dataset) - training_length])

        print('Training and Validation datasets created')
        print('Training set has {} instances'.format(len(training_dataset)))
        print('Validation set has {} instances'.format(len(validation_dataset)))

        self.training_dataloader = DataLoader(
            training_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=collate_function,
        )
        self.validation_dataloader = DataLoader(
            validation_dataset, 
            batch_size=1,
            shuffle=False,
            collate_fn=collate_function,
        )

    def train_step(self, epoch, tb_writer):
        self.model.train(True)
        running_loss = 0.

        for i, data in enumerate(self.training_dataloader):
            print('Processing training for batch', (i + 1))

            inputs, labels = [
                value[0] for value in data
            ], [
                value[1] for value in data
            ]

            self.optimizer.zero_grad()

            with autocast('cuda'):
                outputs = []
                for input in inputs:
                    outputs.append(self.model(input))

                loss = self.loss_function(outputs, labels)

            self.scaler.scale(loss).backward()

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)
            print('Batch {} training average loss: {} --------------------'.format(i + 1, avg_loss))

            tb_x = epoch * len(self.training_dataloader) + i * len(inputs)
            tb_writer.add_scalar('Loss/train', avg_loss, tb_x)

        return avg_loss

    def val_step(self, epoch, tb_writer):
        self.model.eval()
        running_loss = 0.

        with torch.no_grad():
            for i, data in enumerate(self.validation_dataloader):
                print('Processing validation for batch', (i + 1))

                inputs, labels = [
                    value[0] for value in data
                ], [
                    value[1] for value in data
                ]

                with autocast('cuda'):
                    outputs = []
                    for input in inputs:
                        outputs.append(self.model(input))

                    loss = self.loss_function(outputs, labels)

                running_loss += loss.item()
                avg_loss = running_loss / (i + 1)
                print('Batch {} validation average loss: {} --------------------'.format(i + 1, avg_loss))

                tb_x = epoch * len(self.validation_dataloader) + i * len(inputs)
                tb_writer.add_scalar('Loss/val', avg_loss, tb_x)

        return avg_loss

    def train_and_validate(self, EPOCHS):
        print('Starting training...')

        now = datetime.now()
        timestamp = now.strftime('%Y-%m-%d_%H:%M:%S')
        writer = SummaryWriter('./runs/summaries/DUAL_ATTENTION_LIVENFLX_II_SUMMARY_{}'.format(timestamp))
        best_loss = float('inf')

        print('Timestamp:', timestamp)
        print('Total epochs:', EPOCHS)

        for epoch in range(EPOCHS):
            last_time = datetime.now()
            print('EPOCH {}:'.format(epoch + 1))

            avg_train_loss = self.train_step(epoch, writer)
            avg_val_loss = self.val_step(epoch, writer)

            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_train_loss, 'Validation' : avg_val_loss },
                            epoch + 1)
            writer.flush()

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                model_path = './runs/models/state_dict/DUAL_ATTENTION_LIVENFLX_II_{}_EPOCH_{}'.format(timestamp, epoch)
                torch.save(self.model.state_dict(), model_path)
                print('Model saved to', model_path)

            print('Epoch {} training and validation elapsed time: {}'.format(epoch + 1, datetime.now() - last_time))

            self.scheduler.step()

        print('Training and validation finished!')
        delta = datetime.now() - now
        print(f"Elapsed time: {delta}")
        writer.close()

    def get_performance_metrics(self):
        self.model.eval()
        total_plcc = 0
        total_srcc = 0
        total_rmse = 0

        with torch.no_grad():
            for i, data in enumerate(self.validation_dataloader):
                print('Processing validation for batch', (i + 1))

                inputs, labels = [
                    value[0] for value in data
                ], [
                    value[1] for value in data
                ]

                with autocast('cuda'):
                    outputs = []
                    for input in inputs:
                        outputs.append(self.model(input))

                y_pred = outputs[0][1].squeeze().cpu().numpy()
                y_true = labels[0][1].cpu().numpy()

                plcc = pearsonr(y_pred, y_true)[0]
                srcc = spearmanr(y_pred, y_true)[0]
                rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

                total_plcc += plcc
                total_srcc += srcc
                total_rmse += rmse

                print(f"Video {i}: PLCC: {plcc}, SRCC: {srcc}, RMSE: {rmse}")
                print(f"Average: PLCC: {total_plcc / (i + 1)}, SRCC: {total_srcc / (i + 1)}, RMSE: {total_rmse / (i + 1)}")

    def load_model(self, model_path):
        """
        Load the model from a given path.
        """
        self.model.load_state_dict(torch.load(model_path))
        print('Model loaded from', model_path)